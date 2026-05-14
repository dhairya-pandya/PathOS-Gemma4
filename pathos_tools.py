"""
PathOS Tools — Diagnostic Toolkit
==================================
Implements the 5 core agentic tools for the PathOS pipeline.
These are LOCAL functions (no model calls) that process model
outputs into structured clinical artifacts.

Tools:
    1. analyze_patch      — parse model vision output into structured findings
    2. flag_malignancy    — rule-based malignancy risk scoring
    3. suggest_special_stains — recommend IHC / special stains
    4. compare_to_atlas   — Jaccard similarity against TISSUE_KB
    5. generate_report    — compile final PATHOS LAB REPORT

Used by: pathos_inference.py (agentic pipeline)
"""

import re
import json
import datetime
from pathlib import Path

# ============================================================
# TISSUE KNOWLEDGE BASE
# ============================================================

TISSUE_KB = {
    "TUM": {
        "name": "colorectal adenocarcinoma",
        "code": "TUM",
        "features": [
            "irregular glands", "nuclear pleomorphism",
            "hyperchromatic nuclei", "mitotic figures",
            "loss of polarity", "cribriform pattern",
        ],
        "clinical": "malignant; requires oncology staging",
        "risk": "high",
    },
    "LYM": {
        "name": "lymphocytic infiltrate",
        "code": "LYM",
        "features": [
            "dense small round cells", "condensed nuclei",
            "scant cytoplasm", "perivascular cuffing",
        ],
        "clinical": "immune response; exclude lymphoma",
        "risk": "moderate",
    },
    "STR": {
        "name": "cancer-associated stroma",
        "code": "STR",
        "features": [
            "desmoplastic spindle fibroblasts", "dense collagen",
            "myofibroblasts", "reactive stroma",
        ],
        "clinical": "tumor microenvironment; associated with invasion",
        "risk": "moderate",
    },
    "ADI": {
        "name": "adipose tissue",
        "code": "ADI",
        "features": [
            "univacuolated lipid droplets", "peripheral nuclei",
            "thin fibrous septa", "signet-ring appearance",
        ],
        "clinical": "pericolonic fat; assess for infiltration",
        "risk": "low",
    },
    "MUC": {
        "name": "mucinous component",
        "code": "MUC",
        "features": [
            "extracellular mucin pools", "floating epithelial clusters",
            "signet ring cells", "acellular mucin lakes",
        ],
        "clinical": "mucinous adenocarcinoma if tumor-associated",
        "risk": "moderate",
    },
    "MUS": {
        "name": "smooth muscle (muscularis)",
        "code": "MUS",
        "features": [
            "elongated spindle cells", "cigar-shaped nuclei",
            "intersecting fascicles", "eosinophilic cytoplasm",
        ],
        "clinical": "muscularis propria — critical for invasion staging",
        "risk": "low",
    },
    "NORM": {
        "name": "normal colon mucosa",
        "code": "NORM",
        "features": [
            "regular crypts", "columnar epithelium",
            "goblet cells", "intact basement membrane",
        ],
        "clinical": "no pathological abnormality",
        "risk": "low",
    },
    "DEB": {
        "name": "necrotic debris",
        "code": "DEB",
        "features": [
            "ghost cell outlines", "karyolysis",
            "karyorrhexis", "acellular eosinophilic material",
        ],
        "clinical": "tumor necrosis; common in high-grade tumors",
        "risk": "moderate",
    },
    "BACK": {
        "name": "background (non-tissue)",
        "code": "BACK",
        "features": ["glass slide", "no cellular elements"],
        "clinical": "non-diagnostic region",
        "risk": "low",
    },
}

# Integer label → code mapping (NCT-CRC dataset order)
LABEL_MAP = {
    0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC",
    5: "MUS", 6: "NORM", 7: "STR",  8: "TUM",
}

# ============================================================
# STAIN DATABASE
# ============================================================

STAIN_DB = {
    "adenocarcinoma": [
        {"stain": "CDX2 IHC",           "rationale": "Confirms colorectal origin"},
        {"stain": "CK20 / CK7 panel",   "rationale": "Distinguishes primary vs metastatic"},
        {"stain": "MSI / MMR panel",     "rationale": "Lynch syndrome screening — MLH1, MSH2, MSH6, PMS2"},
        {"stain": "KRAS mutation",       "rationale": "Targeted therapy eligibility"},
        {"stain": "Ki-67",               "rationale": "Proliferation index"},
    ],
    "lymphoma": [
        {"stain": "CD20 / CD3",          "rationale": "B-cell vs T-cell lineage"},
        {"stain": "CD10 / BCL-6 / BCL-2","rationale": "Follicular lymphoma workup"},
        {"stain": "Ki-67",               "rationale": "Proliferation index / grade"},
    ],
    "mucinous": [
        {"stain": "PAS / Alcian blue",   "rationale": "Confirm mucin presence"},
        {"stain": "CDX2 IHC",            "rationale": "Colorectal origin"},
        {"stain": "MSI / MMR panel",     "rationale": "Mucinous CRC association with MSI-high"},
    ],
    "necrosis": [
        {"stain": "TUNEL assay",         "rationale": "Apoptosis vs necrosis differentiation"},
        {"stain": "Ki-67",               "rationale": "Viable tumor proliferation at margins"},
    ],
    "stromal": [
        {"stain": "SMA (smooth muscle actin)", "rationale": "Myofibroblast identification"},
        {"stain": "Desmin",              "rationale": "Muscle differentiation"},
        {"stain": "Masson's trichrome",  "rationale": "Collagen vs muscle delineation"},
    ],
    "default": [
        {"stain": "PAS",                 "rationale": "General carbohydrate / basement membrane"},
        {"stain": "Ki-67",               "rationale": "Proliferation index"},
    ],
}

# High-risk morphological keywords
MALIGNANCY_KEYWORDS = {
    "high":     ["carcinoma", "adenocarcinoma", "malignant", "invasive",
                 "pleomorphism", "mitotic", "mitoses", "hyperchromatic",
                 "neoplastic", "dysplasia", "high-grade", "undifferentiated"],
    "moderate": ["atypical", "irregular", "desmoplastic", "necrosis",
                 "mucin", "lymphocytic", "infiltrate", "reactive"],
    "low":      ["normal", "benign", "adipose", "muscle", "background",
                 "regular", "intact", "unremarkable"],
}


# ============================================================
# TOOL 1: analyze_patch
# ============================================================

def analyze_patch(
    model_output: str,
    image_path: str = "",
    magnification: str = "20x",
) -> dict:
    """
    Parse the model's free-text vision output into structured findings.

    Args:
        model_output: Raw text from the PathOS model describing the image.
        image_path:   Path to the original image (for metadata).
        magnification: Slide magnification level.

    Returns:
        dict with tissue_type, features, morphology summary, and raw output.
    """
    output_lower = model_output.lower()

    # Try to match against known tissue types
    detected_tissue = None
    best_score = 0
    for code, info in TISSUE_KB.items():
        score = sum(1 for f in info["features"] if f.lower() in output_lower)
        if info["name"].lower() in output_lower:
            score += 3  # strong boost for exact name match
        if score > best_score:
            best_score = score
            detected_tissue = code

    if detected_tissue is None:
        detected_tissue = "NORM"  # fallback

    tissue_info = TISSUE_KB[detected_tissue]

    # Extract observed features
    observed = [f for f in tissue_info["features"] if f.lower() in output_lower]

    return {
        "tissue_type":  detected_tissue,
        "tissue_name":  tissue_info["name"],
        "observed_features": observed,
        "feature_match_score": best_score,
        "magnification": magnification,
        "image":        Path(image_path).name if image_path else "unknown",
        "raw_output":   model_output[:500],
    }


# ============================================================
# TOOL 2: flag_malignancy
# ============================================================

def flag_malignancy(findings: str, tissue_type: str = "") -> dict:
    """
    Score malignancy risk from morphological findings.
    Uses keyword matching against MALIGNANCY_KEYWORDS + TISSUE_KB risk levels.

    Args:
        findings:    Text description of morphological findings.
        tissue_type: Tissue class code (e.g., "TUM", "NORM").

    Returns:
        dict with risk_level, score, matched indicators, and recommendation.
    """
    findings_lower = findings.lower()

    high_hits = [k for k in MALIGNANCY_KEYWORDS["high"] if k in findings_lower]
    mod_hits  = [k for k in MALIGNANCY_KEYWORDS["moderate"] if k in findings_lower]
    low_hits  = [k for k in MALIGNANCY_KEYWORDS["low"] if k in findings_lower]

    # Weighted scoring
    score = len(high_hits) * 3 + len(mod_hits) * 1 - len(low_hits) * 1

    # Tissue KB risk override
    if tissue_type in TISSUE_KB:
        kb_risk = TISSUE_KB[tissue_type]["risk"]
        if kb_risk == "high":
            score += 3
        elif kb_risk == "moderate":
            score += 1

    # Classify
    if score >= 5:
        risk = "HIGH"
        recommendation = "URGENT: Recommend immediate pathologist review and staging workup."
    elif score >= 2:
        risk = "MODERATE"
        recommendation = "Flag for priority review. Consider additional IHC staining."
    else:
        risk = "LOW"
        recommendation = "Routine review. No immediate concerns identified."

    return {
        "risk_level":     risk,
        "risk_score":     score,
        "high_indicators": high_hits,
        "moderate_indicators": mod_hits,
        "low_indicators": low_hits,
        "recommendation": recommendation,
    }


# ============================================================
# TOOL 3: suggest_special_stains
# ============================================================

def suggest_special_stains(
    preliminary_diagnosis: str,
    clinical_context: str = "",
) -> dict:
    """
    Recommend additional stains or IHC markers based on H&E findings.

    Args:
        preliminary_diagnosis: Working diagnosis from H&E analysis.
        clinical_context: Optional patient clinical history.

    Returns:
        dict with ordered list of recommended stains and rationales.
    """
    diag_lower = preliminary_diagnosis.lower()

    # Match against stain database
    matched_category = "default"
    for category in STAIN_DB:
        if category in diag_lower:
            matched_category = category
            break

    # Additional keyword matching for edge cases
    if any(k in diag_lower for k in ["tumor", "carcinoma", "malignant", "cancer"]):
        matched_category = "adenocarcinoma"
    elif any(k in diag_lower for k in ["lymph", "lymphocyt"]):
        matched_category = "lymphoma"
    elif any(k in diag_lower for k in ["mucin", "mucinous"]):
        matched_category = "mucinous"
    elif any(k in diag_lower for k in ["necro", "debris"]):
        matched_category = "necrosis"
    elif any(k in diag_lower for k in ["strom", "desmoplast", "fibro"]):
        matched_category = "stromal"

    stains = STAIN_DB[matched_category]

    return {
        "matched_category":  matched_category,
        "diagnosis_input":   preliminary_diagnosis,
        "recommended_stains": stains,
        "clinical_context":  clinical_context or "Not provided",
        "note": "These are AI-suggested recommendations. Final stain selection is at the pathologist's discretion.",
    }


# ============================================================
# TOOL 4: compare_to_atlas
# ============================================================

def compare_to_atlas(
    tissue_class: str,
    observed_features: list[str],
) -> dict:
    """
    Compare observed features to reference pathology atlas entries
    using Jaccard similarity.

    Args:
        tissue_class:      NCT-CRC class code or tissue name.
        observed_features: List of observed morphological features.

    Returns:
        dict with ranked matches, similarity scores, and best match.
    """
    observed_set = set(f.lower().strip() for f in observed_features)

    if not observed_set:
        return {
            "best_match": None,
            "similarity": 0.0,
            "ranked_matches": [],
            "note": "No features provided for comparison.",
        }

    rankings = []
    for code, info in TISSUE_KB.items():
        ref_set = set(f.lower() for f in info["features"])

        # Jaccard similarity
        intersection = observed_set & ref_set
        union = observed_set | ref_set
        jaccard = len(intersection) / len(union) if union else 0.0

        # Partial match boost: check substring containment
        partial = sum(
            1 for obs in observed_set
            for ref in ref_set
            if obs in ref or ref in obs
        )
        boosted = jaccard + (partial * 0.05)

        rankings.append({
            "tissue_code":  code,
            "tissue_name":  info["name"],
            "similarity":   round(min(boosted, 1.0), 3),
            "matched_features": list(intersection),
            "reference_features": info["features"],
        })

    rankings.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "query_class":    tissue_class,
        "query_features": list(observed_set),
        "best_match":     rankings[0] if rankings else None,
        "ranked_matches": rankings[:3],  # top 3
    }


# ============================================================
# TOOL 5: generate_report
# ============================================================

def generate_report(
    tissue_type: str,
    primary_finding: str,
    malignancy_score: str,
    morphology: str = "",
    recommended_workup: list[str] | None = None,
    image_name: str = "",
    confidence: str = "Moderate",
) -> dict:
    """
    Generate a structured PATHOS LAB REPORT from all collected findings.

    Args:
        tissue_type:       Identified tissue type.
        primary_finding:   Main pathological finding.
        malignancy_score:  "low", "moderate", or "high".
        morphology:        Key morphological features observed.
        recommended_workup: List of recommended additional tests.
        image_name:        Source image filename.
        confidence:        Confidence level of the analysis.

    Returns:
        dict with the structured report text and all fields.
    """
    workup = recommended_workup or ["No additional workup recommended"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build pathologist note based on risk
    risk = malignancy_score.lower()
    if risk == "high":
        pathologist_note = (
            "HIGH PRIORITY — Findings consistent with malignancy. "
            "Recommend urgent pathologist review, staging workup, "
            "and multidisciplinary tumor board discussion."
        )
    elif risk == "moderate":
        pathologist_note = (
            "PRIORITY REVIEW — Atypical features identified. "
            "Recommend pathologist review with consideration for "
            "additional IHC staining and clinical correlation."
        )
    else:
        pathologist_note = (
            "ROUTINE — No high-risk features identified. "
            "Standard pathologist sign-off recommended."
        )

    # Format the report
    report_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PATHOS LAB REPORT
  Generated by PathOS — AI Pathologist Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Date:            {timestamp}
  Specimen:        {image_name or 'Histopathology patch'}
  Tissue Type:     {tissue_type}
  Primary Finding: {primary_finding}
  Morphology:      {morphology or 'See detailed analysis'}
  Malignancy Risk: {malignancy_score.upper()}
  Confidence:      {confidence}
  Pathologist Note:{pathologist_note}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recommended Additional Workup:
{chr(10).join(f'    • {w}' for w in workup)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚕ This is an AI-assisted preliminary report.
  Final diagnosis requires pathologist review.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return {
        "report_text":       report_text.strip(),
        "timestamp":         timestamp,
        "tissue_type":       tissue_type,
        "primary_finding":   primary_finding,
        "malignancy_score":  malignancy_score,
        "morphology":        morphology,
        "confidence":        confidence,
        "recommended_workup": workup,
        "pathologist_note":  pathologist_note,
    }


# ============================================================
# TOOL REGISTRY (for agentic dispatch)
# ============================================================

TOOL_REGISTRY = {
    "analyze_patch":          analyze_patch,
    "flag_malignancy":        flag_malignancy,
    "suggest_special_stains": suggest_special_stains,
    "compare_to_atlas":       compare_to_atlas,
    "generate_report":        generate_report,
}

# Tool definitions for Gemma 4 function calling format
TOOL_DEFINITIONS = [
    {
        "name": "analyze_patch",
        "description": "Analyze a histopathology image patch. Returns tissue type, morphological features, and observed abnormalities.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_output":  {"type": "string", "description": "Raw model vision output text"},
                "image_path":    {"type": "string", "description": "Path to the histopathology patch image"},
                "magnification": {"type": "string", "enum": ["10x", "20x", "40x"], "description": "Slide scanning magnification"},
            },
            "required": ["model_output"],
        },
    },
    {
        "name": "flag_malignancy",
        "description": "Score malignancy risk from morphological findings. Returns risk level and key indicators.",
        "parameters": {
            "type": "object",
            "properties": {
                "findings":    {"type": "string", "description": "Morphological findings text"},
                "tissue_type": {"type": "string", "description": "Identified tissue class code"},
            },
            "required": ["findings"],
        },
    },
    {
        "name": "suggest_special_stains",
        "description": "Recommend additional stains or IHC markers based on H&E findings.",
        "parameters": {
            "type": "object",
            "properties": {
                "preliminary_diagnosis": {"type": "string", "description": "Working diagnosis from H&E analysis"},
                "clinical_context":      {"type": "string", "description": "Optional patient clinical history"},
            },
            "required": ["preliminary_diagnosis"],
        },
    },
    {
        "name": "compare_to_atlas",
        "description": "Compare observed features to reference pathology atlas entries using Jaccard similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "tissue_class":      {"type": "string", "description": "NCT-CRC class or tissue category"},
                "observed_features": {"type": "array", "items": {"type": "string"}, "description": "List of observed morphological features"},
            },
            "required": ["tissue_class", "observed_features"],
        },
    },
    {
        "name": "generate_report",
        "description": "Generate a structured PATHOS LAB REPORT from all collected findings.",
        "parameters": {
            "type": "object",
            "properties": {
                "tissue_type":       {"type": "string"},
                "primary_finding":   {"type": "string"},
                "malignancy_score":  {"type": "string", "enum": ["low", "moderate", "high"]},
                "morphology":        {"type": "string"},
                "recommended_workup":{"type": "array", "items": {"type": "string"}},
            },
            "required": ["tissue_type", "primary_finding", "malignancy_score"],
        },
    },
]


def dispatch_tool(tool_name: str, arguments: dict) -> dict:
    """
    Dispatch a tool call by name with given arguments.
    Used by the agentic loop in the inference engine.
    """
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return TOOL_REGISTRY[tool_name](**arguments)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  PathOS Tools — Self-Test")
    print("=" * 55)

    # Test analyze_patch
    mock_output = "The image shows irregular glands with nuclear pleomorphism and hyperchromatic nuclei consistent with colorectal adenocarcinoma."
    result = analyze_patch(mock_output, "test_slide.png")
    print(f"\n[analyze_patch] tissue={result['tissue_type']}, features={result['observed_features']}")

    # Test flag_malignancy
    risk = flag_malignancy(mock_output, "TUM")
    print(f"[flag_malignancy] risk={risk['risk_level']}, score={risk['risk_score']}, indicators={risk['high_indicators']}")

    # Test suggest_special_stains
    stains = suggest_special_stains("colorectal adenocarcinoma", "65M, weight loss")
    print(f"[suggest_stains] category={stains['matched_category']}, stains={[s['stain'] for s in stains['recommended_stains']]}")

    # Test compare_to_atlas
    atlas = compare_to_atlas("TUM", ["irregular glands", "nuclear pleomorphism", "mitotic figures"])
    print(f"[compare_atlas] best={atlas['best_match']['tissue_name']}, sim={atlas['best_match']['similarity']}")

    # Test generate_report
    report = generate_report(
        tissue_type="colorectal adenocarcinoma",
        primary_finding="Invasive adenocarcinoma, moderately differentiated",
        malignancy_score="high",
        morphology="Irregular glands, nuclear pleomorphism, mitotic figures",
        recommended_workup=["MSI / MMR panel", "KRAS mutation analysis", "CDX2 IHC"],
        image_name="test_slide.png",
    )
    print(f"\n{report['report_text']}")

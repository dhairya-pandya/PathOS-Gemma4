"""
PathOS Inference Engine v2 — Hybrid Pipeline
=============================================
Deploys the PathOS GGUF model via Ollama with 6 inference-time
accuracy-boosting techniques. No retraining needed.

Models:
    GGUF:    ByteKnight28/pathos-gemma4-histopathology-rl-GGUF
    Adapter: dhairyapandya/pathos-gemma4-histopathology-rl

Usage:
    ollama run hf.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF:F16

    python pathos_inference_engine.py --image slide.png --question "Is carcinoma present?"
    python pathos_inference_engine.py --image slide.png --report
    python pathos_inference_engine.py --interactive

Requirements:
    pip install ollama chromadb sentence-transformers pillow numpy
    ollama must be running: `ollama serve`
"""

import argparse
import base64
import json
import re
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import ollama
from PIL import Image

from pathos_tools import (
    TISSUE_KB, TOOL_DEFINITIONS,
    analyze_patch, flag_malignancy, suggest_special_stains,
    compare_to_atlas, generate_report, dispatch_tool,
)

# ── Optional RAG deps ────────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("[WARN] chromadb / sentence-transformers not installed — RAG disabled.")

# ============================================================
# CONFIG
# ============================================================

# Ollama model name — local Modelfile build
# Build locally:  ollama create pathos -f Modelfile
# Or pull from HF: ollama run hf.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF:F16
OLLAMA_MODEL   = "pathos"
IMAGE_SIZE     = 448
SC_SAMPLES     = 3
SC_TEMPERATURE = 0.7
MAX_RETRIES    = 2
RAG_TOP_K      = 3
ENABLE_RAG     = RAG_AVAILABLE

# ============================================================
# PROMPT TEMPLATES  (Technique 5)
# ============================================================

PROMPT_YN = """\
You are PathOS, an expert AI pathologist specialising in histopathology.

STRICT OUTPUT RULE: Your response MUST start with <answer>yes</answer> or <answer>no</answer>.
Write the tag FIRST. Then add one sentence of reasoning. No exceptions.

Format:
<answer>yes</answer> [one sentence of evidence]
OR
<answer>no</answer> [one sentence of evidence]

Question: {question}
{few_shot}
Analyse the provided image and respond now."""

PROMPT_TISSUE = """\
You are PathOS, an expert AI pathologist specialising in H&E histopathology.

Identify the tissue type and key morphological features visible in this image.
STRICT OUTPUT RULE: End your response with <answer>tissue name</answer>.

Structure:
- Key features observed: [list 2-3 features]
- Tissue classification: [classification]
<answer>tissue name</answer>

{few_shot}
Question: {question}"""

PROMPT_OPEN = """\
You are PathOS, an expert AI pathologist specialising in histopathology.

Reason step by step from the visible morphological evidence, then commit to a specific answer.
STRICT OUTPUT RULE: End your response with <answer>your specific answer</answer>.

Step 1 - Observe: What structures are visible?
Step 2 - Reason: What do these features indicate?
Step 3 - Conclude: <answer>your answer here</answer>

{few_shot}
Question: {question}"""

PROMPT_RETRY_YN = """\
You are PathOS. Answer ONLY with one of these two exact strings:
<answer>yes</answer>
<answer>no</answer>

Do not write anything else. The question was: {question}"""

PROMPT_RETRY_OPEN = """\
You are PathOS. Answer the question below.
Your entire response must be: <answer>THE ANSWER</answer>
Nothing before, nothing after.

Question: {question}"""

# ============================================================
# SEED EXEMPLARS (RAG knowledge base)
# ============================================================

SEED_EXEMPLARS = [
    {"q": "Is nuclear pleomorphism present?",     "a": "yes",
     "context": "Nuclear pleomorphism indicates variation in nuclear size and shape, common in malignant cells."},
    {"q": "Are goblet cells visible?",            "a": "yes",
     "context": "Goblet cells are mucus-secreting columnar cells seen in normal intestinal mucosa."},
    {"q": "Is carcinoma present?",                "a": "no",
     "context": "No malignant glandular structures or invasion patterns visible."},
    {"q": "Does this show a benign lesion?",      "a": "yes",
     "context": "Regular architecture with no pleomorphism or invasion."},
    {"q": "Is mitosis elevated?",                 "a": "yes",
     "context": "Multiple mitotic figures visible, consistent with high-grade malignancy."},
    {"q": "What tissue type is present?",         "a": "colorectal adenocarcinoma",
     "context": "Irregular glands, nuclear pleomorphism, hyperchromatic nuclei."},
    {"q": "What is the primary finding?",         "a": "lymphocytic infiltrate",
     "context": "Dense population of small round lymphocytes with scant cytoplasm."},
    {"q": "What structures are visible?",         "a": "normal colon crypts",
     "context": "Regular crypt architecture with goblet cells and columnar epithelium."},
    {"q": "What does this patch show?",           "a": "adipose tissue",
     "context": "Univacuolated lipid droplets with peripheral compressed nuclei."},
    {"q": "Is desmoplastic stroma present?",      "a": "yes",
     "context": "Dense collagen deposition with spindle fibroblasts around tumor nests."},
    {"q": "What is the tissue architecture?",     "a": "mucinous",
     "context": "Large pools of extracellular mucin with floating epithelial clusters."},
    {"q": "Are mitotic figures present?",         "a": "no",
     "context": "No active cell division visible in this field."},
    {"q": "What type of cells predominate?",      "a": "spindle cells",
     "context": "Elongated smooth muscle cells with cigar-shaped nuclei in intersecting fascicles."},
    {"q": "Is necrosis present?",                 "a": "yes",
     "context": "Ghost cell outlines and karyolysis indicate tumor necrosis."},
    {"q": "What is the inflammatory infiltrate?", "a": "lymphocytic",
     "context": "Dense lymphocytic infiltrate between glandular structures."},
]

# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_and_encode_image(image_path: str, size: int = IMAGE_SIZE) -> str:
    """Load image, resize to 448x448, return as base64 for Ollama."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ============================================================
# QUESTION CLASSIFICATION  (Technique 4)
# ============================================================

YN_PATTERNS = re.compile(
    r"^(is |are |does |do |was |were |has |have |can |could |should |did |will )",
    re.IGNORECASE,
)

TISSUE_PATTERNS = re.compile(
    r"(tissue|patch|slide|cell type|tissue type|classify|classification|stain|h&e|what type|what kind)",
    re.IGNORECASE,
)

def classify_question(question: str) -> str:
    q = question.strip()
    if YN_PATTERNS.match(q):
        return "yn"
    if TISSUE_PATTERNS.search(q):
        return "tissue"
    return "open"

# ============================================================
# ANSWER EXTRACTION
# ============================================================

HEDGE_PATTERNS = re.compile(
    r"\b(cannot|can't|unable|not possible|difficult to|cannot determine|"
    r"cannot confirm|i'm not able|i am not able|not sure|uncertain)\b",
    re.IGNORECASE,
)

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().lower() if m else ""

def is_hedged(text: str) -> bool:
    return bool(HEDGE_PATTERNS.search(text))

# ============================================================
# RAG MODULE  (Technique 3)
# ============================================================

class PathosRAG:
    """Local offline vector store of pathology Q&A exemplars."""

    def __init__(self, db_path: str = "./pathos_rag_db"):
        if not RAG_AVAILABLE:
            self.enabled = False
            return
        self.enabled = True
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.col = self.client.get_or_create_collection("pathos_exemplars")
        if self.col.count() == 0:
            self._seed()

    def _seed(self):
        docs, ids, metas = [], [], []
        for i, ex in enumerate(SEED_EXEMPLARS):
            docs.append(ex["q"] + " " + ex.get("context", ""))
            ids.append(f"seed_{i}")
            metas.append({"question": ex["q"], "answer": ex["a"], "context": ex.get("context", "")})
        for code, info in TISSUE_KB.items():
            text = f"{info['name']} features: {', '.join(info['features'])} clinical: {info['clinical']}"
            docs.append(text)
            ids.append(f"kb_{code}")
            metas.append({"question": f"What is {info['name']}?", "answer": info["name"],
                          "context": ", ".join(info["features"])})
        embeddings = self.embedder.encode(docs).tolist()
        self.col.add(documents=docs, ids=ids, metadatas=metas, embeddings=embeddings)
        print(f"[RAG] Seeded {len(docs)} exemplars into vector store.")

    def add_case(self, question: str, answer: str, context: str = ""):
        if not self.enabled:
            return
        doc_id = f"case_{int(time.time() * 1000)}"
        text = question + " " + context
        emb = self.embedder.encode([text]).tolist()
        self.col.add(documents=[text], ids=[doc_id],
                     metadatas=[{"question": question, "answer": answer, "context": context}],
                     embeddings=emb)

    def retrieve(self, question: str, k: int = RAG_TOP_K) -> str:
        if not self.enabled:
            return ""
        emb = self.embedder.encode([question]).tolist()
        results = self.col.query(query_embeddings=emb, n_results=min(k, self.col.count()))
        metas = results.get("metadatas", [[]])[0]
        if not metas:
            return ""
        lines = ["Relevant examples from pathology knowledge base:"]
        for m in metas:
            lines.append(f"Q: {m['question']}")
            lines.append(f"A: <answer>{m['answer']}</answer>")
            if m.get("context"):
                lines.append(f"   Context: {m['context']}")
            lines.append("")
        return "\n".join(lines)

# ============================================================
# CORE INFERENCE  (Techniques 1, 2, 6)
# ============================================================

def call_ollama(model: str, prompt: str, image_b64: str, temperature: float = 0.1) -> str:
    response = ollama.generate(
        model=model, prompt=prompt, images=[image_b64],
        options={"temperature": temperature, "num_predict": 200, "stop": ["</answer>"]},
    )
    text = response["response"]
    if "<answer>" in text and "</answer>" not in text:
        text += "</answer>"
    return text


def run_with_retry(model, prompt, image_b64, question, q_type, max_retry=MAX_RETRIES):
    """Technique 6 — confidence-gated retry."""
    response = call_ollama(model, prompt, image_b64)
    answer = extract_answer(response)
    if answer:
        return response, answer

    for attempt in range(max_retry):
        retry_prompt = (PROMPT_RETRY_YN if q_type == "yn" else PROMPT_RETRY_OPEN).format(question=question)
        response = call_ollama(model, retry_prompt, image_b64, temperature=0.0)
        answer = extract_answer(response)
        if answer:
            print(f"  [retry {attempt+1}] recovered answer: {answer}")
            return response, answer

    if q_type == "yn":
        words = re.findall(r"\b(yes|no)\b", response.lower())
        if words:
            return response, words[0]
    return response, ""


def self_consistency_vote(answers: list[str], q_type: str) -> str:
    """Technique 2 — majority voting."""
    clean = [a.strip().lower() for a in answers if a.strip()]
    if not clean:
        return ""
    if q_type == "yn":
        counts = Counter(a for a in clean if a in ("yes", "no"))
        if counts:
            return counts.most_common(1)[0][0]
    counts = Counter(clean)
    if counts.most_common(1)[0][1] > 1:
        return counts.most_common(1)[0][0]
    return max(clean, key=len)

# ============================================================
# MAIN PIPELINE
# ============================================================

class PathosEngine:
    """
    Full PathOS inference pipeline with 6 hybrid techniques.
    Integrates with pathos_tools.py for agentic post-processing.

    Models used:
        GGUF:    hf.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF:F16
        Adapter: dhairyapandya/pathos-gemma4-histopathology-rl
    """

    def __init__(self, model: str = OLLAMA_MODEL, rag_db: str = "./pathos_rag_db"):
        self.model = model
        self.rag = PathosRAG(db_path=rag_db)
        print(f"[PathOS] Ready. Model: {model} | RAG: {self.rag.enabled}")

    def analyse(self, image_path: str, question: str,
                sc_samples: int = SC_SAMPLES, use_rag: bool = ENABLE_RAG,
                verbose: bool = False) -> dict:
        t0 = time.time()
        image_b64 = load_and_encode_image(image_path)

        # Technique 4 & 5: classify + select template
        q_type = classify_question(question)
        if verbose:
            print(f"  [classify] q_type={q_type}")

        # Technique 3: RAG retrieval
        few_shot = self.rag.retrieve(question) if use_rag else ""

        # Build prompt
        if q_type == "yn":
            prompt = PROMPT_YN.format(question=question, few_shot=few_shot)
        elif q_type == "tissue":
            prompt = PROMPT_TISSUE.format(question=question, few_shot=few_shot)
        else:
            prompt = PROMPT_OPEN.format(question=question, few_shot=few_shot)

        # Technique 2: self-consistency sampling
        use_sc = (q_type == "yn") or (sc_samples > 1 and q_type == "open")
        n_calls = sc_samples if use_sc else 1

        all_responses, all_answers = [], []
        for i in range(n_calls):
            temp = SC_TEMPERATURE if (use_sc and i > 0) else 0.1
            resp, ans = run_with_retry(self.model, prompt, image_b64, question, q_type)
            all_responses.append(resp)
            all_answers.append(ans)
            if verbose:
                print(f"  [sample {i+1}] answer={ans!r}")
            # Early stopping
            if use_sc and len(set(all_answers)) == 1 and len(all_answers) >= 2:
                if verbose:
                    print(f"  [early stop] all {len(all_answers)} samples agree")
                break

        final_answer = self_consistency_vote(all_answers, q_type)

        # Confidence
        total = len([a for a in all_answers if a])
        if total == 0:
            confidence = 0.0
        elif q_type == "yn":
            confidence = sum(1 for a in all_answers if a == final_answer) / max(total, 1)
        else:
            counts = Counter(a for a in all_answers if a)
            confidence = counts.most_common(1)[0][1] / total if counts else 0.0

        best_response = (all_responses[all_answers.index(final_answer)]
                         if final_answer in all_answers else all_responses[0])
        hedged = is_hedged(best_response)
        elapsed = time.time() - t0

        result = {
            "question": question, "q_type": q_type, "answer": final_answer,
            "confidence": round(confidence, 2), "hedged": hedged,
            "n_samples": len(all_answers), "all_answers": all_answers,
            "reasoning": best_response, "time_sec": round(elapsed, 1),
        }

        # Auto-add to RAG
        if final_answer and confidence >= 0.66:
            self.rag.add_case(question=question, answer=final_answer,
                              context=f"Slide: {Path(image_path).name}")
        return result

    def print_result(self, result: dict):
        conf_pct = int(result["confidence"] * 100)
        flag = " ⚠ UNCERTAIN" if result["hedged"] else ""
        print(f"\n{'━'*55}")
        print(f"  PATHOS ANALYSIS RESULT{flag}")
        print(f"{'━'*55}")
        print(f"  Q:          {result['question']}")
        print(f"  Type:       {result['q_type'].upper()}")
        print(f"  Answer:     {(result['answer'] or '[UNKNOWN]').upper()}")
        print(f"  Confidence: {conf_pct}% ({result['n_samples']} samples)")
        print(f"  Time:       {result['time_sec']}s")
        if result["n_samples"] > 1:
            print(f"  All votes:  {result['all_answers']}")
        print(f"{'━'*55}")
        print(f"  Reasoning:\n  {result['reasoning'][:300]}...")
        print(f"{'━'*55}\n")

    def full_report(self, image_path: str, verbose: bool = False) -> dict:
        """
        Run the full agentic pipeline:
        1. Ask 5 diagnostic questions via hybrid inference
        2. Post-process through pathos_tools (flag, stain, atlas, report)
        """
        questions = [
            "What tissue type is present in this histopathology patch?",
            "Is malignancy present?",
            "Are mitotic figures visible?",
            "Is nuclear pleomorphism present?",
            "What is the primary clinical finding and its significance?",
        ]

        print(f"\n[PathOS] Generating full lab report for {Path(image_path).name}")
        results = []
        for q in questions:
            r = self.analyse(image_path, q, verbose=verbose)
            self.print_result(r)
            results.append(r)

        tissue_ans = results[0]["answer"]
        malignant  = results[1]["answer"]
        clinical   = results[4]["answer"]

        # ── Agentic post-processing via pathos_tools ──
        # 1. Analyze patch
        combined_reasoning = " ".join(r["reasoning"] for r in results)
        patch_analysis = analyze_patch(combined_reasoning, image_path)

        # 2. Flag malignancy
        risk = flag_malignancy(combined_reasoning, patch_analysis["tissue_type"])

        # 3. Suggest stains
        stains = suggest_special_stains(tissue_ans or "unknown", clinical or "")

        # 4. Compare to atlas
        atlas = compare_to_atlas(
            patch_analysis["tissue_type"],
            patch_analysis["observed_features"],
        )

        # 5. Generate report
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        conf_label = "High" if avg_conf >= 0.8 else ("Moderate" if avg_conf >= 0.5 else "Low")

        report = generate_report(
            tissue_type=tissue_ans or patch_analysis["tissue_name"],
            primary_finding=clinical or "See detailed analysis",
            malignancy_score=risk["risk_level"].lower(),
            morphology=", ".join(patch_analysis["observed_features"]),
            recommended_workup=[s["stain"] for s in stains["recommended_stains"]],
            image_name=Path(image_path).name,
            confidence=conf_label,
        )

        print(report["report_text"])

        if atlas["best_match"]:
            print(f"\n  Atlas match: {atlas['best_match']['tissue_name']} "
                  f"(similarity: {atlas['best_match']['similarity']})")

        return {
            "report": report,
            "patch_analysis": patch_analysis,
            "malignancy": risk,
            "stains": stains,
            "atlas": atlas,
            "raw_results": results,
        }

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PathOS — AI histopathology inference engine")
    parser.add_argument("--image",       type=str,            help="Path to slide image")
    parser.add_argument("--question",    type=str,            help="Clinical question")
    parser.add_argument("--report",      action="store_true", help="Run full structured report")
    parser.add_argument("--interactive", action="store_true", help="Interactive lab mode")
    parser.add_argument("--model",       type=str, default=OLLAMA_MODEL)
    parser.add_argument("--rag-db",      type=str, default="./pathos_rag_db")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    engine = PathosEngine(model=args.model, rag_db=args.rag_db)

    if args.interactive:
        print("\n[PathOS] Interactive lab mode. Type 'quit' to exit.\n")
        while True:
            image_path = input("Image path: ").strip()
            if image_path.lower() == "quit":
                break
            if not Path(image_path).exists():
                print(f"  [!] File not found: {image_path}")
                continue
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break
            result = engine.analyse(image_path, question, verbose=args.verbose)
            engine.print_result(result)

    elif args.report and args.image:
        engine.full_report(args.image, verbose=args.verbose)

    elif args.image and args.question:
        result = engine.analyse(args.image, args.question, verbose=args.verbose)
        engine.print_result(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

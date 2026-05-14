# PathOS — Pathology Operating System

> **An open-weight, edge-deployable AI pathologist assistant that analyses H&E stained histopathology slides and generates structured diagnostic reports — running entirely offline on a laptop.**

[![Model](https://img.shields.io/badge/🤗_GGUF-ByteKnight28/pathos--rl--GGUF-blue)](https://huggingface.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF)
[![Adapter](https://img.shields.io/badge/🤗_LoRA-dhairyapandya/pathos--rl-green)](https://huggingface.co/dhairyapandya/pathos-gemma4-histopathology-rl)
[![Base](https://img.shields.io/badge/Base-Gemma_4_E2B-orange)](https://huggingface.co/google/gemma-4-e2b-it)
[![License](https://img.shields.io/badge/License-Apache_2.0-red)](LICENSE)

---

## What is PathOS?

PathOS is a fine-tuned **Gemma 4 E2B** multimodal model that:

- 🔬 Analyses H&E stained histopathology image patches
- 🎯 Identifies tissue types and pathological features with committed answers (no hedging)
- 📋 Generates structured **PATHOS LAB REPORTS**
- 🧠 Uses 6 inference-time techniques to boost accuracy without retraining
- 💻 Runs **fully offline** via Ollama on a standard laptop — no GPU, no cloud, no internet

### Who is this for?

- Pathology labs in low-resource settings with no cloud access
- Diagnostic centres needing AI-assisted first-pass H&E screening
- Researchers working with histopathology data on limited hardware

---

## Repository Structure

```
pathos/
│
├── Modelfile                           # Ollama build configuration
│                                       # Points to local GGUFs, sets system prompt
│
├── pathos_inference_engine.py          # Main inference engine (21 KB)
│                                       # 6 hybrid techniques: SC voting, RAG,
│                                       # selective CoT, retries, templates
│                                       # CLI: --image, --question, --report, --interactive
│
├── pathos_tools.py                     # 5 agentic diagnostic tools (23 KB)
│                                       # analyze_patch, flag_malignancy,
│                                       # suggest_special_stains, compare_to_atlas,
│                                       # generate_report
│
├── pathos_inference.py                 # Standalone inference reference (26 KB)
│                                       # Self-contained version with all techniques
│
├── pathos_inference_techniques_overview.html
│                                       # Visual summary of the 6 techniques
│
├── PathOS_Project_Brief_v2.md          # Full project documentation (48 KB)
│
│── gemma-4-e2b-it.Q4_K_M.gguf         # Language model (3.43 GB) — download from HF
└── gemma-4-e2b-it.F16-mmproj.gguf     # Vision projector (986 MB) — download from HF
```

---

## Quick Start

### 1. Download Model Files

Download both GGUFs from [ByteKnight28/pathos-gemma4-histopathology-rl-GGUF](https://huggingface.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF/tree/main) and place them in this directory:

| File | Size | Role |
|---|---|---|
| `gemma-4-e2b-it.Q4_K_M.gguf` | 3.43 GB | Quantized language + reasoning model |
| `gemma-4-e2b-it.F16-mmproj.gguf` | 986 MB | Vision projector for image processing |

### 2. Build with Ollama

```bash
# Install Ollama if needed: https://ollama.com/download
ollama create pathos -f Modelfile
```

### 3. Test the Model

```bash
# Text-only smoke test
ollama run pathos "What tissue types are found in colorectal biopsies?"

# With an image
ollama run pathos "Is malignancy present?" --images slide.png
```

### 4. Install Python Dependencies

```bash
pip install ollama chromadb sentence-transformers pillow numpy
```

### 5. Run the Inference Engine

```bash
# Single question with verbose output
python pathos_inference_engine.py --image slide.png \
    --question "Is carcinoma present?" --verbose

# Full agentic lab report (5 diagnostic questions + tool pipeline)
python pathos_inference_engine.py --image slide.png --report

# Interactive lab mode
python pathos_inference_engine.py --interactive
```

---

## Running with llama.cpp (No Ollama)

You can also run PathOS directly with llama.cpp — zero wrapper overhead.

### Install llama.cpp

```bash
# macOS
brew install llama.cpp

# Linux — build from source
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build -j --target llama-server llama-cli
```

### Run inference from terminal

```bash
# Text-only
llama-cli \
  -m ./gemma-4-e2b-it.Q4_K_M.gguf \
  --mmproj ./gemma-4-e2b-it.F16-mmproj.gguf \
  -p "What tissue types are found in colorectal biopsies?" \
  --temp 0.1 -n 256

# With an image
llama-cli \
  -m ./gemma-4-e2b-it.Q4_K_M.gguf \
  --mmproj ./gemma-4-e2b-it.F16-mmproj.gguf \
  --image slide.png \
  -p "Is malignancy present in this histopathology patch?" \
  --temp 0.1 -n 256
```

### Run as a local API server

```bash
llama-server \
  -m ./gemma-4-e2b-it.Q4_K_M.gguf \
  --mmproj ./gemma-4-e2b-it.F16-mmproj.gguf \
  --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker model run hf.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF:F16
```

---

## How It Works

### Model Architecture

| Component | Detail |
|---|---|
| **Base model** | `google/gemma-4-e2b-it` (2B params, multimodal) |
| **Fine-tuning** | Unsloth LoRA (r=32, α=32) on vision + language layers |
| **Phase 1** | SFT on 16,700 samples (NCT-CRC + PathVQA×2 + QUILT-LLaVA) |
| **Phase 2** | GRPO with 4 reward functions (anti-hedge, format, accuracy) |
| **Output format** | `<answer>...</answer>` XML tags for exact extraction |
| **Quantization** | GGUF Q4_K_M (3.43 GB) |

### 6 Inference-Time Techniques

These boost accuracy from **44%/26%** → **90%+/60%+** without retraining:

| # | Technique | Cost | Impact |
|---|---|---|---|
| 1 | **Constrained decoding** — force yes/no as first token | Zero | +30–40pp YN |
| 2 | **Self-consistency voting** — sample 3×, majority vote | Low | +15–25pp overall |
| 3 | **RAG retrieval** — ChromaDB exemplars as few-shot context | Low | +10–20pp open |
| 4 | **Selective CoT routing** — simple→direct, complex→reasoning | Zero | −30% latency |
| 5 | **Per-type prompt templates** — YN / tissue / open templates | Zero | +10pp quality |
| 6 | **Confidence-gated retry** — retry with stricter prompt if tag missing | Low | Eliminates blanks |

### Agentic Tool Pipeline

After the model answers diagnostic questions, results flow through 5 local tools:

```
Model output → analyze_patch → flag_malignancy → suggest_special_stains
                                                         ↓
                              PATHOS LAB REPORT ← generate_report ← compare_to_atlas
```

| Tool | Function |
|---|---|
| `analyze_patch` | Parse model text → structured findings (tissue, features, scores) |
| `flag_malignancy` | Keyword-weighted risk scoring (LOW / MODERATE / HIGH) |
| `suggest_special_stains` | Map diagnosis → IHC recommendations (CDX2, MSI, KRAS, etc.) |
| `compare_to_atlas` | Jaccard similarity against 9 NCT-CRC tissue classes |
| `generate_report` | Compile final structured PATHOS LAB REPORT |

---

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PATHOS LAB REPORT
  Generated by PathOS — AI Pathologist Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Date:            2026-05-15 01:40:00
  Specimen:        biopsy_patch_001.png
  Tissue Type:     colorectal adenocarcinoma
  Primary Finding: Invasive adenocarcinoma, moderately differentiated
  Morphology:      irregular glands, nuclear pleomorphism, mitotic figures
  Malignancy Risk: HIGH
  Confidence:      High
  Pathologist Note: URGENT — Recommend immediate pathologist review
                    and staging workup.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recommended Additional Workup:
    • MSI / MMR panel
    • KRAS mutation analysis
    • CDX2 IHC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚕ This is an AI-assisted preliminary report.
  Final diagnosis requires pathologist review.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |
| GPU | Not required | Not required |
| Internet | Not required | Not required |
| OS | Linux / macOS / Windows | Any |

---

## Model Links

| Resource | Link |
|---|---|
| GGUF (quantized) | [ByteKnight28/pathos-gemma4-histopathology-rl-GGUF](https://huggingface.co/ByteKnight28/pathos-gemma4-histopathology-rl-GGUF) |
| LoRA adapter | [dhairyapandya/pathos-gemma4-histopathology-rl](https://huggingface.co/dhairyapandya/pathos-gemma4-histopathology-rl) |
| Base model | [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it) |

---

## Training Data

| Dataset | Samples | Role | License |
|---|---|---|---|
| [NCT-CRC-HE-100K](https://huggingface.co/datasets/1aurent/NCT-CRC-HE) | 2,700 | Tissue grounding (9 classes) | CC BY 4.0 |
| [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | 10,000 (×2) | Clinical Q&A | MIT |
| [QUILT-LLaVA-Instruct](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K) | 4,000 | Histopathology reasoning | CC BY NC ND 3.0 |

---

## Disclaimer

> **PathOS is an AI-assisted screening tool. It does NOT replace board-certified pathologists.**
> All outputs are preliminary and require pathologist review before clinical action.
> PathOS operates within the H&E first-pass screening phase only and does not claim to replace molecular, IHC, or special stain testing.


*Apache 2.0 License*

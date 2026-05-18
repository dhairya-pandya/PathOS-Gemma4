# 🔬 PathOS — AI Pathologist for H&E Histopathology

> Offline AI diagnostic assistant that runs on any laptop via Ollama. No GPU required.

PathOS is a distilled Gemma 4 model fine-tuned for H&E stained histopathology analysis. It produces definitive, structured diagnoses with `<answer>` tags — no hedging, no "cannot determine".

## ⚡ Quick Start (2 commands)

```bash
# Install Ollama (if not installed)
# Windows: winget install Ollama.Ollama
# macOS:   brew install ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh

# Run PathOS (default Q8_0, ~3GB) — downloads automatically on first run
ollama run dhairyapandya/pathos

# Or run the lighter 4-bit version (~2.5GB) for very constrained hardware
ollama run dhairyapandya/pathos:4b
```

That's it. You now have a local AI pathologist.

---

## 🖼️ Analyzing Digitized Slides

### Text-only questions

```bash
ollama run dhairyapandya/pathos "What are the features of colorectal adenocarcinoma?"
```

### With slide images

```bash
# Analyze a single slide
ollama run dhairyapandya/pathos "Is malignancy present? slide.png"

# Tissue identification
ollama run dhairyapandya/pathos "What tissue type is present? biopsy_patch.jpg"

# Agentic Diagnostic Report (Comprehensive Analysis)
ollama run dhairyapandya/pathos "Generate a complete diagnostic pathology report detailing cellular structures and morphological features. slide.png"

# Check for specific features
ollama run dhairyapandya/pathos "Is nuclear pleomorphism present? tumor_section.png"
```

### Interactive session (multiple slides)

```bash
# Start interactive mode
ollama run dhairyapandya/pathos

# Then type questions at the prompt:
>>> Is malignancy present? [attach image via drag-drop]
>>> What tissue type is this? [attach another image]
>>> /bye
```

### Using the API (for integration)

```bash
# Start Ollama server
ollama serve

# Call the API with Vision support
curl http://localhost:11434/api/generate -d '{
  "model": "dhairyapandya/pathos",
  "prompt": "Is malignancy present?",
  "images": ["<base64_encoded_image_string>"],
  "stream": false
}'
```

---

## 📋 Supported Question Types

| Type | Example | Response Format |
|---|---|---|
| **Agentic Report** | "Generate a comprehensive pathology report..." | `**Cellular Structures:**... <answer>diagnosis</answer>` |
| **Yes/No** | "Is malignancy present?" | `Yes. <answer>yes</answer> Evidence...` |
| **Tissue ID** | "What tissue type is present?" | `Features: ... <answer>tissue name</answer>` |
| **Open-ended** | "What is the primary finding?" | `Description. <answer>finding</answer>` |

### Example Questions for Diagnostics

```
Is malignancy present?
What tissue type is present in this histopathology patch?
Is nuclear pleomorphism present?
Are mitotic figures visible?
Is necrosis present?
Are goblet cells visible?
Is lymphocytic infiltrate present?
Is desmoplastic stroma present?
What is the primary clinical finding?
Is carcinoma in situ present?
Are glandular structures regular or irregular?
Is there evidence of invasion beyond the basement membrane?
```

---

## 🏥 For Labs — Deployment Guide

### Option 1: Direct from Ollama Registry (easiest)

```bash
ollama run dhairyapandya/pathos
# or for the 4-bit version:
ollama run dhairyapandya/pathos:4b
```

### Option 2: Custom Modelfile (with enhanced system prompt)

```bash
# Download from HuggingFace
wget https://huggingface.co/dhairyapandya/pathos-gemma4-distilled-GGUF/resolve/main/pathos-Q8_0.gguf
wget https://huggingface.co/dhairyapandya/pathos-gemma4-distilled-GGUF/resolve/main/Modelfile

# Create local model with custom system prompt
ollama create pathos -f Modelfile

# Run (Just append the image path to your prompt)
ollama run pathos "Is malignancy present? slide.png"
```

### Option 3: Direct llama.cpp (Maximum Performance on Edge Devices)

> **💡 Hackathon Track Highlight:** PathOS relies on **llama.cpp** to achieve its extreme performance on resource-constrained hardware. By exporting the Gemma 4 weights into the optimized GGUF format and running inference directly through `llama.cpp`, PathOS eliminates API overhead, enabling 128K context window analysis natively on edge devices with no dedicated GPU.

```bash
# Install llama.cpp
# Windows: winget install llama.cpp
# macOS:   brew install llama.cpp

# Run with web UI
llama-server -hf dhairyapandya/pathos-gemma4-distilled-GGUF:Q8_0

# Or direct CLI
llama-cli -hf dhairyapandya/pathos-gemma4-distilled-GGUF:Q8_0
```

### Option 4: Docker

```bash
docker model run hf.co/dhairyapandya/pathos-gemma4-distilled-GGUF:Q8_0
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **RAM** | 4 GB (for 4b model) | 8 GB+ (for Q8 model) |
| **GPU** | Not required | Any (speeds up inference) |
| **Disk** | 3 GB free (for 4b model) | 5 GB+ free |
| **OS** | Windows 10+, macOS 12+, Linux, Raspberry Pi OS | Any |

### 🚀 Edge Deployment via llama.cpp (Raspberry Pi & SBCs)

Because of the extreme efficiency of the Gemma 4 architecture combined with 4-bit GGUF quantization powered by **llama.cpp**, the `4b` model is specifically designed for Edge AI deployment.

By leveraging `llama.cpp`'s native C/C++ backend, PathOS can run as a completely offline AI pathologist on severely resource-constrained devices like the **Raspberry Pi 5 (8GB)** or **Jetson Nano**, dynamically utilizing CPU threading to maximize token generation without requiring a GPU.

**Performance estimates (No GPU, CPU only via llama.cpp):**

| Hardware | Model Version | Speed |
|---|---|---|
| Modern laptop (16GB RAM) | `latest` (Q8_0) | ~3-5 tokens/sec |
| Desktop with GPU | `latest` (Q8_0) | ~20-30 tokens/sec |
| Older laptop (8GB RAM) | `4b` (Q4_K_M) | ~2-4 tokens/sec |
| Raspberry Pi 5 (8GB) | `4b` (Q4_K_M) | ~0.5-1 tokens/sec |

*Note: Running on a Raspberry Pi is slower, but it allows for a completely offline, battery-powered diagnostic assistant in remote or low-resource clinical settings.*

---

## 🧠 What's Inside

PathOS is a **Gemma 4 E2B** (2.6B params) model with **6 inference-time techniques distilled into the weights**:

1. **Constrained decoding** — always produces `<answer>` tags
2. **Agentic Reporting** — natively generates multi-paragraph structured pathology reports
3. **Anti-hedging** — never says "cannot determine" or "uncertain"
4. **Massive Context Windows** — 32K context on the standard model, and up to **128K context** on the 4B model
5. **Static RAG** — few-shot exemplars embedded in training
6. **CoT routing** — adapts reasoning depth to question complexity
7. **Format compliance** — trained via GRPO reward shaping

### Training Pipeline

```
google/gemma-4-e2b-it (base)
    ↓ SFT on 19.5K histopathology examples
    ↓ GRPO with format/accuracy rewards (150 samples)
    ↓ LoRA merge + GGUF Q8_0 quantization
    = PathOS (~3GB, runs on any laptop)
```

---

## 📁 Repository Structure

```
PathOS-Gemma4/
├── Modelfile                        # Ollama configuration
├── publish_to_ollama.sh             # Steps to publish on Ollama registry
└── Folder structures.md            # Detailed project documentation
```

## 🔗 Links

| Resource | URL |
|---|---|
| **Ollama Model** | [dhairyapandya/pathos](https://ollama.com/dhairyapandya/pathos) |
| **GGUF Model (Q8_0)** | [dhairyapandya/pathos-gemma4-distilled-GGUF](https://huggingface.co/dhairyapandya/pathos-gemma4-distilled-GGUF) |
| **GGUF Model (4B)** | [dhairyapandya/pathos-gemma4-distilled-rl-4B-GGUF](https://huggingface.co/dhairyapandya/pathos-gemma4-distilled-rl-4B-GGUF) |
| **LoRA Adapter** | [dhairyapandya/pathos-gemma4-distilled-rl-histopathology](https://huggingface.co/dhairyapandya/pathos-gemma4-distilled-rl-histopathology) |
| **Base Model** | [google/gemma-4-e2b-it](https://huggingface.co/google/gemma-4-e2b-it) |

## ⚠️ Disclaimer

PathOS is a research tool for **educational and assistive purposes only**. It is NOT a certified medical device. All outputs must be reviewed by a qualified pathologist before clinical use.

## 📄 License

Apache 2.0

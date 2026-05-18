# The Problem: A Pathologist Who Covers Three Districts Alone

A disease growing inside you today might not have a name for another five business days. Not because the test wasn't run. Not because the doctor doesn't care. Because the slide sitting in a city lab is number 340 in a queue and there is one pathologist to read them all.

In rural Gujarat, a single pathologist may serve half a million people. She reviews 80 slides on a good day. Three hundred more are waiting. A biopsy taken Monday won't be read until Friday. In low-resource settings, often not for two to four weeks.

| Region | Pathologists per 100,000 people |
|---|---|
| Sub-Saharan Africa | 0.3 |
| India | 1.2 |
| United States | 5.7 (still overwhelmed) |

A busy lab processes 200-400 slides per day. One pathologist can comfortably review 80-100. **The gap is structural and it is not closing.**

The bottleneck is not molecular testing or immunohistochemistry. Those are ordered *after* the initial H&E screen. **The bottleneck is the first-pass H&E screen itself:** look at every slide, decide normal or abnormal, identify tissue type, determine what additional tests to order, write a preliminary report. This step requires years of training, consumes most of a pathologist's day, and is where cancer is first caught or first missed.

**PathOS intervenes precisely here.** Not at the molecular stage. Not at IHC. At the first pass -- the screening step that creates the queue.

---

# The Solution: A Tireless AI Resident at Every Slide Scanner

PathOS is a fine-tuned multimodal Gemma 4 E2B model that performs H&E histopathology screening: analyzing digitized tissue patches, identifying malignancy indicators, recommending additional workup, and generating structured lab reports. It runs entirely offline on a standard CPU laptop. **No GPU. No cloud. No internet.**

PathOS does not replace pathologists. It does the first pass so the specialist can focus on what actually needs them: confirming findings, handling complex cases, signing reports that go to families waiting for answers.

| | Without PathOS | With PathOS |
|---|---|---|
| Slide transit | 2-day courier to city lab | Scanned locally |
| Queue wait | 3-5 day backlog | No queue |
| First-pass review | Manual, specialist required | AI screening in minutes |
| Report | 1-2 days to send back | Pathologist signs same day |
| **Total turnaround** | **7-14 days** | **Same day** |

The weights are open. The model runs on **Ollama** -- `ollama pull dhairyapandya/pathos` -- one command, no Python environment, works on any machine. A pathology lab in rural India with a $300 laptop can deploy PathOS today, for free. That is not a feature. That is the point.

---

# How We Built It: Challenges First

PathOS did not emerge from a clean three-step plan. It came from running out of disk space at 2 AM, watching kernels die mid-training, and figuring out why a peritoneum patch was being classified as *"cross-sections of multiple fish."*

We burned through **100 hours of Kaggle GPU quota** testing training strategies before landing on what works. T4s ran out of memory loading images. P100s turned out incompatible with the current PyTorch CUDA build (capability 6.0, minimum supported 7.0 -- discovered at the worst possible time). Kaggle's 19.5GB working directory filled up from intermediate checkpoints before we could save the final model. The fix was routing everything through `/tmp`, which has 1300GB that nobody tells you about. Genuinely one of those moments where the solution is embarrassingly simple once you find it.

**Unsloth made the fine-tuning actually possible.** Without it, fitting Gemma 4 E2B into a T4's 15.6GB of VRAM with vision layers enabled would have been a non-starter. Unsloth's 4-bit quantization and memory-efficient LoRA implementation got us to 8.2GB allocated with 7.4GB free -- enough headroom to actually train. The `finetune_vision_layers=True` flag is the one most people skip. We didn't, and it's why PathOS understands what it's looking at rather than pattern-matching text around an image it can't really see.

The multimodal training itself required a custom lazy-loading Dataset class. Loading 800 PIL images into RAM at once killed the kernel every time. The fix: load each image only when the trainer requests that specific batch, then discard it immediately. One image in RAM at a time instead of 800.

---

# The Model: What We Trained and Why It Changed

## First run: the hedging problem

PathOS v1 benchmarked at 26% overall accuracy. The failure analysis was specific:

**82% of yes/no errors were not wrong predictions. They were refusals.**

The model kept saying things like *"Based on the provided image, I cannot confirm..."* This is Gemma 4's base RLHF training -- it rewards expressing uncertainty. Six hundred steps of SFT is not enough to override years of that. We had to be more deliberate.

## What we changed

**Structured output format.** Every response ends with an explicit answer tag: `<answer>yes</answer>` or `<answer>adenocarcinoma with irregular gland formation</answer>`. Evaluation becomes exact tag extraction instead of fragile word-matching.

**More histopathology data, weighted deliberately:**

| Dataset | Samples | Role |
|---|---|---|
| PathVQA (x2 augmented) | 10,000 | Task-specific precision, mirrors benchmark |
| QUILT-LLaVA-Instruct-107K* | 4,000 | Histopathology vocabulary breadth |
| NCT-CRC-HE-100K | 2,700 | Tissue-class grounding, 9 classes |

*Ikezogwo et al., "Quilt-1M: One Million Image-Text Pairs for Histopathology," NeurIPS 2023.

**GRPO with four reward functions** to surgically remove the hedging:

| Reward function | Signal |
|---|---|
| `reward_no_hedge` | -1.0 if hedge language detected |
| `reward_yn_accuracy` | +1.0 exact binary match; -0.5 for opposite |
| `reward_format` | +0.3 if `<answer>` tag present; -0.5 if missing |
| `reward_open_accuracy` | +0.5 if ground-truth words in extracted answer |

SFT teaches what to say. GRPO teaches what not to say.

**Image resolution from 224x224 to 448x448.** At 224px, diagnostic detail is lost. The fish classification incident was a real data point, not a joke -- it told us exactly where the floor was.

## Closing the inference gap

After the second training run, a 55-point accuracy gap appeared between raw `ollama run pathos` (~35%) and the Python inference engine (~90%+). The gap came from six techniques that can't live in a Modelfile: self-consistency voting, RAG few-shot injection, CoT routing, constrained decoding, and confidence-gated retry.

The fix is distilling those techniques back into the weights. `generate_distillation_data.py` runs the full inference engine as a teacher -- voting and retry on 5,000+ pairs, keeping only high-confidence outputs -- then a final SFT pass produces a model that gets it right in a single forward pass. No proxy. No wrapper. Same logic as DeepSeek-R1: compress test-time compute back into the weights.

---

# Native Function Calling: The Agentic Pipeline

PathOS uses Gemma 4's native function calling to run a five-tool diagnostic pipeline that produces a complete structured report ready for pathologist sign-off:

`analyze_patch()` → `flag_malignancy()` → `suggest_special_stains()` → `compare_to_atlas()` → `generate_report()`

```
PATHOS LAB REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tissue Type:      Colorectal adenocarcinoma
Primary Finding:  Malignant glandular epithelium
Morphology:       Irregular glands, nuclear pleomorphism, mitotic figures
Clinical Imp.:    Requires oncology staging workup
Confidence:       High
Pathologist Note: Correlate with TNM staging and molecular markers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The pathologist opens the report, reviews the flagged image, and signs. That is the workflow.

---

# Evaluation

| Version | Yes/No | Open-ended | Overall | Key change |
|---|---|---|---|---|
| Base Gemma 4 E2B | ~50% | ~0% | ~25% | No pathology knowledge |
| PathOS v1 | 32% | 20% | 26% | 82% of yes/no failures were hedges |
| PathOS v2 | 44% | -- | 35% | Answer tags + GRPO anti-hedge |
| **PathOS v3** | **88%** | -- | -- | Distilled inference techniques |

PathOS v3 reaches 88% yes/no accuracy on the 100-sample PathVQA validation set with no proxy, no voting loop, no wrapper. Single forward pass. That number matters because it is the number a rural lab actually gets when they run `ollama run pathos`.

---

# Deployment: Edge-First by Design

**Ollama** is the primary distribution path -- `ollama pull dhairyapandya/pathos`. One command. No Python. No pip install. Works on any machine running Ollama. This matters because the people who need PathOS most are not running Python environments in rural diagnostic labs. Ollama makes the gap between "model exists" and "model is usable" disappear.

**The core of what makes offline deployment real is llama.cpp.** We exported our fine-tuned weights into GGUF format specifically to access llama.cpp's aggressive 4-bit quantization (Q4_K_M) and dynamic CPU threading. Ollama uses llama.cpp under the hood, but for fully air-gapped deployments -- no package manager, no network, no Ollama installed -- the raw GGUF runs directly. This is what shrinks a multimodal architecture down to ~2.5GB without meaningful accuracy loss. **PathOS runs on older clinical laptops. It runs on a Raspberry Pi 5.** The GGUF weights are published on HuggingFace for anyone to download, audit, and deploy independently. The LoRA adapter (~150MB) is there too for researchers who want to fine-tune further on their own pathology data. All training code, GRPO reward functions, distillation pipeline, and evaluation scripts are public.

---

# Why It Matters

Commercial AI pathology tools -- Paige.AI, PathAI, Aiforia -- cost $50,000-$500,000 per year, require cloud connectivity, and run on proprietary weights. The hospitals that can afford them are not the hospitals with the shortage problem.

PathOS is the open alternative. Same task. Verified benchmarks. Published weights. Runs offline. Free.

The pathologist shortage will not be solved by training more pathologists -- that takes decades. What can happen now is that every pathologist gets an AI resident that handles the screening queue, so the specialist focuses on cases that actually need them.

A biopsy taken Monday. A report signed Monday. A family that doesn't spend the week not knowing.

That's what this is for.

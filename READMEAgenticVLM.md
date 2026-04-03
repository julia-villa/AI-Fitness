# AI Fitness Coach Auditor: Zero-Shot VLM Evaluation Pipeline

This repository contains a localized, agentic evaluation pipeline designed to audit biomechanical exercise form. Utilizing a zero-shot Vision-Language Model (Moondream 2, 1.8B parameters), the system analyzes video segments to detect physical anomalies (e.g., rounded spine, caving knees) and generates structured, real-time coaching feedback.

The pipeline is benchmarked against human ground-truth data to evaluate the viability of lightweight, localized VLMs for high-speed biomechanical analysis compared to traditional fine-tuned or frame-by-frame baselines.

## ⚙️ Architecture & Workflow

The system relies on a targeted, three-step chain-of-thought (CoT) prompting strategy to prevent typical small-model hallucinations (e.g., environmental captioning) and force strict biomechanical auditing.

1. **Extraction:** The script reads a manifest of exercise video segments and calculates the temporal midpoint (`mid_relative`) to extract a representative structural frame.
2. **Inference (`auditor.py`):** The frame is passed to Moondream 2 via a highly constrained VQA (Visual Question Answering) prompt, requiring a physical breakdown of joints before delivering a final "Good/Bad" verdict.
3. **Structuring (`run_agent_benchmark.py`):** The output, alongside precise token-usage metrics and inference latencies (`ttft_sec`), is dynamically formatted into a strict JSON schema.
4. **Evaluation (`stage3_eval.py`):** The JSON is scored against a human-annotated manifest using semantic and temporal metrics.

## 🚀 Setup and Installation

This project requires a CUDA-enabled NVIDIA GPU (tested on RTX 2070 Super) for optimal inference speeds.

**1. Create the virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

**2. Install dependencies (with CUDA 12.1 support):**

```bash
python -m pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu121


Step 1: Run the Inference Benchmark
This script processes the video manifest, runs the Moondream 2 VLM on the extracted frames, and generates the predictions.json file.

```bash
python src/run_agent_benchmark.py
Note: Debug frames are saved to eval/debug_frames/ to visually verify the VLM's input context).

Step 2: Evaluate the Output
Score the predictions against the ground truth to generate the final NLP and latency metrics.

```bash
python src/scripts/stage3_eval.py --predictions eval/predictions.json --references eval/be
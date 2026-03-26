# QEVD-FIT-COACH Evaluation Pipeline

## Overview

This pipeline lets you evaluate any model against the QEVD-FIT-COACH benchmark.

```
preprocess_qevd.py   →   output/train.jsonl      →   train your model
                                                            ↓
predict_skeleton.py  →   results/my_model.json   →   evaluate_qevd.py
(plug in your model)      (model predictions)             ↑
                                                  feedbacks_long_range.json
                                                  (raw ground truth)
```

---

## 1. Install Dependencies

```bash
pip install fiftyone mediapipe opencv-python numpy tqdm \
            nltk rouge-score bert-score transformers torch \
            pandas tabulate pyyaml
```

---

## 2. Preprocessing — `preprocess_qevd.py`

### Minimal — benchmark only (no training data, no pose)
```bash
python preprocess_qevd.py \
  --output_dir ./output \
  --skip-train \
  --skip-pose \
  --skip-export
```
Runs Part A only. Downloads the benchmark from HuggingFace into FiftyOne. Good first check before committing to a full run.

### Full run — benchmark + training data + pose + JSONL export
```bash
python preprocess_qevd.py \
  --dataset_root /path/to/combined \
  --output_dir ./output \
  --workers 4
```
Runs everything (Parts A–F). `--dataset_root` is the `combined/` folder containing:
- `fine_grained_labels.json`
- `feedbacks_short_clips.json`
- `feedbacks_long_range.json`
- `questions.json`

### Skip pose extraction (run later or already done)
```bash
python preprocess_qevd.py \
  --dataset_root /path/to/combined \
  --output_dir ./output \
  --skip-pose
```

### Reload existing training dataset without rebuilding
```bash
python preprocess_qevd.py \
  --dataset_root /path/to/combined \
  --output_dir ./output \
  --skip-train
```
> `--skip-train` reuses an existing `qevd-fit-coach-train` FiftyOne dataset if it exists.

### Copy clips instead of symlinks
```bash
python preprocess_qevd.py \
  --dataset_root /path/to/combined \
  --output_dir ./output \
  --no-symlinks
```
Use this when moving data to another machine where symlinks won't resolve.

### Browse the benchmark in FiftyOne
```python
python -c "
import fiftyone as fo
session = fo.launch_app(fo.load_dataset('qevd-fit-coach-benchmark'))
input('Press Enter to exit...')
"
```
Opens a browser at `http://localhost:5151` to view all benchmark videos.

---

## 3. Implement Your Model — `predict_skeleton.py`

Copy `predict_skeleton.py` and rename it for your model (e.g. `predict_gemini.py`).
Fill in **only** `predict_single()` with your model call:

```python
def predict_single(video_path: str, mode: str) -> dict:
    # YOUR MODEL CALL HERE
    response = my_model.generate(video_path)

    feedback: list[str] = response.feedback_strings    # list of feedback strings
    timestamps: list[float] = response.timestamps      # one timestamp per feedback (long mode)
    ttft_ms: float = response.time_to_first_token_ms
    tps: float = response.tokens_per_second
    n_tokens: int = response.total_tokens_used

    # Do not change anything below this line
    ...
```

Everything else — looping over videos, saving the JSON, injecting device info — is handled automatically.

### Expected output format (`results/my_model.json`)

```json
[
  {
    "video_path": "0006.mp4",
    "predicted_feedback": ["Good squat depth", "Keep your back straight"],
    "predicted_timestamps": [5.0, 12.0],
    "inference_metadata": {
      "time_to_first_token_ms": 320.5,
      "time_to_last_token_ms": 1200.0,
      "tokens_per_second": 45.2,
      "total_tokens_used": 128,
      "device_info": {
        "device": "GPU",
        "gpu_name": "NVIDIA A100",
        "gpu_memory_gb": 80.0,
        "platform": "Darwin",
        "python_version": "3.11.0"
      }
    }
  }
]
```

> - `video_path` must exactly match `long_range_video_file` from the ground truth JSON
> - `predicted_timestamps` must be the same length as `predicted_feedback`
> - `predicted_timestamps` only required in `--mode long`
> - 74 dicts total for the full benchmark (one per video)
> - `device_info` is injected automatically — you do not fill this in

### Run predictions

**Long-range videos (benchmark evaluation):**
```bash
python predict_mymodel.py \
  --ground_truth_file /path/to/feedbacks_long_range.json \
  --output_file results/my_model_long.json \
  --mode long
```

**Short clips:**
```bash
python predict_mymodel.py \
  --ground_truth_file /path/to/combined/feedbacks_short_clips.json \
  --output_file results/my_model_short.json \
  --mode short
```

---

## 4. Evaluate — `evaluate_qevd.py`

### Long mode — benchmark (T-F Score + text metrics on matched pairs)
```bash
python evaluate_qevd.py \
  --results_file results/my_model_long.json \
  --ground_truth_file /path/to/feedbacks_long_range.json \
  --output_dir eval_output/my_model_long \
  --model_name "my-model" \
  --mode long
```

### Short mode
```bash
python evaluate_qevd.py \
  --results_file results/my_model_short.json \
  --ground_truth_file /path/to/combined/feedbacks_short_clips.json \
  --output_dir eval_output/my_model_short \
  --model_name "my-model" \
  --mode short
```

### Outputs

| File | Contents |
|---|---|
| `eval_output/.../metrics_report.json` | METEOR, ROUGE-L, BERTScore, T-F Score, inference performance, device info |
| `eval_output/.../per_video_metrics.csv` | Per-video breakdown of all metrics |

### Comparing multiple models
Run each model through predict + evaluate with a different `--model_name` and `--output_dir`:
```bash
python evaluate_qevd.py --results_file results/gemini.json  --model_name "gemini-2.0-flash"  --output_dir eval_output/gemini  ...
python evaluate_qevd.py --results_file results/gpt4v.json   --model_name "gpt-4v"            --output_dir eval_output/gpt4v   ...
python evaluate_qevd.py --results_file results/baseline.json --model_name "baseline"          --output_dir eval_output/baseline ...
```

---

## Metrics Reference

| Metric | Mode | Description |
|---|---|---|
| METEOR | short + long | Text similarity accounting for synonyms and paraphrases |
| ROUGE-L | short + long | Longest common subsequence F1 between prediction and GT |
| BERTScore F1 | short + long | Semantic similarity using roberta-large embeddings |
| T-F Score | long only | Temporal-Feedback F1 — correct feedback at the right time (±1.5s window) |

> In **long mode**, METEOR/ROUGE/BERTScore are computed on temporally matched pairs only (same algorithm as the reference evaluator). In **short mode** they are computed on the full feedback string.

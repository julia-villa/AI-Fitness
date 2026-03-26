#!/usr/bin/env python3
"""
Prediction skeleton for QEVD-FIT-COACH evaluation.

Fill in `predict_single()` with your model logic.
The output of `run_predictions()` is exactly the input schema
expected by evaluate_qevd.py --results_file.

Usage:
    python predict_skeleton.py \
        --ground_truth_file combined/feedbacks_short_clips.json \
        --output_file results/my_model.json \
        --mode short

    Then evaluate:
    python evaluate_qevd.py \
        --results_file results/my_model.json \
        --ground_truth_file combined/feedbacks_short_clips.json \
        --output_dir eval_output/ \
        --model_name "my-model" \
        --mode short
"""

import argparse
import json
import platform
import time
from pathlib import Path


def _get_device_info() -> dict:
    """Collect hardware/platform info for reproducibility tracking."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "device": "CPU",
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "GPU"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        elif torch.backends.mps.is_available():
            info["device"] = "MPS"
            info["gpu_name"] = "Apple Silicon"
    except ImportError:
        pass
    return info


# ===========================================================================
# FILL THIS IN — replace the body with your model call
# ===========================================================================

def predict_single(video_path: str, mode: str) -> dict:
    """
    Run your model on one video and return a single result dict.

    The returned dict is one element of the list that evaluate_qevd.py
    receives as --results_file. Do not change the top-level keys.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file, exactly as it appears
        in the ground truth JSON (so alignment works by string match).
    mode : str
        "short" or "long". In long mode you must also populate
        predicted_timestamps.

    Returns
    -------
    dict matching the evaluate_qevd.py results schema
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # YOUR MODEL CALL GOES HERE
    # e.g.:
    #   response = my_model.generate(video_path)
    #   feedback = response.feedback_strings       # list[str]
    #   timestamps = response.timestamps           # list[float], long mode only
    #   ttft_ms  = response.time_to_first_token_ms
    #   ttlt_ms  = response.time_to_last_token_ms
    #   tps      = response.tokens_per_second
    #   n_tokens = response.total_tokens_used
    # ------------------------------------------------------------------

    feedback: list[str] = []          # TODO: replace with model output
    timestamps: list[float] = []      # TODO: long mode only
    ttft_ms: float = 0.0              # TODO: replace with real latency
    tps: float = 0.0                  # TODO: replace with real throughput
    n_tokens: int = 0                 # TODO: replace with real token count

    ttlt_ms = (time.perf_counter() - t0) * 1000  # wall-clock fallback

    result = {
        "video_path": video_path,
        "predicted_feedback": feedback,
        "inference_metadata": {
            "time_to_first_token_ms": ttft_ms,
            "time_to_last_token_ms": ttlt_ms,
            "tokens_per_second": tps,
            "total_tokens_used": n_tokens,
        },
    }

    if mode == "long":
        result["predicted_timestamps"] = timestamps

    return result


# ===========================================================================
# Runner — do not need to edit below this line
# ===========================================================================

def load_video_paths(ground_truth_file: Path, mode: str) -> list[str]:
    """Extract the list of video paths from the ground truth file."""
    with open(ground_truth_file) as f:
        data = json.load(f)
    key = "long_range_video_file" if mode == "long" else "video_path"
    return [entry[key] for entry in data if key in entry]


def run_predictions(video_paths: list[str], mode: str) -> list[dict]:
    """
    Call predict_single() for every video and return the full results list.
    This return value is the direct input to evaluate_qevd.py --results_file.
    """
    results = []
    for vp in video_paths:
        result = predict_single(vp, mode)
        results.append(result)
        print(f"  predicted: {vp}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--mode", choices=["short", "long"], default="short")
    args = parser.parse_args()

    video_paths = load_video_paths(args.ground_truth_file, args.mode)
    print(f"Found {len(video_paths)} videos in ground truth ({args.mode} mode)")

    device_info = _get_device_info()
    print(f"Device: {device_info.get('device')} — {device_info.get('gpu_name', platform.processor() or 'CPU')}")

    results = run_predictions(video_paths, args.mode)

    # Attach device info to every result's inference_metadata
    for r in results:
        r["inference_metadata"]["device_info"] = device_info

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output_file}")
    print(f"Next step:")
    print(
        f"  python evaluate_qevd.py \\\n"
        f"    --results_file {args.output_file} \\\n"
        f"    --ground_truth_file {args.ground_truth_file} \\\n"
        f"    --output_dir eval_output/ \\\n"
        f"    --model_name my-model \\\n"
        f"    --mode {args.mode}"
    )


if __name__ == "__main__":
    main()

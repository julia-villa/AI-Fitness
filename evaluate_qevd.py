#!/usr/bin/env python3
"""
QEVD-FIT-COACH Model-Agnostic Evaluation Script
================================================
Compares model outputs against QEVD-FIT-COACH ground truth.
Does NOT call any model — accepts a pre-computed results JSON file.

Usage:
    python evaluate_qevd.py \
        --results_file results/gemini-2.0-flash.json \
        --ground_truth_file combined/feedbacks_short_clips.json \
        --output_dir eval_output/ \
        --model_name "gemini-2.0-flash" \
        --mode short

    python evaluate_qevd.py \
        --results_file results/model_long.json \
        --ground_truth_file combined/feedbacks_long_range.json \
        --output_dir eval_output/ \
        --model_name "my-model" \
        --mode long
"""

import argparse
import csv
import json
import logging
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model-agnostic evaluation against QEVD-FIT-COACH benchmark"
    )
    parser.add_argument(
        "--results_file",
        type=Path,
        required=True,
        help="Path to model results JSON (list of prediction objects)",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=Path,
        required=True,
        help="feedbacks_short_clips.json or feedbacks_long_range.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for metrics_report.json and per_video_metrics.csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unknown-model",
        help="Label for this evaluation run (e.g. 'gemini-2.0-flash')",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "long"],
        default="short",
        help="'short' = feedbacks_short_clips, 'long' = feedbacks_long_range (enables T-F Score)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"results_file must be a JSON array, got {type(data)}")
    return data


def load_ground_truth(path: Path, mode: str) -> dict[str, dict]:
    """
    Returns a dict keyed by video_path.
    Each value: { "feedback": list[str], "timestamps": list[float] (long only),
                  "is_transition": list[bool] (long only) }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    gt: dict[str, dict] = {}

    if mode == "short":
        for entry in raw:
            vp = entry.get("video_path", "")
            fb = entry.get("feedback", [])
            if isinstance(fb, str):
                fb = [fb]
            gt[vp] = {"feedback": fb}

    else:  # long
        for entry in raw:
            vp = entry.get("long_range_video_file", "")
            fb = entry.get("feedback", [])
            if isinstance(fb, str):
                fb = [fb]
            timestamps = entry.get("feedback_timestamps", [])
            is_transition = entry.get("is_transition", [])
            gt[vp] = {
                "feedback": fb,
                "timestamps": timestamps,
                "is_transition": is_transition,
            }

    return gt


def align(results: list[dict], gt: dict[str, dict], mode: str) -> list[dict]:
    """
    Match results to ground truth by video_path.
    Returns a list of aligned dicts, warns on mismatches.
    """
    gt_keys = set(gt.keys())
    result_keys = {r["video_path"] for r in results}

    unmatched_pred = result_keys - gt_keys
    unmatched_gt = gt_keys - result_keys

    for vp in sorted(unmatched_pred):
        log.warning(f"No ground truth found for predicted video: {vp}")
    for vp in sorted(unmatched_gt):
        log.warning(f"No prediction found for ground truth video: {vp}")

    aligned = []
    for r in results:
        vp = r["video_path"]
        if vp not in gt:
            continue
        entry = {
            "video_path": vp,
            "predicted_feedback": r.get("predicted_feedback", []),
            "gt_feedback": gt[vp]["feedback"],
            "inference_metadata": r.get("inference_metadata", None),
        }
        if mode == "long":
            entry["predicted_timestamps"] = r.get("predicted_timestamps", [])
            entry["gt_timestamps"] = gt[vp].get("timestamps", [])
        aligned.append(entry)

    log.info(
        f"Coverage: {len(aligned)} matched / {len(results)} predicted / {len(gt)} ground truth"
    )
    return aligned


# ---------------------------------------------------------------------------
# Text metric helpers
# ---------------------------------------------------------------------------

def _flatten_list(lst) -> list[str]:
    """Ensure list[str]; wrap bare string."""
    if isinstance(lst, str):
        return [lst]
    return [s for s in lst if isinstance(s, str) and s.strip()]


def _join(strings: list[str]) -> str:
    """Concatenate feedback strings into a single string for scoring."""
    return " ".join(strings).strip()


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

def compute_meteor(aligned: list[dict]) -> dict:
    log.info("Computing METEOR …")
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor

    # Ensure WordNet is available
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        log.info("  Downloading NLTK wordnet …")
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    scores = []
    per_video = {}

    for entry in tqdm(aligned, desc="METEOR", unit="video"):
        pred_str = _join(_flatten_list(entry["predicted_feedback"]))
        refs = _flatten_list(entry["gt_feedback"])

        if not pred_str or not refs:
            per_video[entry["video_path"]] = None
            continue

        # Score against each reference, take max
        pred_tokens = pred_str.split()
        ref_token_lists = [r.split() for r in refs]
        best = max(_meteor(ref_token_lists, pred_tokens) for _ in [None])

        # meteor_score with multiple references: pass list of token lists
        best = _meteor(ref_token_lists, pred_tokens)

        scores.append(best)
        per_video[entry["video_path"]] = round(best, 6)

    mean = float(np.mean(scores)) if scores else 0.0
    return {"mean": round(mean, 6), "per_video": per_video}


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge(aligned: list[dict]) -> dict:
    log.info("Computing ROUGE-L …")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    per_video = {}

    for entry in tqdm(aligned, desc="ROUGE-L", unit="video"):
        pred_str = _join(_flatten_list(entry["predicted_feedback"]))
        refs = _flatten_list(entry["gt_feedback"])

        if not pred_str or not refs:
            per_video[entry["video_path"]] = None
            continue

        # Take max F1 across all references
        best_f1 = max(
            scorer.score(ref, pred_str)["rougeL"].fmeasure for ref in refs
        )
        scores.append(best_f1)
        per_video[entry["video_path"]] = round(best_f1, 6)

    mean = float(np.mean(scores)) if scores else 0.0
    return {"mean_f1": round(mean, 6), "per_video": per_video}


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(aligned: list[dict]) -> dict:
    log.info("Computing BERTScore (roberta-large) — this may take a while …")
    from bert_score import score as bert_score_fn

    # Build flat lists; for multiple references per video take max F1
    video_paths = []
    predictions = []
    references_per_video = []  # list of list[str]

    for entry in aligned:
        pred_str = _join(_flatten_list(entry["predicted_feedback"]))
        refs = _flatten_list(entry["gt_feedback"])
        if not pred_str or not refs:
            continue
        video_paths.append(entry["video_path"])
        predictions.append(pred_str)
        references_per_video.append(refs)

    if not predictions:
        return {"mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0, "per_video": {}}

    # Compute BERTScore for each (prediction, reference) pair
    # We flatten: one pred row per (pred, ref) combination, then group back
    flat_preds = []
    flat_refs = []
    flat_keys = []  # (video_idx, ref_idx)

    for vid_idx, (pred, refs) in enumerate(zip(predictions, references_per_video)):
        for ref_idx, ref in enumerate(refs):
            flat_preds.append(pred)
            flat_refs.append(ref)
            flat_keys.append((vid_idx, ref_idx))

    P_flat, R_flat, F1_flat = bert_score_fn(
        flat_preds,
        flat_refs,
        model_type="roberta-large",
        lang="en",
        verbose=False,
        device=None,  # auto-select GPU/CPU
    )

    P_flat = P_flat.numpy()
    R_flat = R_flat.numpy()
    F1_flat = F1_flat.numpy()

    # Group back: max F1 per video
    per_vid_scores: dict[int, dict] = defaultdict(lambda: {"P": [], "R": [], "F1": []})
    for i, (vid_idx, _) in enumerate(flat_keys):
        per_vid_scores[vid_idx]["P"].append(P_flat[i])
        per_vid_scores[vid_idx]["R"].append(R_flat[i])
        per_vid_scores[vid_idx]["F1"].append(F1_flat[i])

    per_video = {}
    all_P, all_R, all_F1 = [], [], []

    for vid_idx, vp in enumerate(video_paths):
        if vid_idx not in per_vid_scores:
            per_video[vp] = None
            continue
        best_idx = int(np.argmax(per_vid_scores[vid_idx]["F1"]))
        p = float(per_vid_scores[vid_idx]["P"][best_idx])
        r = float(per_vid_scores[vid_idx]["R"][best_idx])
        f = float(per_vid_scores[vid_idx]["F1"][best_idx])
        per_video[vp] = {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f, 6)}
        all_P.append(p)
        all_R.append(r)
        all_F1.append(f)

    return {
        "mean_precision": round(float(np.mean(all_P)) if all_P else 0.0, 6),
        "mean_recall": round(float(np.mean(all_R)) if all_R else 0.0, 6),
        "mean_f1": round(float(np.mean(all_F1)) if all_F1 else 0.0, 6),
        "per_video": per_video,
    }


# ---------------------------------------------------------------------------
# T-F Score (Temporal-Feedback Score) — long mode only
# ---------------------------------------------------------------------------

TEMPORAL_TOLERANCE_SEC = 2.0


def compute_tf_score(aligned: list[dict]) -> dict:
    """
    For each predicted feedback item, check if a ground truth feedback
    exists within ±TEMPORAL_TOLERANCE_SEC of the predicted timestamp.
    Reports precision, recall, F1 across all videos.
    """
    log.info("Computing T-F Score (temporal-feedback) …")

    total_tp = 0
    total_pred = 0
    total_gt = 0
    per_video = {}

    for entry in aligned:
        pred_ts = entry.get("predicted_timestamps", [])
        gt_ts = entry.get("gt_timestamps", [])

        if not isinstance(pred_ts, list):
            pred_ts = []
        if not isinstance(gt_ts, list):
            gt_ts = []

        n_pred = len(pred_ts)
        n_gt = len(gt_ts)
        total_pred += n_pred
        total_gt += n_gt

        # Greedy matching: each GT timestamp can only match one prediction
        gt_matched = [False] * n_gt
        tp = 0

        for pt in pred_ts:
            for j, gtt in enumerate(gt_ts):
                if not gt_matched[j] and abs(pt - gtt) <= TEMPORAL_TOLERANCE_SEC:
                    tp += 1
                    gt_matched[j] = True
                    break

        total_tp += tp

        prec = tp / n_pred if n_pred > 0 else 0.0
        rec = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_video[entry["video_path"]] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "n_pred": n_pred,
            "n_gt": n_gt,
        }

    global_prec = total_tp / total_pred if total_pred > 0 else 0.0
    global_rec = total_tp / total_gt if total_gt > 0 else 0.0
    global_f1 = (
        2 * global_prec * global_rec / (global_prec + global_rec)
        if (global_prec + global_rec) > 0
        else 0.0
    )

    return {
        "precision": round(global_prec, 6),
        "recall": round(global_rec, 6),
        "f1": round(global_f1, 6),
        "total_tp": total_tp,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "tolerance_sec": TEMPORAL_TOLERANCE_SEC,
        "per_video": per_video,
    }


# ---------------------------------------------------------------------------
# Inference performance metrics
# ---------------------------------------------------------------------------

def _percentile(arr: list[float], p: float) -> float:
    return float(np.percentile(arr, p)) if arr else 0.0


def compute_performance_metrics(aligned: list[dict]) -> dict:
    log.info("Computing inference performance metrics …")

    ttft, ttlt, tps, tokens = [], [], [], []
    missing = 0

    for entry in aligned:
        meta = entry.get("inference_metadata")
        if not meta:
            missing += 1
            continue
        if meta.get("time_to_first_token_ms") is not None:
            ttft.append(float(meta["time_to_first_token_ms"]))
        if meta.get("time_to_last_token_ms") is not None:
            ttlt.append(float(meta["time_to_last_token_ms"]))
        if meta.get("tokens_per_second") is not None:
            tps.append(float(meta["tokens_per_second"]))
        if meta.get("total_tokens_used") is not None:
            tokens.append(float(meta["total_tokens_used"]))

    if missing:
        log.warning(f"  inference_metadata missing for {missing} video(s) — skipped")

    def stats(arr: list[float]) -> dict:
        if not arr:
            return {"mean": None, "p50": None, "p95": None, "p99": None}
        return {
            "mean": round(float(np.mean(arr)), 3),
            "p50": round(_percentile(arr, 50), 3),
            "p95": round(_percentile(arr, 95), 3),
            "p99": round(_percentile(arr, 99), 3),
        }

    return {
        "time_to_first_token_ms": stats(ttft),
        "time_to_last_token_ms": stats(ttlt),
        "tokens_per_second": stats(tps),
        "total_tokens_used": {
            "mean": round(float(np.mean(tokens)), 3) if tokens else None,
            "total": int(sum(tokens)) if tokens else None,
        },
        "videos_with_metadata": len(ttft),
        "videos_missing_metadata": missing,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_report(
    model_name: str,
    mode: str,
    aligned: list[dict],
    meteor: dict,
    rouge: dict,
    bert: dict,
    tf: dict | None,
    perf: dict,
):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    def table(headers: list[str], rows: list[list]) -> str:
        if use_tabulate:
            return tabulate(rows, headers=headers, tablefmt="rounded_outline")
        # Fallback plain text
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]
        sep = "  ".join("-" * w for w in col_widths)
        hdr = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        body = "\n".join(
            "  ".join(str(r[i]).ljust(w) for i, w in enumerate(col_widths))
            for r in rows
        )
        return f"{hdr}\n{sep}\n{body}"

    print("\n" + "=" * 70)
    print("  QEVD-FIT-COACH EVALUATION REPORT")
    print("=" * 70)
    print(f"  Model          : {model_name}")
    print(f"  Evaluation date: {date.today().isoformat()}")
    print(f"  Mode           : {mode}")
    print(f"  Videos matched : {len(aligned)}")
    print("=" * 70)

    # ---- Text quality metrics ----
    print("\n[Text Quality Metrics]")
    rows = [
        ["METEOR",    _fmt(meteor["mean"]),         "—",              "—",              "—"],
        ["ROUGE-L F1",_fmt(rouge["mean_f1"]),        "—",              "—",              "—"],
        ["BERTScore P",_fmt(bert["mean_precision"]), "—",              "—",              "—"],
        ["BERTScore R",_fmt(bert["mean_recall"]),    "—",              "—",              "—"],
        ["BERTScore F1",_fmt(bert["mean_f1"]),       "—",              "—",              "—"],
    ]
    print(table(["Metric", "Mean", "P50", "P95", "P99"], rows))

    # ---- T-F Score (long mode only) ----
    if tf is not None:
        print("\n[Temporal-Feedback (T-F) Score]")
        rows = [
            ["Precision", _fmt(tf["precision"])],
            ["Recall",    _fmt(tf["recall"])],
            ["F1",        _fmt(tf["f1"])],
            ["TP / Pred / GT",
             f"{tf['total_tp']} / {tf['total_pred']} / {tf['total_gt']}"],
            ["Tolerance", f"±{tf['tolerance_sec']}s"],
        ]
        print(table(["Metric", "Value"], rows))

    # ---- Inference performance ----
    print("\n[Inference Performance]")

    def perf_row(label: str, key: str) -> list:
        s = perf.get(key, {})
        return [label, _fmt(s.get("mean")), _fmt(s.get("p50")),
                _fmt(s.get("p95")), _fmt(s.get("p99"))]

    tok_info = perf.get("total_tokens_used", {})
    rows = [
        perf_row("TTFT (ms)",      "time_to_first_token_ms"),
        perf_row("TTLT (ms)",      "time_to_last_token_ms"),
        perf_row("Tokens/sec",     "tokens_per_second"),
        ["Total tokens",
         _fmt(tok_info.get("total")), "—", "—", "—"],
        ["Tokens/video (mean)",
         _fmt(tok_info.get("mean")), "—", "—", "—"],
    ]
    print(table(["Metric", "Mean", "P50", "P95", "P99"], rows))
    print(
        f"  (metadata present for {perf['videos_with_metadata']} / "
        f"{len(aligned)} videos)"
    )

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Per-video CSV + JSON report
# ---------------------------------------------------------------------------

def build_per_video_rows(
    aligned: list[dict],
    meteor: dict,
    rouge: dict,
    bert: dict,
    tf: dict | None,
) -> list[dict]:
    rows = []
    for entry in aligned:
        vp = entry["video_path"]
        brt = bert["per_video"].get(vp) or {}
        tf_v = (tf["per_video"].get(vp) or {}) if tf else {}

        row = {
            "video_path": vp,
            "meteor": meteor["per_video"].get(vp),
            "rouge_l_f1": rouge["per_video"].get(vp),
            "bertscore_precision": brt.get("precision"),
            "bertscore_recall": brt.get("recall"),
            "bertscore_f1": brt.get("f1"),
        }
        if tf is not None:
            row["tf_precision"] = tf_v.get("precision")
            row["tf_recall"] = tf_v.get("recall")
            row["tf_f1"] = tf_v.get("f1")
            row["tf_n_pred"] = tf_v.get("n_pred")
            row["tf_n_gt"] = tf_v.get("n_gt")

        meta = entry.get("inference_metadata") or {}
        row["ttft_ms"] = meta.get("time_to_first_token_ms")
        row["ttlt_ms"] = meta.get("time_to_last_token_ms")
        row["tokens_per_second"] = meta.get("tokens_per_second")
        row["total_tokens_used"] = meta.get("total_tokens_used")

        rows.append(row)
    return rows


def save_outputs(
    output_dir: Path,
    model_name: str,
    mode: str,
    aligned: list[dict],
    meteor: dict,
    rouge: dict,
    bert: dict,
    tf: dict | None,
    perf: dict,
    results_path: Path,
    gt_path: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- metrics_report.json ----
    report = {
        "model_name": model_name,
        "evaluation_date": date.today().isoformat(),
        "mode": mode,
        "results_file": str(results_path),
        "ground_truth_file": str(gt_path),
        "total_videos_evaluated": len(aligned),
        "metrics": {
            "meteor": {k: v for k, v in meteor.items() if k != "per_video"},
            "rouge_l": {k: v for k, v in rouge.items() if k != "per_video"},
            "bertscore": {k: v for k, v in bert.items() if k != "per_video"},
            "tf_score": (
                {k: v for k, v in tf.items() if k != "per_video"}
                if tf else None
            ),
            "inference_performance": perf,
        },
    }
    report_path = output_dir / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved metrics report → {report_path}")

    # ---- per_video_metrics.csv ----
    rows = build_per_video_rows(aligned, meteor, rouge, bert, tf)
    csv_path = output_dir / "per_video_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    log.info(f"Saved per-video CSV → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    log.info(f"Model: {args.model_name}  |  Mode: {args.mode}")
    log.info(f"Results file     : {args.results_file}")
    log.info(f"Ground truth file: {args.ground_truth_file}")

    # Load
    results = load_results(args.results_file)
    gt = load_ground_truth(args.ground_truth_file, args.mode)
    aligned = align(results, gt, args.mode)

    if not aligned:
        log.error("No aligned videos to evaluate. Check that video_path values match between results and ground truth.")
        sys.exit(1)

    # Metrics
    meteor = compute_meteor(aligned)
    rouge = compute_rouge(aligned)
    bert = compute_bertscore(aligned)
    tf = compute_tf_score(aligned) if args.mode == "long" else None
    perf = compute_performance_metrics(aligned)

    # Output
    print_report(
        model_name=args.model_name,
        mode=args.mode,
        aligned=aligned,
        meteor=meteor,
        rouge=rouge,
        bert=bert,
        tf=tf,
        perf=perf,
    )

    save_outputs(
        output_dir=args.output_dir,
        model_name=args.model_name,
        mode=args.mode,
        aligned=aligned,
        meteor=meteor,
        rouge=rouge,
        bert=bert,
        tf=tf,
        perf=perf,
        results_path=args.results_file,
        gt_path=args.ground_truth_file,
    )

    # Coverage summary
    n_pred = len(results)
    n_gt = len(gt)
    n_matched = len(aligned)
    print(f"Coverage: {n_matched} matched / {n_pred} predicted / {n_gt} ground truth")


if __name__ == "__main__":
    main()

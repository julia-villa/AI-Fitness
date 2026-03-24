#!/usr/bin/env python3
"""
QEVD-FIT-COACH Dataset Preprocessing Script
============================================
Parts:
  A - Load benchmark split from HuggingFace via FiftyOne
  B - Load full training split from raw files
  C - MediaPipe pose extraction (landmarks + joint angles)
  D - Fine-tuning JSONL export
  E - Organised clip symlink/copy structure
  F - Summary and FiftyOne App launch command

Usage:
    python preprocess_qevd.py \
        --dataset_root /path/to/combined \
        --output_dir /path/to/output \
        [--split_filter all|train|test] \
        [--no-symlinks] \
        [--workers 4] \
        [--skip-pose] \
        [--skip-export] \
        [--skip-train]
"""

import argparse
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess QEVD-FIT-COACH dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Path to the combined/ directory containing raw training files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write JSONL exports, organised clips, pose files, errors.log",
    )
    parser.add_argument(
        "--split_filter",
        choices=["train", "test", "all"],
        default="all",
        help="Which split(s) to load from raw files (Part B). Default: all",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        dest="no_symlinks",
        help="Copy clips instead of symlinking in Part E",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for pose extraction (default: 4)",
    )
    parser.add_argument(
        "--skip-pose",
        action="store_true",
        dest="skip_pose",
        help="Skip Part C (MediaPipe pose extraction)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        dest="skip_export",
        help="Skip Part D (JSONL export)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        dest="skip_train",
        help="Skip Part B (do not load raw training files)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PART A — Load benchmark split from HuggingFace
# ---------------------------------------------------------------------------

def load_benchmark_dataset():
    """Load the QEVD benchmark split via FiftyOne HuggingFace integration."""
    log.info("=== PART A: Loading benchmark split from HuggingFace ===")

    import fiftyone as fo
    from fiftyone.utils.huggingface import load_from_hub

    # Delete existing dataset to allow clean reload
    if fo.dataset_exists("qevd-fit-coach-benchmark"):
        log.info("Dataset 'qevd-fit-coach-benchmark' already exists — reusing.")
        dataset = fo.load_dataset("qevd-fit-coach-benchmark")
    else:
        log.info("Downloading benchmark dataset from Voxel51/qualcomm-exercise-video-dataset-benchmark …")
        dataset = load_from_hub("Voxel51/qualcomm-exercise-video-dataset-benchmark")
        dataset.name = "qevd-fit-coach-benchmark"
        dataset.persistent = True

    log.info(f"Benchmark dataset loaded: {len(dataset)} samples")

    # Tag all samples with "benchmark"
    untagged = dataset.match(~fo.ViewField("tags").contains("benchmark"))
    if len(untagged) > 0:
        dataset.tag_samples("benchmark")
        log.info("Tagged all samples with 'benchmark'")
    else:
        log.info("Samples already tagged 'benchmark'")

    return dataset


# ---------------------------------------------------------------------------
# PART B — Load training split from raw files
# ---------------------------------------------------------------------------

def _safe_load_json(path: Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_label(label: str) -> tuple[str, str, str]:
    """
    Parse label string:
      "<exercise_name> (<general_variant>) - <fine_grained_variant>"
    Returns (exercise_name, general_variant, fine_grained_variant).
    Falls back gracefully if the format doesn't match.
    """
    pattern = r"^(.+?)\s*\((.+?)\)\s*-\s*(.+)$"
    m = re.match(pattern, label.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    # Fallback: return whole label as exercise name
    return label.strip(), "", ""


def load_train_dataset(dataset_root: Path, split_filter: str):
    """Build 'qevd-fit-coach-train' FiftyOne dataset from raw files."""
    log.info("=== PART B: Loading training split from raw files ===")

    import fiftyone as fo

    combined = dataset_root

    # ---- Load JSON annotation files ----------------------------------------
    fine_grained_path = combined / "fine_grained_labels.json"
    feedbacks_short_path = combined / "feedbacks_short_clips.json"
    feedbacks_long_path = combined / "feedbacks_long_range.json"
    questions_path = combined / "questions.json"

    fine_grained_labels = _safe_load_json(fine_grained_path)
    log.info(f"  fine_grained_labels.json: {len(fine_grained_labels)} entries")

    feedbacks_short_raw = _safe_load_json(feedbacks_short_path)
    log.info(f"  feedbacks_short_clips.json: {len(feedbacks_short_raw)} entries")

    feedbacks_long_raw = _safe_load_json(feedbacks_long_path)
    log.info(f"  feedbacks_long_range.json: {len(feedbacks_long_raw)} entries")

    questions_raw = _safe_load_json(questions_path)
    log.info(f"  questions.json: {len(questions_raw)} entries")

    # ---- Index by video_path ------------------------------------------------
    # feedbacks_short: keyed by video_path
    feedbacks_short_idx: dict[str, list[str]] = {}
    for entry in feedbacks_short_raw:
        vp = entry.get("video_path", "")
        fb = entry.get("feedback", [])
        if isinstance(fb, str):
            fb = [fb]
        feedbacks_short_idx.setdefault(vp, []).extend(fb)

    # questions: keyed by video_path
    questions_idx: dict[str, dict] = {}
    for entry in questions_raw:
        vp = entry.get("video_path", "")
        questions_idx[vp] = entry

    # ---- Create or reuse dataset --------------------------------------------
    if fo.dataset_exists("qevd-fit-coach-train"):
        log.info("Dataset 'qevd-fit-coach-train' already exists — deleting and rebuilding.")
        fo.delete_dataset("qevd-fit-coach-train")

    dataset = fo.Dataset("qevd-fit-coach-train")
    dataset.persistent = True

    # ---- Short clips --------------------------------------------------------
    log.info("  Building short clip samples …")
    short_samples = []

    for entry in tqdm(fine_grained_labels, desc="Short clips", unit="clip"):
        video_path_rel = entry.get("video_path", "")
        split = entry.get("split", "train")

        if split_filter != "all" and split != split_filter:
            continue

        # Resolve absolute path
        video_path = combined / video_path_rel
        if not video_path.exists():
            # Try treating video_path_rel as already absolute
            video_path = Path(video_path_rel)

        labels = entry.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        labels_descriptive = entry.get("labels_descriptive", [])
        if isinstance(labels_descriptive, str):
            labels_descriptive = [labels_descriptive]

        # Parse exercise meta from first label
        exercise_name, general_variant, fine_grained_variant = ("", "", "")
        if labels:
            exercise_name, general_variant, fine_grained_variant = _parse_label(labels[0])

        feedback = feedbacks_short_idx.get(video_path_rel, [])

        q_entry = questions_idx.get(video_path_rel, {})
        questions_high_level = q_entry.get("high_level", [])
        questions_fine_grain = q_entry.get("fine_grain", [])

        sample = fo.Sample(filepath=str(video_path))
        sample["labels"] = labels
        sample["labels_descriptive"] = labels_descriptive
        sample["exercise_name"] = exercise_name
        sample["general_variant"] = general_variant
        sample["fine_grained_variant"] = fine_grained_variant
        sample["feedback"] = feedback
        sample["questions_high_level"] = questions_high_level
        sample["questions_fine_grain"] = questions_fine_grain
        sample["clip_type"] = "short"

        short_samples.append((sample, split))

    log.info(f"  {len(short_samples)} short clip samples built")

    # ---- Long-range videos --------------------------------------------------
    log.info("  Building long-range video samples …")
    long_samples = []

    for entry in tqdm(feedbacks_long_raw, desc="Long-range videos", unit="vid"):
        video_file = entry.get("long_range_video_file", "")
        video_path = combined / video_file
        if not video_path.exists():
            video_path = Path(video_file)

        feedback = entry.get("feedback", [])
        if isinstance(feedback, str):
            feedback = [feedback]
        feedback_timestamps = entry.get("feedback_timestamps", [])
        is_transition = entry.get("is_transition", [])

        # video_timestamps: optional field describing when each feedback occurs
        video_timestamps = entry.get("video_timestamps", [])

        sample = fo.Sample(filepath=str(video_path))
        sample["feedback"] = feedback
        sample["feedback_timestamps"] = feedback_timestamps
        sample["video_timestamps"] = video_timestamps
        sample["is_transition"] = is_transition
        sample["clip_type"] = "long_range"

        long_samples.append((sample, "train"))

    log.info(f"  {len(long_samples)} long-range video samples built")

    # ---- Add all samples to dataset and tag --------------------------------
    all_pairs = short_samples + long_samples
    samples_to_add = [s for s, _ in all_pairs]
    splits_list = [sp for _, sp in all_pairs]

    dataset.add_samples(samples_to_add)

    # Re-fetch in order to apply tags (add_samples returns in order)
    view = dataset.view()
    for sample, split_tag in zip(view.iter_samples(autosave=True), splits_list):
        sample.tags = list(set(sample.tags) | {split_tag})

    log.info(f"Train dataset built: {len(dataset)} samples total")
    return dataset


# ---------------------------------------------------------------------------
# PART C — MediaPipe pose extraction (worker function + dispatcher)
# ---------------------------------------------------------------------------

# Joint angle definitions: (landmark_A, vertex, landmark_B)
# The angle is computed at the vertex between rays vertex->A and vertex->B.
#
# MediaPipe Pose landmark indices (0-based):
#   11=left_shoulder,  12=right_shoulder
#   13=left_elbow,     14=right_elbow
#   15=left_wrist,     16=right_wrist
#   23=left_hip,       24=right_hip
#   25=left_knee,      26=right_knee
#   27=left_ankle,     28=right_ankle
#
# NOTE: The three-point definitions below (which landmarks bound each joint)
# are anatomical conventions — verify against the QEVD paper if exact angle
# semantics matter for your model.


#DOUBLE CHECK VALUES WITH RESEARCH & MEDIAPIPE
JOINT_ANGLE_DEFS = {
    "LEFT_KNEE":       (23, 25, 27),  # left_hip    -> left_knee    -> left_ankle
    "RIGHT_KNEE":      (24, 26, 28),  # right_hip   -> right_knee   -> right_ankle
    "LEFT_HIP":        (25, 23, 11),  # left_knee   -> left_hip     -> left_shoulder
    "RIGHT_HIP":       (26, 24, 12),  # right_knee  -> right_hip    -> right_shoulder
    "LEFT_ELBOW":      (11, 13, 15),  # left_shoulder  -> left_elbow  -> left_wrist
    "RIGHT_ELBOW":     (12, 14, 16),  # right_shoulder -> right_elbow -> right_wrist
    "LEFT_SHOULDER":   (13, 11, 23),  # left_elbow  -> left_shoulder -> left_hip
    "RIGHT_SHOULDER":  (14, 12, 24),  # right_elbow -> right_shoulder -> right_hip
}


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute angle (degrees) at vertex b, between rays b->a and b->c.
    Uses only x,y,z (ignores visibility).
    """
    ba = a[:3] - b[:3]
    bc = c[:3] - b[:3]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def _extract_pose_for_video(args: tuple) -> dict:
    """
    Worker function — runs in a subprocess.
    args = (video_path, pose_npy_path, joint_angles_path)
    Returns dict with keys: status, video_path, num_frames, fps, error
    """
    video_path, pose_npy_path, joint_angles_path = args

    try:
        import cv2
        import mediapipe as mp

        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "status": "failed",
                "video_path": str(video_path),
                "error": f"Cannot open video: {video_path}",
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        all_landmarks = []   # list of np.ndarray shape (33, 4) per frame
        joint_angles_all = {}  # "frame_N": { joint: float }

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    arr = np.array(
                        [[l.x, l.y, l.z, l.visibility] for l in lm],
                        dtype=np.float32,
                    )  # shape (33, 4)
                else:
                    arr = np.zeros((33, 4), dtype=np.float32)

                all_landmarks.append(arr)

                # Compute joint angles for this frame
                frame_angles = {}
                for joint_name, (idx_a, idx_v, idx_b) in JOINT_ANGLE_DEFS.items():
                    a = arr[idx_a]
                    v = arr[idx_v]
                    b = arr[idx_b]
                    frame_angles[joint_name] = round(_angle_between(a, v, b), 4)
                joint_angles_all[f"frame_{frame_idx}"] = frame_angles

                frame_idx += 1

        cap.release()

        num_frames = len(all_landmarks)
        if num_frames == 0:
            return {
                "status": "failed",
                "video_path": str(video_path),
                "error": "No frames decoded",
            }

        # Save landmarks as .npy: shape (num_frames, 33, 4)
        landmarks_array = np.stack(all_landmarks, axis=0)
        pose_npy_path = Path(pose_npy_path)
        pose_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(pose_npy_path), landmarks_array)

        # Save joint angles as JSON
        joint_angles_path = Path(joint_angles_path)
        joint_angles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(joint_angles_path, "w") as f:
            json.dump(joint_angles_all, f)

        return {
            "status": "success",
            "video_path": str(video_path),
            "pose_npy_path": str(pose_npy_path),
            "joint_angles_path": str(joint_angles_path),
            "num_frames": num_frames,
            "fps": fps,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "video_path": str(video_path),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def run_pose_extraction(datasets: list, output_dir: Path, workers: int):
    """
    Part C: run MediaPipe on all samples across all datasets.
    Skips samples where pose_npy_path is already set and file exists.
    """
    log.info("=== PART C: MediaPipe pose extraction ===")

    import fiftyone as fo

    errors_log = output_dir / "errors.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work list
    work_items = []  # (video_path, pose_npy_path, angles_path, dataset_name, sample_id)

    for dataset in datasets:
        for sample in dataset.iter_samples():
            # Skip if already processed
            existing_npy = sample.get_field("pose_npy_path")
            if existing_npy and Path(existing_npy).exists():
                continue

            video_path = Path(sample.filepath)
            stem = video_path.with_suffix("")
            pose_npy_path = Path(str(stem) + "_pose.npy")
            angles_path = Path(str(stem) + "_angles.json")

            work_items.append(
                (str(video_path), str(pose_npy_path), str(angles_path),
                 dataset.name, sample.id)
            )

    total = len(work_items)
    log.info(f"  Samples to process: {total}")

    if total == 0:
        log.info("  Nothing to process — all samples already have pose data.")
        return {"processed": 0, "skipped": 0, "failed": 0}

    # Count pre-existing (skipped)
    skipped = sum(
        1 for ds in datasets
        for s in ds.iter_samples()
        if s.get_field("pose_npy_path") and Path(s.get_field("pose_npy_path")).exists()
    )

    # Run extraction in pool
    extract_args = [(v, p, a) for v, p, a, _, _ in work_items]

    processed = 0
    failed = 0
    failed_entries = []

    # Build a lookup: video_path -> (dataset_name, sample_id)
    path_to_sample = {v: (dn, sid) for v, _, _, dn, sid in work_items}

    # Load FiftyOne datasets once keyed by name
    ds_map = {ds.name: ds for ds in datasets}

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_extract_pose_for_video, extract_args),
            total=total,
            desc="Pose extraction",
            unit="video",
        ):
            vp = result["video_path"]
            if result["status"] == "success":
                processed += 1
                # Update FiftyOne sample
                ds_name, sid = path_to_sample.get(vp, (None, None))
                if ds_name and sid:
                    try:
                        sample = ds_map[ds_name][sid]
                        sample["pose_npy_path"] = result["pose_npy_path"]
                        sample["joint_angles_path"] = result["joint_angles_path"]
                        sample["num_frames"] = result["num_frames"]
                        sample["fps"] = result["fps"]
                        sample.save()
                    except Exception as e:
                        log.warning(f"Could not update sample {sid}: {e}")
            else:
                failed += 1
                failed_entries.append(result)

    # Write errors log
    if failed_entries:
        with open(errors_log, "a") as f:
            for entry in failed_entries:
                f.write(f"[FAILED] {entry['video_path']}\n")
                f.write(f"  {entry.get('error', 'unknown error')}\n\n")
        log.warning(f"  {failed} failures logged to {errors_log}")

    log.info(f"  Pose extraction complete — processed: {processed}, skipped: {skipped}, failed: {failed}")
    return {"processed": processed, "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# PART D — Fine-tuning JSONL export
# ---------------------------------------------------------------------------

def export_jsonl(train_dataset, output_dir: Path):
    """
    Part D: export train + test JSONL files from 'qevd-fit-coach-train'.
    Only includes short clips where pose extraction succeeded.
    """
    log.info("=== PART D: Exporting fine-tuning JSONL ===")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train.jsonl"
    test_out = output_dir / "test.jsonl"

    train_lines = 0
    test_lines = 0

    with open(train_out, "w") as ftrain, open(test_out, "w") as ftest:
        for sample in tqdm(
            train_dataset.iter_samples(), desc="Exporting JSONL", unit="sample"
        ):
            # Only short clips
            if sample.get_field("clip_type") != "short":
                continue

            # Only samples with successful pose extraction
            angles_path = sample.get_field("joint_angles_path")
            if not angles_path or not Path(angles_path).exists():
                continue

            with open(angles_path, "r") as f:
                joint_angles = json.load(f)

            labels_descriptive = sample.get_field("labels_descriptive") or []
            label_descriptive = labels_descriptive[0] if labels_descriptive else ""

            # Determine split from tags
            split = "train"
            tags = sample.tags or []
            if "test" in tags:
                split = "test"
            elif "train" in tags:
                split = "train"

            record = {
                "video_path": sample.filepath,
                "exercise_name": sample.get_field("exercise_name") or "",
                "general_variant": sample.get_field("general_variant") or "",
                "fine_grained_variant": sample.get_field("fine_grained_variant") or "",
                "label_descriptive": label_descriptive,
                "joint_angles": joint_angles,
                "feedback": sample.get_field("feedback") or [],
                "split": split,
            }

            line = json.dumps(record, ensure_ascii=False)
            if split == "test":
                ftest.write(line + "\n")
                test_lines += 1
            else:
                ftrain.write(line + "\n")
                train_lines += 1

    log.info(f"  JSONL export complete — train: {train_lines} lines, test: {test_lines} lines")
    log.info(f"  Written to: {train_out}, {test_out}")
    return train_lines, test_lines


# ---------------------------------------------------------------------------
# PART E — Organised clip structure
# ---------------------------------------------------------------------------

def _sanitize_path_component(s: str) -> str:
    """Replace characters that are unsafe in directory/file names."""
    s = s.strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def organise_clips(train_dataset, output_dir: Path, use_symlinks: bool):
    """
    Part E: create organised clip structure under output_dir/organized_clips/.
    """
    log.info("=== PART E: Organising clip structure ===")

    base = output_dir / "organized_clips"
    base.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for sample in tqdm(
        train_dataset.iter_samples(), desc="Organising clips", unit="clip"
    ):
        if sample.get_field("clip_type") != "short":
            continue

        src = Path(sample.filepath)
        if not src.exists():
            skipped += 1
            continue

        exercise = _sanitize_path_component(sample.get_field("exercise_name") or "unknown")
        general = _sanitize_path_component(sample.get_field("general_variant") or "unknown")
        fine = _sanitize_path_component(sample.get_field("fine_grained_variant") or "unknown")

        dest_dir = base / exercise / general / fine
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        if dest.exists() or dest.is_symlink():
            skipped += 1
            continue

        if use_symlinks:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)

        created += 1

    action = "symlinks" if use_symlinks else "copies"
    log.info(f"  Organised clips: {created} {action} created, {skipped} skipped")


# ---------------------------------------------------------------------------
# PART F — Summary
# ---------------------------------------------------------------------------

def print_summary(benchmark_ds, train_ds, pose_stats, jsonl_stats):
    """Part F: print human-readable summary."""
    import fiftyone as fo

    log.info("=== PART F: Summary ===")

    print("\n" + "=" * 70)
    print("QEVD-FIT-COACH PREPROCESSING SUMMARY")
    print("=" * 70)

    # Benchmark
    bm_count = len(benchmark_ds) if benchmark_ds else 0
    print(f"\n[Benchmark dataset] 'qevd-fit-coach-benchmark'")
    print(f"  Samples : {bm_count}")
    print(f"  Tags    : benchmark")

    # Train
    if train_ds:
        total_train = len(train_ds)
        print(f"\n[Train dataset] 'qevd-fit-coach-train'")
        print(f"  Samples total : {total_train}")

        # Breakdown by split tag
        for tag in ["train", "test"]:
            n = len(train_ds.match_tags(tag))
            print(f"    split={tag}: {n}")

        # Breakdown by exercise_name (short clips only)
        short_view = train_ds.match(fo.ViewField("clip_type") == "short")
        exercise_counts: dict[str, int] = {}
        for s in short_view.iter_samples():
            ex = s.get_field("exercise_name") or "unknown"
            exercise_counts[ex] = exercise_counts.get(ex, 0) + 1
        print(f"\n  Exercise breakdown (short clips):")
        for ex, cnt in sorted(exercise_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {ex}: {cnt}")
        if len(exercise_counts) > 20:
            print(f"    … and {len(exercise_counts) - 20} more exercises")

    # Pose extraction
    if pose_stats:
        print(f"\n[Pose extraction]")
        print(f"  Processed : {pose_stats['processed']}")
        print(f"  Skipped   : {pose_stats['skipped']}")
        print(f"  Failed    : {pose_stats['failed']}")

    # JSONL
    if jsonl_stats:
        train_lines, test_lines = jsonl_stats
        print(f"\n[JSONL export]")
        print(f"  train.jsonl : {train_lines} lines")
        print(f"  test.jsonl  : {test_lines} lines")

    # App launch command
    print(f"\n[Launch FiftyOne App]")
    print(f"  python -c \"import fiftyone as fo; fo.launch_app(fo.load_dataset('qevd-fit-coach-benchmark'))\"")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_ds = None
    train_ds = None
    pose_stats = None
    jsonl_stats = None

    # ---- Part A ----
    benchmark_ds = load_benchmark_dataset()

    # ---- Part B ----
    if not args.skip_train:
        if args.dataset_root is None:
            log.error("--dataset_root is required for Part B (loading training data).")
            sys.exit(1)
        if not args.dataset_root.exists():
            log.error(f"--dataset_root does not exist: {args.dataset_root}")
            sys.exit(1)
        train_ds = load_train_dataset(args.dataset_root, args.split_filter)
    else:
        log.info("Skipping Part B (--skip-train)")
        import fiftyone as fo
        if fo.dataset_exists("qevd-fit-coach-train"):
            train_ds = fo.load_dataset("qevd-fit-coach-train")
            log.info("Loaded existing 'qevd-fit-coach-train' dataset")

    # ---- Part C ----
    if not args.skip_pose:
        datasets_for_pose = [d for d in [benchmark_ds, train_ds] if d is not None]
        pose_stats = run_pose_extraction(datasets_for_pose, args.output_dir, args.workers)
    else:
        log.info("Skipping Part C (--skip-pose)")

    # ---- Part D ----
    if not args.skip_export:
        if train_ds is None:
            log.warning("No train dataset available — skipping JSONL export.")
        else:
            jsonl_stats = export_jsonl(train_ds, args.output_dir)
    else:
        log.info("Skipping Part D (--skip-export)")

    # ---- Part E ----
    if train_ds is not None:
        organise_clips(train_ds, args.output_dir, use_symlinks=not args.no_symlinks)
    else:
        log.info("No train dataset — skipping Part E")

    # ---- Part F ----
    print_summary(benchmark_ds, train_ds, pose_stats, jsonl_stats)


if __name__ == "__main__":
    main()

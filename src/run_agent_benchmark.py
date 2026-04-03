import os
import sys
import cv2
import torch
import time
from tqdm import tqdm

# --- PATH FIX ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the VLM logic and the Stage 3 Helpers
from auditor import vlm_auditor  
from stage3.manifest import load_segment_manifest
from stage3.predictions import save_predictions # <--- Utilizing your helper now

def run_benchmark():
    # 1. Setup
    manifest_path = "eval/benchmark_manifest.json"
    output_path = "eval/predictions.json"
    debug_folder = "eval/debug_frames"
    
    if not os.path.exists(manifest_path):
        print(f"Error: Could not find {manifest_path}")
        return
    if not os.path.exists(debug_folder): 
        os.makedirs(debug_folder)

    segments = load_segment_manifest(manifest_path)
    all_predictions = []

    print(f"--- Starting Benchmark: {len(segments)} Segments ---")

    # 2. The Benchmark Loop
    for seg in tqdm(segments, desc="Processing"):
        if not os.path.exists(seg.video_path):
            continue

        cap = cv2.VideoCapture(seg.video_path)
        if not cap.isOpened():
            continue

        # Jump to middle of the segment
        duration = seg.exercise_end_timestamp - seg.exercise_start_timestamp
        mid_relative = duration / 2
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame = int(mid_relative * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        cap.release() 
        
        feedback_text = "Inference Error"
        wall_start = time.perf_counter()

        if success:
            temp_img_path = f"{debug_folder}/{seg.segment_id.replace(':', '_')}.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            try:
                # Calls Moondream 2 and extracts ALL metrics
                result = vlm_auditor(temp_img_path, seg.exercise_name)
                feedback_text = result.get("feedback", "No feedback provided")
                prompt_tokens = result.get("prompt_tokens", 0)
                generated_tokens = result.get("generated_tokens", 0)
                ttft_sec = result.get("ttft_sec", 0.0)
            except Exception as e:
                print(f"\n[ERROR] VLM failed: {e}")
                prompt_tokens = 0
                generated_tokens = 0
                ttft_sec = 0.0
        
        wall_time = time.perf_counter() - wall_start

        # --- 3. RECORDING (Strict Format) ---
        prediction_entry = {
            "segment_id": seg.segment_id,
            "video_id": getattr(seg, 'video_id', "unknown"),
            "exercise_name": seg.exercise_name,
            "pred_feedbacks": [feedback_text],
            "pred_feedback_timestamps": [mid_relative],
            "prompt_tokens": prompt_tokens,               # Now pulling real data
            "generated_tokens": generated_tokens,         # Now pulling real data
            "total_tokens": prompt_tokens + generated_tokens, # Calculated dynamically
            "generation_wall_time_sec": wall_time,
            "timing_events": [
                {
                    "ttft_sec": ttft_sec,                 # Now pulling real data
                    "time_to_last_token_sec": wall_time,
                    "text": feedback_text,
                    "generated_token_count": generated_tokens,
                    "feedback_index": 0,
                    "pred_timestamp_sec": mid_relative
                }
            ]
        }
        all_predictions.append(prediction_entry)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Save using the official project helper
    save_predictions(all_predictions, output_path)
    print(f"\nSUCCESS: {len(all_predictions)} results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
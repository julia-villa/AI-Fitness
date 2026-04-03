import cv2
import os
import torch
from auditor import vlm_auditor

# 1. Configuration - Replace this with a real path from your manifest!
VIDEO_PATH = r"src\\data\\combined\\long_range_videos_benchmark\\0011.mp4" 
EXERCISE_NAME = "squats"

def test_single():
    print(f"--- Starting Single Video Test: {EXERCISE_NAME} ---")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found at {VIDEO_PATH}")
        return

    # 2. Extract the middle frame (Same logic as benchmark)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = cap.read()
    cap.release()

    if not success:
        print("ERROR: Could not read frame from video.")
        return

    # 3. Save temp frame
    temp_path = "debug_test_frame.jpg"
    cv2.imwrite(temp_path, frame)
    print(f"Frame extracted and saved to {temp_path}")

    # 4. Run Auditor
    print("Sending to VLM (Moondream 2)...")
    try:
        result = vlm_auditor(temp_path, EXERCISE_NAME)
        print("\n--- VLM RESPONSE ---")
        print(f"Feedback: {result['feedback']}")
        print("--------------------")
    except Exception as e:
        print(f"TEST FAILED: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_single()
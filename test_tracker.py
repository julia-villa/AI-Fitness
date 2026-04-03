import cv2
import time
import os
import threading
from tracker import PoseTracker
from update_coach_logic import update_coach_logic
from auditor import vlm_auditor

# Reverted to original signature
def background_audit(snapshot_path, state, exercise_name):
    state["is_processing_vlm"] = True
    start_time = time.perf_counter()
    
    # Calls the hardcoded auditor
    result = vlm_auditor(snapshot_path, exercise_name)
    
    state["vlm_feedback"] = result["feedback"]
    state["is_processing_vlm"] = False
    print(f"--- [VLM FINISH] Advice received in {time.perf_counter() - start_time:.2f}s ---")

def main():
    tracker = PoseTracker()
    system_mode = "IDENTIFYING" 
    exercise_name = "None"
    
    state = {
        "reps": 0, "phase": "up", "is_anomaly": False, 
        "consecutive_stuck_frames": 0, "vlm_feedback": "Move to identify...",
        "is_processing_vlm": False, "timer": 0.0
    }
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    while True:
        ret, frame = cap.read()
        if not ret: break

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('x'): break 
        if key == ord('r'):
            system_mode = "IDENTIFYING"; exercise_name = "None"
            tracker.history.clear(); state["vlm_feedback"] = "Manual reset."
            continue 

        landmarks, data = tracker.process_frame(frame, exercise_name)
        
        if landmarks:
            if system_mode == "IDENTIFYING":
                prediction = tracker.analyze_movement_signature()
                if prediction: 
                    exercise_name = prediction; system_mode = "COACHING"
                    state["reps"] = 0; state["vlm_feedback"] = f"Locked in: {exercise_name}"

            elif system_mode == "COACHING" and data:
                state = update_coach_logic(state, data, exercise_name)
                
                # Simple Trigger Logic
                is_stretch = any(x in exercise_name for x in ["stretch", "gators", "touchers"])
                trigger_vlm = False
                
                if is_stretch:
                    state["timer"] += 1/30
                    if state["timer"] > 5.0:
                        trigger_vlm = True; state["timer"] = 0
                elif state["is_anomaly"]:
                    trigger_vlm = True

                if trigger_vlm and not state["is_processing_vlm"]:
                    if not os.path.exists("audits"): os.makedirs("audits")
                    snapshot_path = f"audits/audit_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    # Reverted call
                    threading.Thread(target=background_audit, args=(snapshot_path, state, exercise_name)).start()

        # Overlays (simplified for brevity)
        cv2.putText(frame, f"MODE: {exercise_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"COACH: {state['vlm_feedback'][:45]}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Autonomous Situated Agent", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
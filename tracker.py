import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # TEMPORAL MEMORY: Stores the last 45 frames (1.5s at 30fps)
        self.history = deque(maxlen=45) 

    def get_exercise_config(self, exercise_name):
        """The Complete Table 5 Master Registry"""
        configs = {
            # --- WARM-UP ---
            "jumping_jacks": {"joints": [15, 16, 27, 28, 0], "type": "spatial", "label": "Width/Height"},
            "high_knees": {"joints": [23, 25, 24, 26], "type": "height", "label": "Knee Drive"},
            "butt_kickers": {"joints": [25, 27, 26, 28], "type": "height", "label": "Heel Drive"},
            "air_jump_rope": {"joints": [15, 16], "type": "spatial", "label": "Wrist Circle"},
            "good_mornings": {"joints": [11, 23, 25], "type": "angle", "label": "Hip Hinge"},

            # --- MAIN WORKOUT ---
            "push-ups": {"joints": [11, 13, 15], "type": "angle", "label": "Elbow Angle"},
            "plank_taps": {"joints": [15, 16], "type": "spatial", "label": "Hand Tap"},
            "moving_plank": {"joints": [11, 23, 25], "type": "angle", "label": "Core Flatness"},
            "squats": {"joints": [23, 25, 27], "type": "angle", "label": "Knee Angle"},
            "walking_lunges": {"joints": [23, 25, 27], "type": "angle", "label": "Lead Knee"},
            "lunge_jumps": {"joints": [23, 25, 27], "type": "angle", "label": "Explosive Depth"},
            "puddle_jumps": {"joints": [23, 24], "type": "spatial", "label": "Lateral Shift"},
            "mountain_climbers": {"joints": [13, 25, 14, 26], "type": "spatial", "label": "Knee-to-Elbow"},
            "floor_touches": {"joints": [15, 27], "type": "spatial", "label": "Reach"},
            "quick_feet": {"joints": [27, 28], "type": "height", "label": "Foot Cadence"},
            "squat_jumps": {"joints": [23, 25, 27], "type": "angle", "label": "Jump Depth"},
            "squat_kicks": {"joints": [23, 25, 27], "type": "angle", "label": "Knee Flexion"},
            "standing_kicks": {"joints": [23, 25], "type": "height", "label": "Kick Height"},
            "boxing_squat_punches": {"joints": [23, 25, 15, 16], "type": "angle", "label": "Squat-Punch"},

            # --- COOL-DOWN ---
            "deltoid_stretch": {"joints": [13, 14, 11, 12], "type": "vlm_only", "label": "Shoulder Hold"},
            "quad_stretch": {"joints": [25, 27, 26, 28], "type": "vlm_only", "label": "Balance Hold"},
            "shoulder_gators": {"joints": [13, 14], "type": "spatial", "label": "Arm Opening"},
            "toe_touchers": {"joints": [15, 27, 16, 28], "type": "spatial", "label": "Reach"}
        }
        return configs.get(exercise_name, configs["squats"])

    def analyze_movement_signature(self):
        """
        Deduces the exercise family from the 23 candidates using joint variance.
        """
        if len(self.history) < 45: return None
        data = np.array(self.history)
        
        # 1. Orientation Check
        avg_nose_y = np.mean(data[:, 0, 1])
        avg_ankle_y = np.mean((data[:, 27, 1] + data[:, 28, 1]) / 2)
        is_floor_level = abs(avg_nose_y - avg_ankle_y) < 0.4 
        
        # Calculate Variances (How much are things moving?)
        wrist_y_var = np.var(data[:, 15, 1]) + np.var(data[:, 16, 1])
        wrist_x_var = np.var(data[:, 15, 0]) + np.var(data[:, 16, 0])
        ankle_x_var = np.var(data[:, 27, 0]) + np.var(data[:, 28, 0])
        knee_y_var = np.var(data[:, 25, 1]) + np.var(data[:, 26, 1])
        knee_x_var = np.var(data[:, 25, 0]) + np.var(data[:, 26, 0])
        nose_y_var = np.var(data[:, 0, 1]) # Indicates total body vertical movement
        
        # ==========================================
        # --- NEW: X-RAY DEBUG PRINT ---
        # ==========================================
        print(f"\n[DEBUG] Floor Level: {is_floor_level}")
        print(f"[DEBUG] Wrist Y: {wrist_y_var:.4f} | Wrist X: {wrist_x_var:.4f}")
        print(f"[DEBUG] Knee Y: {knee_y_var:.4f} | Nose Y: {nose_y_var:.4f}")
        print(f"[DEBUG] Ankle X: {ankle_x_var:.4f}")
        # ==========================================

        # --- THE CLASSIFICATION TREE ---
        if is_floor_level:
            if wrist_x_var > 0.005: return "plank_taps"
            elif knee_x_var > 0.01: return "mountain_climbers"
            else: return "push-ups" # Default floor base
            
        else: # Standing Exercises
            if wrist_y_var > 0.02 and ankle_x_var > 0.01:
                return "jumping_jacks"
            elif knee_y_var > 0.015 and nose_y_var < 0.005:
                # Knees are coming up, but head stays relatively still
                return "high_knees" 
            elif nose_y_var > 0.008:
                # The whole body is moving up and down
                if ankle_x_var > 0.01: return "puddle_jumps"
                elif np.var(data[:, 27, 1]) > 0.01: return "squat_jumps" # Feet leaving floor
                else: return "squats" # Default deep bend base
            elif wrist_x_var > 0.01 and nose_y_var < 0.002:
                return "shoulder_gators"
            elif wrist_y_var < 0.001 and knee_y_var < 0.001:
                # Total stillness implies a hold/stretch
                return "deltoid_stretch" 
            
            return None # Keep collecting data if it's too ambiguous

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def process_frame(self, frame, exercise_name):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark
        self.history.append([(pt.x, pt.y) for pt in lm])
        
        config = self.get_exercise_config(exercise_name)
        data = None
        
        # Generic Metric Extraction based on the 23 configs
        if config["type"] == "angle":
            ids = config["joints"][:3]
            if all(lm[i].visibility > 0.5 for i in ids):
                p1, p2, p3 = [[lm[i].x, lm[i].y] for i in ids]
                data = {"angle": self.calculate_angle(p1, p2, p3), "label": config["label"]}
                
        elif config["type"] == "spatial":
            if exercise_name == "jumping_jacks":
                hand_y = (lm[15].y + lm[16].y) / 2
                data = {"hand_y_diff": lm[0].y - hand_y, "foot_distance": np.abs(lm[27].x - lm[28].x), "label": config["label"]}
            else:
                p1, p2 = [lm[config["joints"][0]], lm[config["joints"][1]]]
                dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                data = {"dist_val": dist, "label": config["label"]}
                
        elif config["type"] == "height":
            l_val = lm[23].y - lm[config["joints"][1]].y
            r_val = lm[24].y - lm[config["joints"][3]].y if len(config["joints"]) > 2 else l_val
            data = {"height_val": max(l_val, r_val), "label": config["label"]}
            
        elif config["type"] == "vlm_only":
            data = {"label": config["label"], "status": "Auditing..."}
                
        return lm, data
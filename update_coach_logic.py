from typing import TypedDict, Dict, Any

class CoachState(TypedDict):
    reps: int
    phase: str
    is_anomaly: bool
    consecutive_stuck_frames: int
    vlm_feedback: str
    is_processing_vlm: bool
    timer: float
    perfect_reps: int     # NEW: Tracks consecutive good reps
    anomaly_reason: str   # NEW: Tells the VLM what went wrong

# 1.5 seconds at 30 FPS
STUCK_LIMIT = 45 

def update_coach_logic(state: CoachState, data: Dict[str, Any], exercise_name: str) -> CoachState:
    """
    Generalized Policy Engine for all 23 Table 5 exercises.
    Uses generic primitives: angle, dist_val, and height_val.
    """
    if not data:
        return state

    # --- 1. ANGLE-BASED (Squats, Push-ups, Lunges, Good Mornings, etc.) ---
    if "angle" in data:
        angle = data["angle"]
        # Broad thresholds to cover all variants (Squat Jumps, Kicks, Punches)
        thresholds = {
            "push-ups": {"down": 115, "up": 155},
            "bicep_curl": {"down": 70, "up": 155},
            "default": {"down": 110, "up": 160} # Works for all Squat/Lunge variants
        }
        t = thresholds.get(exercise_name, thresholds["default"])
        
        if angle < t["down"] and state["phase"] == "up":
            state["phase"] = "down"
            state["consecutive_stuck_frames"] = 0
        elif angle > t["up"] and state["phase"] == "down":
            state["phase"] = "up"
            state["reps"] += 1
            state["perfect_reps"] += 1  # Streak goes up!
            state["consecutive_stuck_frames"] = 0

        # Anomaly check
        if state["phase"] == "down" and angle < (t["down"] + 15):
            state["consecutive_stuck_frames"] += 1
        else:
            state["consecutive_stuck_frames"] = 0

    # --- 2. SPATIAL-BASED (Jumping Jacks, Plank Taps, Mountain Climbers, Toe Touchers) ---
    elif "hand_y_diff" in data: # Specific to Jumping Jacks
        is_open = data["hand_y_diff"] > 0.05 and data["foot_distance"] > 0.25
        if is_open and state["phase"] == "closed":
            state["phase"] = "open"
        elif not is_open and state["phase"] == "open":
            state["phase"] = "closed"
            state["reps"] += 1
            state["perfect_reps"] += 1  # Streak goes up!
            
    elif "dist_val" in data: # Generic Distance (Plank Taps, Mountain Climbers, etc.)
        val = data["dist_val"]
        # Logic: Rep counts when distance gets small (touching/climbing)
        if val < 0.30 and state["phase"] == "down":
            state["phase"] = "up"
        elif val > 0.45 and state["phase"] == "up":
            state["phase"] = "down"
            state["reps"] += 1
            state["perfect_reps"] += 1  # Streak goes up!

    # --- 3. HEIGHT-BASED (High Knees, Butt Kickers, Quick Feet, Standing Kicks) ---
    elif "height_val" in data:
        val = data["height_val"]
        if val > 0.02 and state["phase"] == "low":
            state["phase"] = "high"
        elif val < -0.02 and state["phase"] == "high":
            state["phase"] = "low"
            state["reps"] += 1
            state["perfect_reps"] += 1  # Streak goes up!

    # --- 4. COOL-DOWN / STATIC ---
    elif any(x in exercise_name for x in ["stretch", "gators"]):
        state["consecutive_stuck_frames"] = 0 

    # --- GLOBAL ANOMALY TRIGGER ---
    if state["consecutive_stuck_frames"] >= STUCK_LIMIT:
        state["is_anomaly"] = True
        state["anomaly_reason"] = f"Holding the bottom position too long or struggling to complete the {exercise_name}."
    else:
        state["is_anomaly"] = False

    return state
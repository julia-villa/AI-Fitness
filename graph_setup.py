from typing import Annotated, TypedDict, List, Union
import operator

class CoachState(TypedDict):
    # Current exercise type (e.g., "squat")
    exercise: str
    # Number of successful reps
    reps: int
    # Tracks the phase of the movement (e.g., "up" or "down")
    phase: str
    # Flag to trigger the VLM Auditor
    is_anomaly: bool
    # List of messages/feedback sent to the user
    messages: Annotated[List, operator.add]
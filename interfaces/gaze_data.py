# interfaces/gaze_data.py
# =============================================================
# Person 1 output format.
# Person 4 reads this during calibration + mapping.
# =============================================================

from dataclasses import dataclass, field


@dataclass
class GazeOutput:
    """
    Raw gaze output from Person 1's L2CS pipeline.

    yaw       : horizontal gaze angle in degrees
                negative = looking left, positive = looking right
    pitch     : vertical gaze angle in degrees
                negative = looking down, positive = looking up
    confidence: model confidence in [0.0, 1.0]
                derived from average max softmax probability
                across yaw and pitch heads.
    """
    yaw: float = 0.0
    pitch: float = 0.0
    confidence: float = 0.0

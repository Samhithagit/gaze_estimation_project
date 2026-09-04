# interfaces/filter_data.py
# =============================================================
# Person 3 output format.
# Person 4 uses filtered_x / filtered_y for cursor movement
# when Person 3's Kalman filter is integrated.
# =============================================================

from dataclasses import dataclass


@dataclass
class FilteredPosition:
    """
    Kalman-filtered gaze position from Person 3's pipeline.

    x         : filtered screen x coordinate (pixels)
    y         : filtered screen y coordinate (pixels)
    velocity  : gaze velocity magnitude (pixels/frame)
    innovation: filter innovation (measurement residual)
    Q         : current process noise estimate
    R         : current measurement noise estimate
    """
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    innovation: float = 0.0
    Q: float = 0.0
    R: float = 0.0

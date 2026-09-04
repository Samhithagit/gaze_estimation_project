# module4/cursor_controller.py
# =============================================================
# Smooth cursor movement via Exponential Moving Average (EMA).
#
# Integration note (Person 3):
#   When Person 3's Kalman filter is ready, set use_kalman=True
#   in config.toml. Pass FilteredPosition(x, y, ...) to
#   move_to_filtered() instead of move_to_raw().
# =============================================================

import pyautogui
import numpy as np


# Disable pyautogui fail-safe (optional — re-enable if you want
# the top-left corner to kill the program)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0   # remove default inter-call delay


class CursorController:
    """
    Moves the system cursor based on gaze-mapped screen coordinates.

    Parameters
    ----------
    screen_w, screen_h : int — logical screen dimensions (pixels)
    smoothing_alpha    : float — EMA weight in (0,1).
                         Low alpha = more smoothing but higher lag.
                         High alpha = more responsive but jittery.
    dead_zone_px       : float — minimum movement to trigger cursor update.
    """

    def __init__(
        self,
        screen_w,
        screen_h,
        smoothing_alpha=0.2,
        dead_zone_px=20.0,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.alpha = smoothing_alpha
        self.dead_zone = dead_zone_px

        # EMA state — initialise to screen centre
        self.smooth_x = screen_w / 2.0
        self.smooth_y = screen_h / 2.0

        self._first = True

    # =========================
    # Move — from raw mapped position
    # =========================

    def move_to_raw(self, raw_x, raw_y):
        """
        Update cursor from raw gaze-mapped (screen_x, screen_y).
        Applies EMA smoothing internally.

        Parameters
        ----------
        raw_x, raw_y : float — unmapped screen coordinates

        Returns
        -------
        (cursor_x, cursor_y) : float, float — actual position set
        """
        if self._first:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
            self._first = False

        # EMA
        self.smooth_x = self.alpha * raw_x + (1 - self.alpha) * self.smooth_x
        self.smooth_y = self.alpha * raw_y + (1 - self.alpha) * self.smooth_y

        return self._apply(self.smooth_x, self.smooth_y)

    # =========================
    # Move — from Person 3's Kalman filter
    # =========================

    def move_to_filtered(self, filtered_pos):
        """
        Update cursor from Person 3's FilteredPosition.
        No additional smoothing applied (KF already filters).

        Parameters
        ----------
        filtered_pos : interfaces.filter_data.FilteredPosition

        Returns
        -------
        (cursor_x, cursor_y) : float, float
        """
        return self._apply(filtered_pos.x, filtered_pos.y)

    # =========================
    # Internal
    # =========================

    def _apply(self, x, y):
        """Clamp, dead-zone check, then move cursor."""

        # Clamp to screen bounds with small margin
        cx = float(np.clip(x, 0, self.screen_w - 1))
        cy = float(np.clip(y, 0, self.screen_h - 1))

        # Dead zone: skip if movement is tiny
        curr_x, curr_y = pyautogui.position()
        dist = np.hypot(cx - curr_x, cy - curr_y)
        if dist < self.dead_zone:
            return float(curr_x), float(curr_y)

        pyautogui.moveTo(int(cx), int(cy))
        return cx, cy

    def reset(self):
        """Reset EMA state to screen centre."""
        self.smooth_x = self.screen_w / 2.0
        self.smooth_y = self.screen_h / 2.0
        self._first = True

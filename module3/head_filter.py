import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class HeadPoseFilter:
    """
    Real-time Butterworth low-pass filter for head yaw, pitch, and roll.

    The filter is causal, so it can be used frame-by-frame in a live
    webcam system.
    """

    def __init__(self, fps=30.0, cutoff_frequency=2.0, order=2):
        self.fps = float(fps)
        self.cutoff_frequency = float(cutoff_frequency)
        self.order = int(order)

        if self.fps <= 0:
            raise ValueError("fps must be greater than 0")

        if self.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be greater than 0")

        if self.cutoff_frequency >= self.fps / 2:
            raise ValueError(
                "cutoff_frequency must be less than the Nyquist frequency"
            )

        # Design Butterworth low-pass filter.
        self.sos = butter(
            self.order,
            self.cutoff_frequency,
            btype="lowpass",
            fs=self.fps,
            output="sos"
        )

        # Filter state for each angle.
        self.yaw_state = sosfilt_zi(self.sos)
        self.pitch_state = sosfilt_zi(self.sos)
        self.roll_state = sosfilt_zi(self.sos)

        self.initialized = False

    def reset(self):
        """Reset the filter state."""
        self.yaw_state = sosfilt_zi(self.sos)
        self.pitch_state = sosfilt_zi(self.sos)
        self.roll_state = sosfilt_zi(self.sos)

        self.initialized = False

    def update(self, head_pose):
        """
        Filter one head-pose measurement.

        Input:
            {
                "yaw": ...,
                "pitch": ...,
                "roll": ...,
                "reprojection_error": ...
            }

        Output:
            {
                "yaw": ...,
                "pitch": ...,
                "roll": ...,
                "reprojection_error": ...
            }
        """

        if head_pose is None:
            return None

        yaw = float(head_pose["yaw"])
        pitch = float(head_pose["pitch"])
        roll = float(head_pose["roll"])

        values = [yaw, pitch, roll]

        if not all(np.isfinite(value) for value in values):
            return None

        # Initialize the filter around the first valid measurement.
        # This avoids a large startup transient.
        if not self.initialized:
            self.yaw_state = self.yaw_state * yaw
            self.pitch_state = self.pitch_state * pitch
            self.roll_state = self.roll_state * roll

            self.initialized = True

            return {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "reprojection_error": float(
                    head_pose["reprojection_error"]
                )
            }

        # Filter one sample at a time.
        filtered_yaw, self.yaw_state = sosfilt(
            self.sos,
            [yaw],
            zi=self.yaw_state
        )

        filtered_pitch, self.pitch_state = sosfilt(
            self.sos,
            [pitch],
            zi=self.pitch_state
        )

        filtered_roll, self.roll_state = sosfilt(
            self.sos,
            [roll],
            zi=self.roll_state
        )

        return {
            "yaw": float(filtered_yaw[0]),
            "pitch": float(filtered_pitch[0]),
            "roll": float(filtered_roll[0]),
            "reprojection_error": float(
                head_pose["reprojection_error"]
            )
        }
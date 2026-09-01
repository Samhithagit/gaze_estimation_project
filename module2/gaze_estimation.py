from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.resnet import Bottleneck

from l2cs.model import L2CS


class GazeEstimator:

    def __init__(self, weights_path, device="cpu"):

        self.device = torch.device(device)

        # =========================
        # Load L2CS Model
        # =========================

        self.model = L2CS(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            num_bins=90
        )

        # =========================
        # Load Pretrained Weights
        # =========================

        state_dict = torch.load(
            weights_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        self.model.eval()

        # =========================
        # Image Transform
        # =========================

        self.transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # =========================
        # Angle Bins
        # =========================

        self.idx_tensor = torch.arange(
            90,
            dtype=torch.float32
        ).to(self.device)

    # =========================
    # Predict Gaze
    # =========================

    def predict_gaze(self, eye_image):

        if eye_image is None:
            return None, None

        # =========================
        # Image → Tensor
        # =========================

        img = self.transform(eye_image)

        img = img.unsqueeze(0)

        img = img.to(self.device)

        # =========================
        # Forward Pass
        # =========================

        with torch.no_grad():

            gaze_pitch, gaze_yaw = self.model(img)

            pitch_predicted = torch.softmax(
                gaze_pitch,
                dim=1
            )

            yaw_predicted = torch.softmax(
                gaze_yaw,
                dim=1
            )

            print("\nYaw Softmax Probabilities:")
            print(yaw_predicted)

            print("\nPitch Softmax Probabilities:")
            print(pitch_predicted)

            pitch = torch.sum(
                pitch_predicted * self.idx_tensor,
                dim=1
            ) * 4 - 180

            yaw = torch.sum(
                yaw_predicted * self.idx_tensor,
                dim=1
            ) * 4 - 180

            pitch = pitch.item()
            yaw = yaw.item()

        return yaw, pitch
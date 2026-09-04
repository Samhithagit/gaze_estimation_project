import torch
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms as transforms
import math


class GazeEstimator:

    def __init__(self, model_path, device="cpu"):

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        # =====================================================
        # RESNET50 BACKBONE
        # =====================================================

        self.backbone = resnet50(weights=None)

        # Remove ImageNet classifier
        self.backbone.fc = nn.Identity()

        # =====================================================
        # GAZE HEADS
        #
        # IMPORTANT:
        # Checkpoint contains:
        #
        # fc_yaw_gaze.weight   [90, 2048]
        # fc_yaw_gaze.bias     [90]
        #
        # fc_pitch_gaze.weight [90, 2048]
        # fc_pitch_gaze.bias   [90]
        # =====================================================

        self.fc_yaw_gaze = nn.Linear(
            2048,
            90
        )

        self.fc_pitch_gaze = nn.Linear(
            2048,
            90
        )

        # =====================================================
        # LOAD CHECKPOINT
        # =====================================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Handle checkpoints containing state_dict
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Remove DataParallel prefix
        checkpoint = {
            key.replace("module.", ""): value
            for key, value in checkpoint.items()
        }

        # =====================================================
        # LOAD BACKBONE
        # =====================================================

        backbone_state = {}

        for key, value in checkpoint.items():

            if (
                key.startswith("conv1")
                or key.startswith("bn1")
                or key.startswith("layer1")
                or key.startswith("layer2")
                or key.startswith("layer3")
                or key.startswith("layer4")
            ):
                backbone_state[key] = value

        self.backbone.load_state_dict(
            backbone_state,
            strict=False
        )

        # =====================================================
        # LOAD YAW HEAD
        # =====================================================

        self.fc_yaw_gaze.load_state_dict({
            "weight": checkpoint["fc_yaw_gaze.weight"],
            "bias": checkpoint["fc_yaw_gaze.bias"]
        })

        # =====================================================
        # LOAD PITCH HEAD
        # =====================================================

        self.fc_pitch_gaze.load_state_dict({
            "weight": checkpoint["fc_pitch_gaze.weight"],
            "bias": checkpoint["fc_pitch_gaze.bias"]
        })

        # =====================================================
        # MOVE TO DEVICE
        # =====================================================

        self.backbone.to(self.device)
        self.fc_yaw_gaze.to(self.device)
        self.fc_pitch_gaze.to(self.device)

        self.backbone.eval()
        self.fc_yaw_gaze.eval()
        self.fc_pitch_gaze.eval()

        # =====================================================
        # IMAGE PREPROCESSING
        # =====================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        # =====================================================
        # GAZE BINS
        # =====================================================

        self.num_bins = 90

        self.angle_min = -90.0
        self.angle_max = 90.0

        self.bin_width = (
            self.angle_max - self.angle_min
        ) / self.num_bins

        print("Backbone loaded")
        print("Yaw/Pitch heads loaded")

    # =========================================================
    # CONVERT CLASSIFICATION BINS TO ANGLE
    # =========================================================

    def bins_to_angle(self, logits):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # -----------------------------------------------------
        # Bin centers
        # -----------------------------------------------------

        bin_centers = torch.linspace(
            self.angle_min + self.bin_width / 2,
            self.angle_max - self.bin_width / 2,
            self.num_bins,
            device=logits.device
        )

        # -----------------------------------------------------
        # Expected angle
        # -----------------------------------------------------

        angle = torch.sum(
            probabilities * bin_centers,
            dim=1
        )

        # -----------------------------------------------------
        # ENTROPY-BASED CONFIDENCE
        #
        # A concentrated probability distribution
        # = high confidence
        #
        # A spread-out distribution
        # = low confidence
        # -----------------------------------------------------

        entropy = -torch.sum(
            probabilities *
            torch.log(
                probabilities + 1e-8
            ),
            dim=1
        )

        max_entropy = math.log(
            self.num_bins
        )

        confidence = 1.0 - (
            entropy / max_entropy
        )

        # Keep confidence in [0, 1]
        confidence = torch.clamp(
            confidence,
            0.0,
            1.0
        )

        return (
            angle.item(),
            confidence.item()
        )

    # =========================================================
    # PREDICT GAZE
    #
    # This function handles ONE eye.
    #
    # Fusion of LEFT + RIGHT should happen outside.
    # =========================================================

    def predict_gaze(self, eye_image):

        if eye_image is None:
            return None, None, 0.0

        try:

            # -------------------------------------------------
            # OpenCV image -> PIL
            # -------------------------------------------------

            image = Image.fromarray(
                eye_image
            ).convert("RGB")

            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------

            tensor = self.transform(
                image
            )

            tensor = tensor.unsqueeze(0)

            tensor = tensor.to(
                self.device
            )

            # -------------------------------------------------
            # MODEL
            # -------------------------------------------------

            with torch.no_grad():

                features = self.backbone(
                    tensor
                )

                yaw_logits = self.fc_yaw_gaze(
                    features
                )

                pitch_logits = self.fc_pitch_gaze(
                    features
                )

            # -------------------------------------------------
            # CONVERT TO ANGLES
            # -------------------------------------------------

            yaw, yaw_conf = self.bins_to_angle(
                yaw_logits
            )

            pitch, pitch_conf = self.bins_to_angle(
                pitch_logits
            )

            # -------------------------------------------------
            # COMBINE YAW + PITCH CONFIDENCE
            # -------------------------------------------------

            confidence = (
                yaw_conf + pitch_conf
            ) / 2.0

            return (
                yaw,
                pitch,
                confidence
            )

        except Exception as e:

            print(
                "Gaze prediction error:",
                e
            )

            return (
                None,
                None,
                0.0
            )
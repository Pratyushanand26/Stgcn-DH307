"""
Wraps the pretrained ST-GCN (tsstg-model.pth) + MediaPipe pose extraction.

Responsibility:
    1. Frame -> exact 13-joint skeleton used by the pretrained TSSTG
    2. 30-frame skeleton window -> 7-class probability vector

Strategy logic (FPS simulation, frame reconstruction, computation
strategies) lives outside this class.

IMPORTANT:
    The MediaPipe -> 13-joint mapping and preprocessing below are
    copied from the already-tested STGCNDetector implementation.
"""

import os
import glob

import cv2
import numpy as np
import torch
import mediapipe as mp

from stgn_model import (
    TwoStreamSpatialTemporalGraph,
    normalize_points_with_size,
    scale_pose,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

WEIGHTS_PATH = (
    "./Human-Falling-Detect-Tracks/"
    "Models/TSSTG/tsstg-model.pth"
)

ACTION_CLASSES = [
    "Standing",
    "Walking",
    "Sitting",
    "Lying Down",
    "Stand up",
    "Sit down",
    "Fall Down",
]

WINDOW_SIZE = 30

# Set True if the input video is actually a depth|RGB split-screen.
CROP_RGB_HALF = False


class STGCNDetector:

    # -----------------------------------------------------------------
    # Exact MediaPipe -> TSSTG 13-joint mapping
    #
    # Order:
    #   0  Nose
    #   1  Left Shoulder
    #   2  Right Shoulder
    #   3  Left Elbow
    #   4  Right Elbow
    #   5  Left Wrist
    #   6  Right Wrist
    #   7  Left Hip
    #   8  Right Hip
    #   9  Left Knee
    #   10 Right Knee
    #   11 Left Ankle
    #   12 Right Ankle
    # -----------------------------------------------------------------

    MP_TO_COCO13 = [
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        23,
        24,
        25,
        26,
        27,
        28,
    ]

    ACTION_CLASSES = ACTION_CLASSES

    def __init__(
        self,
        weights_path=WEIGHTS_PATH,
        window_size=WINDOW_SIZE,
    ):

        if window_size != WINDOW_SIZE:
            print(
                f"[WARN] window_size={window_size} != 30. "
                f"Checkpoint was trained on 30-frame windows."
            )

        self.window_size = window_size

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[STGCN] device: {self.device}")

        # -------------------------------------------------------------
        # MediaPipe Pose
        # -------------------------------------------------------------

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # -------------------------------------------------------------
        # Load exact ST-GCN architecture
        # -------------------------------------------------------------

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Could not find ST-GCN weights:\n"
                f"{weights_path}"
            )

        print(
            f"[STGCN] loading weights from: {weights_path}"
        )

        self.model = TwoStreamSpatialTemporalGraph(
            graph_args={
                "strategy": "spatial"
            },
            num_class=len(ACTION_CLASSES),
        ).to(self.device)

        # -------------------------------------------------------------
        # Load checkpoint
        # -------------------------------------------------------------

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
        )

        # Some checkpoints contain:
        #
        # {
        #     "model": state_dict,
        #     ...
        # }
        #
        # Others are directly the state_dict.

        state_dict = checkpoint.get(
            "model",
            checkpoint
        )

        result = self.model.load_state_dict(
            state_dict,
            strict=False,
        )

        total_params = len(
            list(self.model.state_dict().keys())
        )

        missing = len(result.missing_keys)
        unexpected = len(result.unexpected_keys)

        matched = total_params - missing

        print(
            f"[STGCN] weight loading: "
            f"{matched}/{total_params} model params matched"
        )

        if missing:
            print(
                f"[STGCN] WARNING: {missing} missing keys"
            )

        if unexpected:
            print(
                f"[STGCN] WARNING: "
                f"{unexpected} unexpected checkpoint keys"
            )

        # -------------------------------------------------------------
        # Evaluation mode
        # -------------------------------------------------------------

        self.model.eval()

        # -------------------------------------------------------------
        # Temporal skeleton buffer
        #
        # Each element:
        #     (13, 3)
        #
        # where each joint is:
        #     [x_pixel, y_pixel, visibility]
        # -------------------------------------------------------------

        self.window = []

    # =================================================================
    # OPTIONAL RGB CROP
    # =================================================================

    def _prepare_frame(self, frame):
        """
        Optionally crop a depth|RGB split-screen frame.

        Assumes RGB is the RIGHT half.

        If your videos are already RGB-only, leave
        CROP_RGB_HALF = False.
        """

        if not CROP_RGB_HALF:
            return frame

        h, w = frame.shape[:2]

        # Keep right half = RGB
        frame = frame[:, w // 2:]

        return frame

    # =================================================================
    # FRAME -> 13 JOINTS
    # =================================================================

    def extract_keypoints(self, frame):
        """
        Input:
            frame: OpenCV BGR image

        Output:
            numpy array of shape:

                (13, 3)

            Each joint:

                [x_pixel, y_pixel, visibility]

        Returns:
            None if MediaPipe does not detect a person.

        This is the exact MediaPipe -> 13-joint extraction used by
        the previously working ST-GCN detector.
        """

        frame = self._prepare_frame(frame)

        h, w = frame.shape[:2]

        # BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        raw = np.zeros(
            (13, 3),
            dtype=np.float32,
        )

        for i, idx in enumerate(
            self.MP_TO_COCO13
        ):

            lm = landmarks[idx]

            raw[i] = [
                lm.x * w,
                lm.y * h,
                lm.visibility,
            ]

        return raw

    # =================================================================
    # NORMALIZATION + 14TH NECK JOINT
    # =================================================================

    def normalize_sequence(
        self,
        keypoints_seq,
        frame_w,
        frame_h,
    ):
        """
        Input:

            keypoints_seq:
                (T, 13, 3)

            where each joint is:

                [x_pixel, y_pixel, visibility]

        Output:

            normalized skeleton:

                (T, 14, 3)

        Processing exactly follows the existing working ST-GCN
        implementation:

            1. normalize x/y by image dimensions
            2. per-frame min/max pose scaling
            3. append neck joint
        """

        # -------------------------------------------------------------
        # (T, 13, 3)
        # -------------------------------------------------------------

        pts = np.stack(
            keypoints_seq,
            axis=0,
        ).astype(np.float32)

        # -------------------------------------------------------------
        # 1. Normalize pixel coordinates by image dimensions
        #
        # x -> x / width
        # y -> y / height
        #
        # Visibility is NOT modified.
        # -------------------------------------------------------------

        pts[:, :, :2] = normalize_points_with_size(
            pts[:, :, :2],
            frame_w,
            frame_h,
        )

        # -------------------------------------------------------------
        # 2. Per-frame min/max scaling
        #
        # x/y are mapped approximately to [-1, 1].
        # -------------------------------------------------------------

        pts[:, :, :2] = scale_pose(
            pts[:, :, :2]
        )

        # -------------------------------------------------------------
        # 3. Add neck
        #
        # Neck = midpoint of:
        #
        #   joint 1 = Left Shoulder
        #   joint 2 = Right Shoulder
        #
        # 13 joints -> 14 joints
        # -------------------------------------------------------------

        neck = np.expand_dims(
            (
                pts[:, 1, :]
                +
                pts[:, 2, :]
            ) / 2,
            axis=1,
        )

        pts = np.concatenate(
            (
                pts,
                neck,
            ),
            axis=1,
        )

        return pts

    # =================================================================
    # CLASSIFICATION
    # =================================================================

    def classify(
        self,
        keypoints_seq,
        frame_w,
        frame_h,
    ):
        """
        Classify a complete skeleton sequence.

        Expected input:

            keypoints_seq.shape == (30, 13, 3)

        Returns:

            {
                "pred_class": ...,
                "fall_flagged": ...,
                "prob_Standing": ...,
                ...
                "prob_Fall_Down": ...
            }
        """

        if len(keypoints_seq) != self.window_size:
            raise ValueError(
                f"ST-GCN requires {self.window_size} frames, "
                f"got {len(keypoints_seq)}"
            )

        # -------------------------------------------------------------
        # Normalize + add neck
        #
        # Result:
        #
        #   (30, 14, 3)
        # -------------------------------------------------------------

        pts = self.normalize_sequence(
            keypoints_seq,
            frame_w,
            frame_h,
        )

        # -------------------------------------------------------------
        # POINT STREAM
        #
        # Current representation:
        #
        #   (T, V, C)
        #
        #   T = 30
        #   V = 14 joints
        #   C = 3 [x, y, visibility]
        #
        # Convert to:
        #
        #   (N, C, T, V)
        #
        #   (1, 3, 30, 14)
        # -------------------------------------------------------------

        pts_t = torch.tensor(
            pts,
            dtype=torch.float32,
            device=self.device,
        )

        pts_t = (
            pts_t
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        # -------------------------------------------------------------
        # MOTION STREAM
        #
        # Frame-to-frame x/y displacement:
        #
        #   motion[t] = position[t] - position[t-1]
        #
        # Therefore:
        #
        #   (1, 2, T-1, 14)
        #
        # = (1, 2, 29, 14)
        #
        # This exactly matches the working implementation.
        # -------------------------------------------------------------

        motion_t = (
            pts_t[:, :2, 1:, :]
            -
            pts_t[:, :2, :-1, :]
        )

        # -------------------------------------------------------------
        # ST-GCN inference
        # -------------------------------------------------------------

        with torch.no_grad():

            output = self.model(
                (
                    pts_t,
                    motion_t,
                )
            )

        # The TwoStreamSpatialTemporalGraph already applies
        # sigmoid() in its forward() method.
        probs = (
            output
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------------------
        # Top class
        # -------------------------------------------------------------

        top_idx = int(
            np.argmax(probs)
        )

        top_class = ACTION_CLASSES[
            top_idx
        ]

        # -------------------------------------------------------------
        # Return standardized result
        # -------------------------------------------------------------

        result = {
            "pred_class": top_class,
            "fall_flagged": (
                top_class == "Fall Down"
            ),
        }

        for i, cls in enumerate(
            ACTION_CLASSES
        ):

            result[
                f"prob_{cls.replace(' ', '_')}"
            ] = float(probs[i])

        return result

    # =================================================================
    # UTILITY: CLEAR BUFFER
    # =================================================================

    def reset(self):
        """
        Clear the temporal window.

        Useful when starting a new video/person/experiment.
        """

        self.window.clear()

    # =================================================================
    # CLEANUP
    # =================================================================

    def close(self):
        """
        Release MediaPipe resources.
        """

        if hasattr(self.pose, "close"):
            self.pose.close()
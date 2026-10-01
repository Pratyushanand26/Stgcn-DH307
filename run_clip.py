"""
Run one video through one strategy.

Usage:
    python run_clip.py <video_path> <clip_id> <mode> [k]

Modes:
    full30
        Native FPS, 30-frame sliding window, inference every frame.

    fps_hold
        Keep every 6th frame from the 30-frame native window
        and reconstruct the 30-frame sequence using zero-order hold.

    fps_interp
        Keep every 6th frame from the 30-frame native window
        and reconstruct the 30-frame sequence using interpolation.

    cost_skip
        Run MediaPipe every k frames and reuse the previous pose
        on skipped frames.

    cost_stride
        Run MediaPipe every frame, but run ST-GCN inference
        only every k frames.

Examples:
    python run_clip.py ../videos/adl-01-cam0.mp4 adl-01-cam0 full30

    python run_clip.py ../videos/adl-01-cam0.mp4 adl-01-cam0 fps_hold

    python run_clip.py ../videos/adl-01-cam0.mp4 adl-01-cam0 fps_interp

    python run_clip.py ../videos/adl-01-cam0.mp4 adl-01-cam0 cost_skip 3

    python run_clip.py ../videos/adl-01-cam0.mp4 adl-01-cam0 cost_stride 3
"""

import sys
import os
import csv
import time
import collections

from collections import Counter

import numpy as np
import psutil

from video_source import iter_frames
from stgcn_detector import (
    STGCNDetector,
    ACTION_CLASSES,
    WINDOW_SIZE,
)

from strategies import (
    FPS_STRATEGIES,
    cost_skip_should_extract_pose,
    cost_stride_should_run_inference,
)


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = "results"

MASTER_RESULTS_PATH = os.path.join(RESULTS_DIR, "consolidated_results.csv")


VALID_MODES = [
    "full30",
    "fps_hold",
    "fps_interp",
    "cost_skip",
    "cost_stride",
]


DEFAULT_K = 3


# ============================================================
# OPTIONAL GROUND TRUTH
# ============================================================
#
# Leave empty if you don't have labels yet.
#
# Example:
#
# GROUND_TRUTH = {
#     "adl-01-cam0": False,
#     "adl-02-cam0": False,
#     "fall-01-cam0": True,
# }
#
# False = no fall
# True  = fall
#
# If a clip is not present here, ground truth remains blank.
# ============================================================

GROUND_TRUTH = {
    # "adl-01-cam0": False,
    # "adl-02-cam0": False,
    # "fall-01-cam0": True,
}


# ============================================================
# PER-FRAME CSV
# ============================================================

def csv_path_for(mode):
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    return os.path.join(
        RESULTS_DIR,
        f"results_{mode}.csv",
    )


def make_row_writer(mode):
    """
    Opens the detailed per-frame CSV.

    One row = one video frame.
    """

    path = csv_path_for(mode)

    is_new = not os.path.exists(path)

    f = open(
        path,
        "a",
        newline="",
    )

    fieldnames = [
        "clip_id",
        "frame_index",
        "timestamp_sec",
        "mode",
        "k_param",
        "compute_time_ms",
        "cpu_percent",
        "ram_mb",
        "pred_class",
        "fall_flagged",
    ]

    # Keep the human-readable CSV column names.
    fieldnames += [
        f"prob_{c}"
        for c in ACTION_CLASSES
    ]

    fieldnames += [
        "first_fall_frame",
        "first_fall_time_sec",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    if is_new:
        writer.writeheader()

    return f, writer


# ============================================================
# RAM
# ============================================================

def get_process_ram_mb(process):
    """
    Return current process RSS in MB.
    """

    try:
        return process.memory_info().rss / (
            1024 * 1024
        )
    except Exception:
        return 0.0


# ============================================================
# ACTION / PROBABILITY HELPERS
# ============================================================

def detector_probability_key(action_name):
    """
    Convert:

        'Lying Down'

    into:

        'prob_Lying_Down'

    because STGCNDetector uses underscores internally.
    """

    return (
        "prob_"
        + action_name.replace(" ", "_")
    )


def get_probability(result, action_name):
    """
    Safely retrieve one action probability.
    """

    if result is None:
        return None

    key = detector_probability_key(
        action_name
    )

    value = result.get(
        key,
        None,
    )

    if value is None:
        return None

    return float(value)


# ============================================================
# MASTER CSV
# ============================================================

def master_fieldnames():
    """
    Columns for consolidated_results.csv.

    One row = one clip × one strategy.
    """

    fields = [
        "clip_id",
        "strategy",
        "k_param",

        "total_frames",
        "duration_sec",

        "ground_truth_fall",
        "fall_detected",
        "correct",

        "fall_frame",
        "fall_time_sec",

        "action_at_fall",
        "dominant_action",

        # ----------------------------------------
        # Compute
        # ----------------------------------------

        "total_compute_time_sec",
        "avg_compute_time_ms",
        "median_compute_time_ms",
        "peak_compute_time_ms",

        "effective_fps",

        # ----------------------------------------
        # Actual ST-GCN inference
        # ----------------------------------------

        "inference_frames",
        "inference_rate_fps",

        "avg_inference_compute_ms",
        "median_inference_compute_ms",
        "peak_inference_compute_ms",

        # ----------------------------------------
        # CPU
        # ----------------------------------------

        "avg_cpu_pct",
        "peak_cpu_pct",

        # ----------------------------------------
        # RAM
        # ----------------------------------------

        "avg_ram_mb",
        "peak_ram_mb",

        # ----------------------------------------
        # Fall statistics
        # ----------------------------------------

        "fall_inference_frames",
        "max_prob_Fall_Down",
        "avg_prob_Fall_Down",

        # ----------------------------------------
        # Probability at fall
        # ----------------------------------------

        "fall_prob_Standing",
        "fall_prob_Walking",
        "fall_prob_Sitting",
        "fall_prob_Lying_Down",
        "fall_prob_Stand_up",
        "fall_prob_Sit_down",
        "fall_prob_Fall_Down",
    ]

    # ----------------------------------------
    # Average probabilities over all
    # actual inference frames
    # ----------------------------------------

    for c in ACTION_CLASSES:
        fields.append(
            f"avg_prob_{c.replace(' ', '_')}"
        )

    # ----------------------------------------
    # Maximum probability over all inference
    # frames
    # ----------------------------------------

    for c in ACTION_CLASSES:
        fields.append(
            f"max_prob_{c.replace(' ', '_')}"
        )

    return fields


def write_master_result(row):
    """
    Insert/update one clip × strategy row.

    If the same clip and strategy already exist,
    replace the old row rather than duplicating it.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    fieldnames = master_fieldnames()

    existing_rows = []

    if os.path.exists(
        MASTER_RESULTS_PATH
    ):

        with open(
            MASTER_RESULTS_PATH,
            "r",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for old_row in reader:

                same_clip = (
                    old_row.get("clip_id")
                    == row["clip_id"]
                )

                same_strategy = (
                    old_row.get("strategy")
                    == row["strategy"]
                )

                if not (
                    same_clip
                    and same_strategy
                ):
                    existing_rows.append(
                        old_row
                    )

    existing_rows.append(row)

    # Sort by clip then strategy.
    existing_rows.sort(
        key=lambda x: (
            x.get("clip_id", ""),
            x.get("strategy", ""),
        )
    )

    with open(
        MASTER_RESULTS_PATH,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            existing_rows
        )


# ============================================================
# RUN ONE CLIP
# ============================================================

def run_clip(
    path,
    clip_id,
    mode,
    k=DEFAULT_K,
):
    """
    Process one video using one strategy.
    """

    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            f"Expected one of {VALID_MODES}"
        )

    if mode in (
        "cost_skip",
        "cost_stride",
    ):
        if k < 1:
            raise ValueError(
                "k must be >= 1"
            )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    detector = STGCNDetector(
        window_size=WINDOW_SIZE
    )

    # --------------------------------------------------------
    # Process statistics
    # --------------------------------------------------------

    process = psutil.Process(
        os.getpid()
    )

    psutil.cpu_percent(
        interval=None
    )

    # --------------------------------------------------------
    # Temporal buffer
    # --------------------------------------------------------

    keypoint_buffer = collections.deque(
        maxlen=WINDOW_SIZE
    )

    last_keypoints = None

    # Latest classification.
    #
    # Used by cost_stride between actual
    # inference calls.
    last_classify_result = None

    # --------------------------------------------------------
    # Video statistics
    # --------------------------------------------------------

    total_frames = 0

    last_timestamp = 0.0

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    compute_times = []

    inference_compute_times = []

    wall_start = time.perf_counter()

    # --------------------------------------------------------
    # CPU / RAM
    # --------------------------------------------------------

    cpu_values = []

    ram_values = []

    # --------------------------------------------------------
    # Inference statistics
    # --------------------------------------------------------

    inference_frames = 0

    predicted_classes = []

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    probability_history = {
        c: []
        for c in ACTION_CLASSES
    }

    # --------------------------------------------------------
    # Fall information
    # --------------------------------------------------------

    first_fall_frame = None
    first_fall_time = None

    fall_inference_frames = 0

    # Probability vector at first detected fall.
    fall_probability_snapshot = {
        c: None
        for c in ACTION_CLASSES
    }

    # --------------------------------------------------------
    # Per-frame CSV
    # --------------------------------------------------------

    f, writer = make_row_writer(
        mode
    )

    try:

        # ====================================================
        # FRAME LOOP
        # ====================================================

        for frame_idx, ts, frame in iter_frames(path):

            total_frames += 1

            last_timestamp = ts
            frame_start = time.perf_counter()

            # ------------------------------------------------
            # POSE EXTRACTION
            # ------------------------------------------------

            if (
                mode == "cost_skip"
                and not cost_skip_should_extract_pose(
                    frame_idx,
                    k,
                )
            ):

                # Do not run MediaPipe.
                #
                # Reuse the last valid pose.
                keypoints = last_keypoints

            else:

                keypoints = (
                    detector.extract_keypoints(
                        frame
                    )
                )

                if keypoints is not None:

                    last_keypoints = (
                        keypoints
                    )

            # ------------------------------------------------
            # Add valid pose to buffer
            # ------------------------------------------------

            if keypoints is not None:

                keypoint_buffer.append(
                    keypoints
                )

            # ------------------------------------------------
            # Start with previous result
            # ------------------------------------------------

            result = (
                last_classify_result
            )

            ran_inference = False

            # ------------------------------------------------
            # ST-GCN requires 30 poses
            # ------------------------------------------------

            if (
                len(keypoint_buffer)
                == WINDOW_SIZE
            ):

                window = np.stack(keypoint_buffer, axis=0)
                inference_start = time.perf_counter()

                # --------------------------------------------
                # FPS reconstruction
                # --------------------------------------------

                if mode in FPS_STRATEGIES:

                    window = (
                        FPS_STRATEGIES[
                            mode
                        ](window)
                    )

                # --------------------------------------------
                # COST STRIDE
                # --------------------------------------------

                if mode == "cost_stride":

                    if (
                        cost_stride_should_run_inference(
                            frame_idx,
                            k,
                        )
                    ):

                        result = detector.classify(window, frame.shape[1], frame.shape[0])
                        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
                        ran_inference = True

                # --------------------------------------------
                # FULL30 / FPS modes
                # --------------------------------------------

                else:

                    result = detector.classify(window, frame.shape[1], frame.shape[0])
                    inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
                    ran_inference = True

            # ------------------------------------------------
            # Prediction/window compute time only
            # ------------------------------------------------

            # compute_time_ms = inference_time_ms if ran_inference else 0.0
            # if ran_inference: compute_times.append(compute_time_ms)
            compute_time_ms = (time.perf_counter() - frame_start) * 1000.0
            compute_times.append(compute_time_ms)

            # ------------------------------------------------
            # CPU
            # ------------------------------------------------

            cpu_pct = psutil.cpu_percent(
                interval=None
            )

            cpu_values.append(
                cpu_pct
            )

            # ------------------------------------------------
            # RAM
            # ------------------------------------------------

            ram_mb = get_process_ram_mb(
                process
            )

            ram_values.append(
                ram_mb
            )

            # =================================================
            # ACTUAL INFERENCE
            # =================================================

            if ran_inference:

                inference_frames += 1

                inference_compute_times.append(compute_time_ms)

                # --------------------------------------------
                # Save latest result
                # --------------------------------------------

                last_classify_result = (
                    result
                )

                # --------------------------------------------
                # Predicted class
                # --------------------------------------------

                predicted_classes.append(
                    result["pred_class"]
                )

                # --------------------------------------------
                # Probabilities
                # --------------------------------------------

                for c in ACTION_CLASSES:

                    probability = (
                        get_probability(
                            result,
                            c,
                        )
                    )

                    if probability is not None:

                        probability_history[
                            c
                        ].append(
                            probability
                        )

                # --------------------------------------------
                # Fall
                # --------------------------------------------

                if result["fall_flagged"]:

                    fall_inference_frames += 1

                    if first_fall_frame is None:

                        first_fall_frame = (
                            frame_idx
                        )

                        first_fall_time = (
                            ts
                        )

                        # Save the entire probability
                        # vector at first fall.
                        for c in ACTION_CLASSES:

                            fall_probability_snapshot[
                                c
                            ] = get_probability(
                                result,
                                c,
                            )

            # ------------------------------------------------
            # If result exists but no new inference occurred,
            # it remains the previous result.
            # ------------------------------------------------

            if result is not None:

                last_classify_result = (
                    result
                )

            # =================================================
            # PER-FRAME CSV ROW
            # =================================================

            row = {
                "clip_id": clip_id,

                "frame_index": frame_idx,

                "timestamp_sec": round(
                    ts,
                    4,
                ),

                "mode": mode,

                "k_param": (
                    k
                    if mode in (
                        "cost_skip",
                        "cost_stride",
                    )
                    else ""
                ),

                "compute_time_ms": round(
                    compute_time_ms,
                    3,
                ),

                "cpu_percent": round(
                    cpu_pct,
                    2,
                ),

                "ram_mb": round(
                    ram_mb,
                    2,
                ),

                "pred_class": (
                    result["pred_class"]
                    if result
                    else ""
                ),

                "fall_flagged": (
                    result["fall_flagged"]
                    if result
                    else ""
                ),

                "first_fall_frame": (
                    first_fall_frame
                    if first_fall_frame is not None
                    else ""
                ),

                "first_fall_time_sec": (
                    round(
                        first_fall_time,
                        4,
                    )
                    if first_fall_time is not None
                    else ""
                ),
            }

            # ------------------------------------------------
            # All seven probabilities
            # ------------------------------------------------

            for c in ACTION_CLASSES:

                key = detector_probability_key(
                    c
                )

                row[
                    f"prob_{c}"
                ] = (
                    result.get(
                        key,
                        "",
                    )
                    if result
                    else ""
                )

            writer.writerow(
                row
            )

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            if ran_inference:

                fall_text = ""

                if result["fall_flagged"]:
                    fall_text = (
                        "  *** FALL ***"
                    )

                print(
                    f"[{clip_id}] "
                    f"frame {frame_idx} "
                    f"t={ts:.2f}s "
                    f"-> "
                    f"{result['pred_class']}"
                    f"{fall_text}"
                )

    finally:

        f.close()

        try:
            detector.close()
        except Exception:
            pass

    # ========================================================
    # POST-PROCESSING
    # ========================================================

    wall_time_sec = (
        time.perf_counter()
        - wall_start
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration_sec = last_timestamp

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = (
        GROUND_TRUTH.get(
            clip_id,
            None,
        )
    )

    fall_detected = (
        first_fall_frame is not None
    )

    if ground_truth is None:

        correct = ""

    else:

        correct = (
            fall_detected
            == ground_truth
        )

    # --------------------------------------------------------
    # Action at fall
    # --------------------------------------------------------

    action_at_fall = ""

    if first_fall_frame is not None:

        # We need the classification that triggered
        # the fall. That is necessarily "Fall Down"
        # under the current detector's fall_flagged logic.
        #
        # Keep the explicit action name here.
        action_at_fall = "Fall Down"

    # --------------------------------------------------------
    # Dominant action
    # --------------------------------------------------------

    if predicted_classes:

        dominant_action = (
            Counter(
                predicted_classes
            )
            .most_common(1)[0][0]
        )

    else:

        dominant_action = ""

    # --------------------------------------------------------
    # Compute statistics
    # --------------------------------------------------------

    if compute_times:

        avg_compute_ms = float(
            np.mean(
                compute_times
            )
        )

        median_compute_ms = float(
            np.median(
                compute_times
            )
        )

        peak_compute_ms = float(
            np.max(
                compute_times
            )
        )

    else:

        avg_compute_ms = 0.0
        median_compute_ms = 0.0
        peak_compute_ms = 0.0

    total_compute_time_sec = (
        sum(compute_times)
        / 1000.0
    )

    # --------------------------------------------------------
    # Effective FPS
    # --------------------------------------------------------

    if wall_time_sec > 0:

        effective_fps = (
            total_frames
            / wall_time_sec
        )

    else:

        effective_fps = 0.0

    # --------------------------------------------------------
    # Inference FPS
    # --------------------------------------------------------

    if duration_sec > 0:

        inference_rate_fps = (
            inference_frames
            / duration_sec
        )

    else:

        inference_rate_fps = 0.0

    # --------------------------------------------------------
    # Inference computation
    # --------------------------------------------------------

    if inference_compute_times:

        avg_inference_compute_ms = float(
            np.mean(
                inference_compute_times
            )
        )

        median_inference_compute_ms = float(
            np.median(
                inference_compute_times
            )
        )

        peak_inference_compute_ms = float(
            np.max(
                inference_compute_times
            )
        )

    else:

        avg_inference_compute_ms = 0.0
        median_inference_compute_ms = 0.0
        peak_inference_compute_ms = 0.0

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    if cpu_values:

        avg_cpu_pct = float(
            np.mean(
                cpu_values
            )
        )

        peak_cpu_pct = float(
            np.max(
                cpu_values
            )
        )

    else:

        avg_cpu_pct = 0.0
        peak_cpu_pct = 0.0

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    if ram_values:

        avg_ram_mb = float(
            np.mean(
                ram_values
            )
        )

        peak_ram_mb = float(
            np.max(
                ram_values
            )
        )

    else:

        avg_ram_mb = 0.0
        peak_ram_mb = 0.0

    # ========================================================
    # MASTER RESULT ROW
    # ========================================================

    master_row = {
        "clip_id": clip_id,

        "strategy": mode,

        "k_param": (
            k
            if mode in (
                "cost_skip",
                "cost_stride",
            )
            else ""
        ),

        "total_frames": total_frames,

        "duration_sec": round(
            duration_sec,
            4,
        ),

        "ground_truth_fall": (
            ground_truth
            if ground_truth is not None
            else ""
        ),

        "fall_detected": (
            fall_detected
        ),

        "correct": correct,

        "fall_frame": (
            first_fall_frame
            if first_fall_frame is not None
            else ""
        ),

        "fall_time_sec": (
            round(
                first_fall_time,
                4,
            )
            if first_fall_time is not None
            else ""
        ),

        "action_at_fall": (
            action_at_fall
        ),

        "dominant_action": (
            dominant_action
        ),

        # Compute
        "total_compute_time_sec": round(
            total_compute_time_sec,
            4,
        ),

        "avg_compute_time_ms": round(
            avg_compute_ms,
            3,
        ),

        "median_compute_time_ms": round(
            median_compute_ms,
            3,
        ),

        "peak_compute_time_ms": round(
            peak_compute_ms,
            3,
        ),

        "effective_fps": round(
            effective_fps,
            3,
        ),

        # Inference
        "inference_frames": (
            inference_frames
        ),

        "inference_rate_fps": round(
            inference_rate_fps,
            3,
        ),

        "avg_inference_compute_ms": round(
            avg_inference_compute_ms,
            3,
        ),

        "median_inference_compute_ms": round(
            median_inference_compute_ms,
            3,
        ),

        "peak_inference_compute_ms": round(
            peak_inference_compute_ms,
            3,
        ),

        # CPU
        "avg_cpu_pct": round(
            avg_cpu_pct,
            2,
        ),

        "peak_cpu_pct": round(
            peak_cpu_pct,
            2,
        ),

        # RAM
        "avg_ram_mb": round(
            avg_ram_mb,
            2,
        ),

        "peak_ram_mb": round(
            peak_ram_mb,
            2,
        ),

        # Fall
        "fall_inference_frames": (
            fall_inference_frames
        ),

        "max_prob_Fall_Down": round(
            max(
                probability_history[
                    "Fall Down"
                ],
                default=0.0,
            ),
            5,
        ),

        "avg_prob_Fall_Down": round(
            float(
                np.mean(
                    probability_history[
                        "Fall Down"
                    ]
                )
            )
            if probability_history[
                "Fall Down"
            ]
            else 0.0,
            5,
        ),
    }

    # ========================================================
    # Probability statistics
    # ========================================================

    for c in ACTION_CLASSES:

        key_name = (
            c.replace(
                " ",
                "_",
            )
        )

        values = (
            probability_history[c]
        )

        # Average probability
        master_row[
            f"avg_prob_{key_name}"
        ] = round(
            float(
                np.mean(values)
            )
            if values
            else 0.0,
            5,
        )

        # Maximum probability
        master_row[
            f"max_prob_{key_name}"
        ] = round(
            float(
                np.max(values)
            )
            if values
            else 0.0,
            5,
        )

        # Probability vector at first fall
        fall_value = (
            fall_probability_snapshot[c]
        )

        master_row[
            f"fall_prob_{key_name}"
        ] = (
            round(
                float(fall_value),
                5,
            )
            if fall_value is not None
            else ""
        )

    # ========================================================
    # Write master CSV
    # ========================================================

    write_master_result(
        master_row
    )

    # ========================================================
    # Final output
    # ========================================================

    print()

    print(
        f"[{clip_id}] done."
    )

    print(
        f"  strategy: "
        f"{mode}"
    )

    print(
        f"  frames: "
        f"{total_frames}"
    )

    print(
        f"  inference frames: "
        f"{inference_frames}"
    )

    print(
        f"  first fall frame: "
        f"{first_fall_frame}"
    )

    print(
        f"  effective FPS: "
        f"{effective_fps:.2f}"
    )

    print(
        f"  avg compute: "
        f"{avg_compute_ms:.2f} ms/frame"
    )

    print(
        f"  wrote: "
        f"{csv_path_for(mode)}"
    )

    print(
        f"  master summary: "
        f"{MASTER_RESULTS_PATH}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 4:

        print(__doc__)

        sys.exit(1)

    video_path = sys.argv[1]

    clip_id = sys.argv[2]

    mode = sys.argv[3]

    k = int(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_K

    if mode not in VALID_MODES:

        print(
            f"mode must be one of "
            f"{VALID_MODES}"
        )

        sys.exit(1)

    run_clip(
        video_path,
        clip_id,
        mode,
        k,
    )
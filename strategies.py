"""
Four strategies, two different questions:

FPS STRATEGIES -- "what if the camera only gave us 5fps, how do we rebuild
a 30-frame window from that?" Both take a full 30-frame native buffer,
keep only every 6th sample (5 real samples spanning the same 1 real
second), and reconstruct back to 30 frames two different ways.

COST STRATEGIES -- "how do we cut compute per window without changing what
the camera gives us?" These don't touch the keypoint sequence, they change
how often the expensive steps (pose extraction, model inference) run.
"""
import numpy as np

FPS_KEEP_EVERY = 6  # 30fps -> 5fps


# ---------- FPS reconstruction strategies ----------

def fps_hold(keypoints_30):
    """Strategy A: keep every 6th frame, repeat each one 6x (zero-order hold).
    Cheapest, produces a 'staircase' motion with no invented in-between poses."""
    keep_idx = list(range(0, 30, FPS_KEEP_EVERY))  # [0, 6, 12, 18, 24]
    kept = keypoints_30[keep_idx]  # (5, 13, 3)
    out = np.repeat(kept, FPS_KEEP_EVERY, axis=0)[:30]
    return out


def fps_interp(keypoints_30):
    """Strategy B: keep every 6th frame, linearly interpolate the gaps.
    Smoother reconstruction, but invents motion the camera never captured --
    likely to blur a fast fall rather than lose it entirely."""
    keep_idx = np.array(range(0, 30, FPS_KEEP_EVERY))
    kept = keypoints_30[keep_idx]  # (5, 13, 3)
    all_idx = np.arange(30)
    out = np.empty_like(keypoints_30)
    for j in range(keypoints_30.shape[1]):      # per joint
        for c in range(keypoints_30.shape[2]):  # per channel (x, y, conf)
            out[:, j, c] = np.interp(all_idx, keep_idx, kept[:, j, c])
    return out


FPS_STRATEGIES = {
    "fps_hold": fps_hold,
    "fps_interp": fps_interp,
}


# ---------- Cost-reduction strategies ----------
# These are read by run_clip.py's loop, not applied to a finished buffer.

def cost_skip_should_extract_pose(frame_idx, k):
    """Strategy A (pose-skip): only run MediaPipe every k frames; reuse the
    last computed keypoints otherwise. Cuts pose-estimation cost, the
    normally-expensive part -- but keypoints go stale between real updates."""
    return frame_idx % k == 0


def cost_stride_should_run_inference(frame_idx, k):
    """Strategy B (inference-stride): run MediaPipe every frame (full pose
    fidelity kept), but only call the ST-GCN forward pass every k frames.
    Cuts classification cost -- but predictions update less often, i.e.
    higher latency to flag a fall even though pose tracking stays smooth."""
    return frame_idx % k == 0

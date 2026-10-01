"""
Reads frames from either an .mp4 file or a folder of timestamped photos.
Used the same way regardless of strategy -- strategies operate on the
keypoint buffer downstream, not on which frames get read here.
"""
import os
import re
import glob
import cv2

TS_RE = re.compile(r"(\d{2})_(\d{2})_(\d{2}\.\d+)")


def _is_photo_dir(path):
    return os.path.isdir(path)


def _photo_timestamp(fname):
    m = TS_RE.search(os.path.basename(fname))
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def video_fps(path):
    """Native fps of the source (used only for output-video playback speed)."""
    if _is_photo_dir(path):
        return 30.0  # adjust if your photo folders aren't nominally 30fps
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def iter_frames(path):
    """Yields (frame_index, timestamp_seconds, frame) for every real frame.
    No subsampling here -- fps simulation happens later on the keypoint
    buffer (see strategies.py), so we always read every native frame once."""
    if _is_photo_dir(path):
        files = sorted(glob.glob(os.path.join(path, "*")))
        t0 = None
        idx = 0
        for f in files:
            t = _photo_timestamp(f)
            if t is None:
                continue
            if t0 is None:
                t0 = t
            frame = cv2.imread(f)
            if frame is None:
                continue
            yield idx, t - t0, frame
            idx += 1
    else:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, idx / fps, frame
            idx += 1
        cap.release()

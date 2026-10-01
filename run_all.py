"""
python run_all.py <videos_dir>

Runs every clip in <videos_dir> (e.g. videos/fall-01-cam0.mp4, adl-05-cam0.mp4)
through all 5 modes, appending to the 5 results_*.csv files.
Re-run safely -- each run_clip call opens its CSV in append mode.
"""
import sys
import os
import glob

from run_clip import run_clip

MODES = ["cost_stride"]
#"fps_hold", "fps_interp", "cost_skip", "cost_stride"]
COST_K = 3  # shared k for cost_skip / cost_stride sweeps -- change here to sweep other values


def main(videos_dir):
    clips = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not clips:
        print(f"no .mp4 files found in {videos_dir}")
        return

    for clip_path in clips:
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        for mode in MODES:
            print(f"\n=== {clip_id} | mode={mode} ===")
            run_clip(clip_path, clip_id, mode, k=COST_K)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])

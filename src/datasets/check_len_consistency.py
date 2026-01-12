#!/usr/bin/env python3
"""Check length consistency between video frame folders and target .npy files.

Usage examples:
  python src/datasets/check_len_consistency.py --video-root data/Surgery/frames --target-root data/Surgery/targets --chunk-size 32
  python src/datasets/check_len_consistency.py --video-root /path/to/frames --target-root /path/to/targets --chunk-size 16 --sessions sessions.txt
"""
import argparse
import os
import sys
import numpy as np
from src.datasets.utils import get_frame_count_and_fps, get_video


def check_session(session, video_root, target_root, chunk_size):
    video_path = os.path.join(video_root, session)
    target_path = os.path.join(target_root, session + ".npy")

    if not os.path.isdir(video_path):
        ext = ['mp4', 'mpg']
        video_path = get_video(video_root, session, ext)
        real_frame_count, real_fps = get_frame_count_and_fps(video_path)
        # logical
        scale_factor = real_fps / 24
        frame_length = int(real_frame_count / scale_factor)
        return False, f"Missing video directory: {video_path}"
    else:
        frame_length = len(os.listdir(video_path))
    if not os.path.isfile(target_path):
        return False, f"Missing target file: {target_path}"

    num_chunks = int(frame_length // chunk_size)

    try:
        target = np.load(target_path)
    except Exception as e:
        return False, f"Failed loading target {target_path}: {e}"

    try:
        target_chunks = int(target.shape[0])
    except Exception:
        # fallback if target is 1D list-like
        try:
            target_chunks = int(len(target))
        except Exception:
            return False, f"Unable to determine target length for {target_path}"

    if num_chunks != target_chunks:
        msg = (
            f"Mismatch for session={session}: frame_length={frame_length}, "
            f"chunk_size={chunk_size}, num_chunks={num_chunks}, target_chunks={target_chunks}"
        )
        return False, msg

    return True, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-root", required=True, help="Root folder containing per-session frame folders")
    p.add_argument("--target-root", required=True, help="Root folder containing per-session .npy targets")
    p.add_argument("--chunk-size", type=int, default=6, help="Value for cfg.MODEL.CHUNK_SIZE used when building dataset")
    p.add_argument("--sessions", help="Optional file with one session name per line, or comma-separated list of session names")
    p.add_argument("--fail-on-mismatch", action="store_true", help="Exit with non-zero code when any mismatch is found")
    args = p.parse_args()

    if args.sessions:
        if os.path.isfile(args.sessions):
            with open(args.sessions, "r") as f:
                sessions = [l.strip() for l in f if l.strip()]
        else:
            sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        # default: all entries in video root
        sessions = sorted([name for name in os.listdir(args.video_root) if os.path.isdir(os.path.join(args.video_root, name))])

    errors = []
    for session in sessions:
        ok, msg = check_session(session, args.video_root, args.target_root, args.chunk_size)
        if not ok:
            print(msg, file=sys.stderr)
            errors.append((session, msg))

    if errors:
        print(f"Found {len(errors)} problematic sessions", file=sys.stderr)
        if args.fail_on_mismatch:
            sys.exit(2)
        else:
            sys.exit(0)

    print("All sessions consistent")


if __name__ == "__main__":
    main()

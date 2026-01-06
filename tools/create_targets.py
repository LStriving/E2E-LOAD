#!/usr/bin/env python3
"""Create binary one-hot target numpy files for Surgery dataset.

Usage:
  python tools/create_targets.py --json-path ../oad-surgery/masked_label_v5.json --target-dir data/Surgery/targets --test-plot

This creates .npy files in target-dir with shape (length, num_classes+1),
where length = frame_count // 6, num_classes = 8 (labels 0-7), +1 for background.
Each row is one-hot: 1 at the active class, or at background if no action.
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def create_targets(json_path, target_dir, frame_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Define class names (from label_ids 0-7 + background)
    class_names = [
        "carbachol_injection",      # 0
        "CVA_injection",            # 1
        "direct_gonioscopy",        # 2
        "Goniotomy",                # 3
        "CVA_irrigation/aspiration", # 4
        "suture",                   # 5
        "corneal_incision 30 degree", # 6
        "corneal_incision 15 degree", # 7
        "background"                # 8
    ]

    os.makedirs(target_dir, exist_ok=True)

    for video_id, video_data in data.items():
        video_frame_dir = os.path.join(frame_dir, video_id)
        if not os.path.exists(video_frame_dir):
            print(f"Warning: Frame directory {video_frame_dir} does not exist. Skipping.")
            continue

        frame_count = len([f for f in os.listdir(video_frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
        length = frame_count // 6
        num_classes = 8  # 0-7
        target_shape = (length, num_classes + 1)
        target = np.zeros(target_shape, dtype=np.int32)

        # Set actions
        for ann in video_data["annotations"]:
            start_frame, end_frame = ann["segment(frames)"]
            label_id = ann["label_id"]
            start_idx = start_frame // 6
            end_idx = (end_frame // 6) + 1  # inclusive
            target[start_idx:end_idx, label_id] = 1

        # Set background where no action
        for i in range(length):
            if np.sum(target[i, :-1]) == 0:
                target[i, -1] = 1

        # Save
        target_path = os.path.join(target_dir, f"{video_id}.npy")
        np.save(target_path, target)
        print(f"Saved {target_path}, shape {target.shape}")

    return class_names


def plot_heatmap(target_path, class_names, save_path=None):
    target = np.load(target_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(target.T, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.xlabel('Time steps (each ~6 frames)')
    plt.ylabel('Classes')
    plt.title(f'Action Heatmap for {os.path.basename(target_path)}')
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar(label='Presence (0/1)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def check_missing_files(json_path, frame_dir, target_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    missing_frames = []
    missing_targets = []

    for video_id in data.keys():
        video_frame_dir = os.path.join(frame_dir, video_id)
        target_file = os.path.join(target_dir, f"{video_id}.npy")

        if not os.path.exists(video_frame_dir):
            missing_frames.append(video_id)

        if not os.path.exists(target_file):
            missing_targets.append(video_id)

    if missing_frames:
        print("Missing frame directories:")
        for video_id in missing_frames:
            print(f"  - {video_id}")

    if missing_targets:
        print("Missing target files:")
        for video_id in missing_targets:
            print(f"  - {video_id}")

    if not missing_frames and not missing_targets:
        print("All frame directories and target files are present.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-path", default="../oad-surgery/masked_label_v5.json", help="Path to masked_label_v5.json")
    p.add_argument("--target-dir", default="data/Surgery/targets/class8", help="Directory to save .npy targets")
    p.add_argument("--frame-dir", default="data/Surgery/frames", help="Directory containing extracted frames")
    p.add_argument("--test-plot", action="store_true", help="Plot heatmap for first created target")
    p.add_argument("--plot-save", help="Save plot to this path instead of showing")
    p.add_argument("--check-missing", action="store_true", help="Check for missing frame directories or target files")
    args = p.parse_args()

    if args.check_missing:
        check_missing_files(args.json_path, args.frame_dir, args.target_dir)
        return

    class_names = create_targets(args.json_path, args.target_dir, args.frame_dir)

    if args.test_plot:
        # Plot the first one
        npy_files = [f for f in os.listdir(args.target_dir) if f.endswith('.npy')]
        if npy_files:
            target_path = os.path.join(args.target_dir, npy_files[0])
            plot_heatmap(target_path, class_names, args.plot_save)


if __name__ == "__main__":
    main()
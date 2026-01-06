import os
from pathlib import Path
from PIL import Image
from collections import defaultdict
import argparse

def check_frame_dimensions(dataset_root):
    """
    Check that all extracted video frames have consistent height and width across all folders.
    
    Args:
        dataset_root: Path to the dataset root directory
    """
    frames_dir = Path(dataset_root) / "frames"
    
    if not frames_dir.exists():
        print(f"Error: 'frames' directory not found in {dataset_root}")
        return
    
    # Dictionary to store dimensions for each video folder
    dimensions_info = defaultdict(list)
    inconsistent_folders = []
    all_dimensions = set()
    
    print(f"Checking frame dimensions in: {frames_dir}")
    print("-" * 60)
    
    # Iterate through all video folders
    for video_folder in sorted(frames_dir.iterdir()):
        if not video_folder.is_dir():
            continue
            
        frame_files = sorted([f for f in video_folder.glob("*.jpg")])
        
        if not frame_files:
            print(f"Warning: No .jpg files found in {video_folder.name}")
            continue
        
        # Check dimensions of each frame in this folder
        folder_dimensions = []
        inconsistent_in_folder = []
        
        for i, frame_path in enumerate(frame_files):
            try:
                with Image.open(frame_path) as img:
                    width, height = img.size
                    folder_dimensions.append((width, height))
                    
                    # Check if first frame dimensions match current frame
                    if i > 0 and folder_dimensions[i] != folder_dimensions[0]:
                        inconsistent_in_folder.append({
                            'frame': frame_path.name,
                            'dimensions': (width, height),
                            'index': i
                        })
                        
            except Exception as e:
                print(f"Error reading {frame_path}: {e}")
                continue
        
        if folder_dimensions:
            # Get unique dimensions in this folder
            unique_dims = set(folder_dimensions)
            first_dim = folder_dimensions[0]
            
            # Store information
            dimensions_info[video_folder.name] = {
                'dimensions': first_dim,
                'unique_dimensions': len(unique_dims),
                'total_frames': len(frame_files),
                'inconsistent_frames': inconsistent_in_folder
            }
            
            all_dimensions.add(first_dim)
            
            # Report on this folder
            if len(unique_dims) > 1:
                inconsistent_folders.append(video_folder.name)
                print(f"❌ {video_folder.name}: INCONSISTENT")
                print(f"   First frame: {first_dim}")
                print(f"   Found {len(unique_dims)} different dimensions:")
                for dim in unique_dims:
                    count = folder_dimensions.count(dim)
                    print(f"     - {dim}: {count} frames")
            else:
                print(f"✓ {video_folder.name}: {first_dim} ({len(frame_files)} frames)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not dimensions_info:
        print("No valid frame folders found.")
        return
    
    # Overall consistency check
    if len(all_dimensions) == 1:
        print(f"✅ ALL FOLDERS CONSISTENT: {list(all_dimensions)[0]}")
    elif len(all_dimensions) > 1:
        print(f"⚠️  INCONSISTENT ACROSS FOLDERS: Found {len(all_dimensions)} different dimensions")
        print("\nDimension breakdown:")
        for dim in sorted(all_dimensions):
            folders_with_dim = [name for name, info in dimensions_info.items() 
                               if info['dimensions'] == dim]
            print(f"  {dim}: {len(folders_with_dim)} folders")
            if len(folders_with_dim) <= 10:  # Don't list too many
                for folder in folders_with_dim:
                    print(f"    - {folder}")
    
    # Report on folders with internal inconsistency
    if inconsistent_folders:
        print(f"\n⚠️  {len(inconsistent_folders)} folders have internal inconsistencies:")
        for folder in inconsistent_folders:
            info = dimensions_info[folder]
            print(f"\n  {folder}:")
            print(f"    Expected: {info['dimensions']}")
            for inc in info['inconsistent_frames']:
                print(f"    Frame {inc['frame']} (index {inc['index']}): {inc['dimensions']}")

def main():
    parser = argparse.ArgumentParser(
        description='Check consistency of video frame dimensions'
    )
    parser.add_argument(
        'dataset_root',
        help='Path to the dataset root directory containing "frames/" folder'
    )
    
    args = parser.parse_args()
    check_frame_dimensions(args.dataset_root)

if __name__ == "__main__":
    main()
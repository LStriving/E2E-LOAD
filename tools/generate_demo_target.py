import os
import numpy as np
from tqdm import tqdm

def generate(videos_frames_root, output_dir, video_name=None, resume=True):
    """
    Generate the target file for the demo.
    """
    if not os.path.exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        os.makedirs(output_dir)
    if video_name is not None:
        to_extract = [video_name]
    for video in tqdm(to_extract if video_name is not None else os.listdir(videos_frames_root)):
        video_path = os.path.join(videos_frames_root, video)
        target_path = os.path.join(output_dir, video + ".npy")
        if resume and os.path.exists(target_path):
            continue
        frame_length = len(os.listdir(video_path))
        
        num_chunks = int(frame_length // 6)
        target = np.zeros((num_chunks, 9), dtype=np.int64)
        
        np.save(target_path, target)
    
if __name__ == "__main__":
    videos_frames_root = "/mnt/cephfs/home/liyirui/project/E2E-LOAD/data/Surgery_new/frames"
    output_dir = "/mnt/cephfs/home/liyirui/project/E2E-LOAD/data/Surgery_new/targets"
    
    generate(videos_frames_root, output_dir)

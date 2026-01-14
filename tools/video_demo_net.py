#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
import time
import pickle
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from bisect import bisect_right
from decord import VideoReader, cpu

# Add project root to path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Local imports
import src.utils.checkpoint as cu
import src.utils.logging as log_utils
from src.models import build_model
import src.datasets.utils as utils
from src.utils import evalution as evaluation  # Fixed typo

logger = log_utils.get_logger(__name__)


class VideoInferenceRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-calculate normalization tensors on GPU to speed up loop
        self.mean = torch.tensor(cfg.DATA.MEAN, device=self.device).view(1, 1, 1, 3)
        self.std = torch.tensor(cfg.DATA.STD, device=self.device).view(1, 1, 1, 3)
        
        self.model = self._setup_model()
        
    def _setup_model(self):
        """Builds model and loads checkpoints."""
        logger.info("Building model...")
        model = build_model(self.cfg)
        model.eval()
        cu.load_test_checkpoint(self.cfg, model)
        return model.to(self.device)

    def _preprocess_frame_batch(self, raw_frames_numpy):
        """
        Handles Normalization, Permutation, and Spatial Sampling on GPU.
        Input: [T, H, W, C] numpy
        Output: [1, C, T, H, W] torch tensor
        """
        # To GPU and Normalize
        frames = torch.tensor(raw_frames_numpy, device=self.device).float()
        frames = frames / 255.0
        frames = (frames - self.mean) / self.std
        
        # Permute: [T, H, W, C] -> [C, T, H, W]
        frames = frames.permute(3, 0, 1, 2)
        
        # Spatial Sampling (Center Crop)
        crop_size = self.cfg.DATA.TEST_CROP_SIZE
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=1,  # Center crop
            min_scale=crop_size,
            max_scale=crop_size,
            crop_size=crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False,
            motion_shift=False,
        )
        
        # Add batch dimension: [C, T, H, W] -> [1, C, T, H, W]
        return frames.unsqueeze(0)

    def _calculate_memory_indices(self, work_start):
        """Calculates Long-Term Memory indices for the specific model architecture."""
        long_end = work_start - 1
        long_start = long_end - self.cfg.MODEL.LONG_MEMORY_NUM_SAMPLES * self.cfg.MODEL.LONG_MEMORY_SAMPLE_RATE
        long_indices = np.arange(long_start + 1, long_end + 1)
        
        # Get unique indices and counts
        ulong_indices, repeat_times = np.unique(long_indices, return_counts=True)
        
        # Create mask
        memory_key_padding_mask = np.zeros(len(long_indices))
        last_zero = bisect_right(long_indices, 0) - 1
        if last_zero > 0:
            memory_key_padding_mask[:last_zero] = float('-inf')
            
        mask_tensor = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0).to(self.device)
        
        return ulong_indices, repeat_times, mask_tensor

    def run_session(self, session_name: str, data_root: Path):
        """Runs inference for a single video session."""
        
        # 1. Path Setup
        video_dir = data_root / self.cfg.DATA.VIDEO_FOLDER 
        target_dir = data_root / self.cfg.DATA.TARGET_FOLDER
        
        video_path = utils.get_video(str(video_dir), session_name, self.cfg.DATA.VIDEO_EXT)
        target_path = target_dir / (session_name + ".npy")
        
        # 2. Load Video
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        real_frame_count = len(vr)
        
        # 3. Calculate "Logical" Frames (Frame Rate alignment)
        scale_factor = vr.get_avg_fps() / self.cfg.DATA.TARGET_FPS
        frame_length = int(real_frame_count / scale_factor)
        num_chunks = int(frame_length // self.cfg.MODEL.CHUNK_SIZE)
        
        # 4. Load Targets (if available)
        gt_target = None
        if target_path.exists():
            gt_target = np.load(target_path)
            min_len = min(num_chunks, gt_target.shape[0])
            gt_target = gt_target[:min_len]

        # 5. Prepare Chunks
        frame_indices = np.arange(frame_length)
        chunk_indices = np.array(np.split(
            frame_indices[: num_chunks * self.cfg.MODEL.CHUNK_SIZE],
            num_chunks,
            axis=0
        ))
        
        if self.cfg.MODEL.CHUNK_SIZE > 1:
            chunk_indices = chunk_indices[:, self.cfg.MODEL.CHUNK_SAMPLE_RATE // 2 :: self.cfg.MODEL.CHUNK_SAMPLE_RATE]

        # 6. Inference Loop
        preds = []
        collected_gts = []
        self.model.empty_cache() 
        
        with torch.no_grad():
            # === RESTORED ORIGINAL LOOP STRUCTURE ===
            # This ensures work_start increments by 1 (sliding window), not jumping chunks
            range_start = range(0, num_chunks + 1)
            range_end = range(self.cfg.MODEL.WORK_MEMORY_NUM_SAMPLES, num_chunks + 1)
            
            for work_start, work_end in zip(range_start, range_end):
                
                # work_indices calculation must mimic original: [start, end)
                # This ensures `last_idx` aligns with the sliding window edge
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::self.cfg.MODEL.WORK_MEMORY_SAMPLE_RATE]
                
                if work_start == 0:
                    # First step: Load full batch
                    selected_chunks = chunk_indices[work_indices]
                    frames_to_load = selected_chunks.flatten()
                    if gt_target is not None:
                        current_gt = gt_target[::self.cfg.MODEL.WORK_MEMORY_SAMPLE_RATE][work_indices]
                else:
                    # Sliding step: Load only the newest frame at the end of the window
                    last_idx = work_indices[-1]
                    frames_to_load = chunk_indices[last_idx]
                    if gt_target is not None:
                        # Original logic: take the last target corresponding to the new frame
                        current_gt = [gt_target[::self.cfg.MODEL.WORK_MEMORY_SAMPLE_RATE][last_idx]]

                frames_to_load = np.clip(frames_to_load, 0, real_frame_count - 1)
                
                # Long Memory Indices (Correctly relative to work_start)
                ulong_indices, repeat_times, mask = self._calculate_memory_indices(work_start)

                # Load & Process
                raw_frames = vr.get_batch(frames_to_load).asnumpy()
                input_tensor = self._preprocess_frame_batch(raw_frames)

                # Inference
                start = time.time()
                scores = self.model.stream_inference(input_tensor, ulong_indices, repeat_times, mask)
                duration = time.time() - start
                scores = scores[0].softmax(dim=-1).cpu().numpy()

                # Collect Results
                if work_start == 0:
                    preds.extend(list(scores))
                else:
                    preds.extend([list(scores[-1])])
                
                if gt_target is not None:
                    collected_gts.extend(current_gt)

        return np.array(preds), (np.array(collected_gts) if gt_target is not None else None), num_chunks, duration

def demo(cfg):
    """
    Main entry point for inference.
    """
    log_utils.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Starting Online Inference Speed Test.")

    # Initialize Runner
    runner = VideoInferenceRunner(cfg)
    
    # Determine Input Sessions
    if cfg.DEMO.ALL_TEST:
        sessions = getattr(cfg.DATA, "TEST_SESSION_SET")
    else:
        sessions = cfg.DEMO.INPUT_VIDEO

    # Metrics
    total_inference_time = 0
    total_frames_processed = 0
    all_preds = {}
    all_gts = {}
    
    # Output Setup
    save_dir = Path(cfg.OUTPUT_DIR) / f"inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(cfg.DATA.PATH_TO_DATA_DIR) / cfg.DATA.PATH_PREFIX

    for i, session in enumerate(tqdm(sessions)):
        try:
            
            
            # Run Inference
            pred_scores, gt_targets, num_chunks, duration = runner.run_session(session, data_root)
            
            
            total_inference_time += duration
            total_frames_processed += num_chunks
            
            # Save individual result
            np.save(save_dir / f"res_{session}.npy", pred_scores)
            
            # Evaluation per session
            if gt_targets is not None:
                result = evaluation.eval_perframe(cfg, gt_targets, pred_scores)
                logger.info(f'[{i+1}/{len(sessions)}] {session} | mAP: {result["mean_AP"]:.4f}')
                all_preds[session] = pred_scores
                all_gts[session] = gt_targets
                
        except Exception as e:
            logger.error(f"Failed processing session {session}: {e}")
            continue

    # Final Evaluation & Stats
    if all_gts:
        logger.info('Performing Global Evaluation...')
        final_results = evaluation.eval_perframe(
            cfg,
            np.concatenate(list(all_gts.values()), axis=0),
            np.concatenate(list(all_preds.values()), axis=0),
        )
        logger.info(f"Global mAP: {final_results['mean_AP']}")
        
        # Save complete results
        with open(save_dir / 'full_results.pkl', 'wb') as f:
            pickle.dump({
                'cfg': cfg,
                'preds': all_preds,
                'gts': all_gts
            }, f)

    fps = total_frames_processed / total_inference_time if total_inference_time > 0 else 0.0
    logger.info(f"Total Time: {total_inference_time:.2f}s | FPS: {fps:.2f}")
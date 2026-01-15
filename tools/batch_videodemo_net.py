#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
import time
import pickle
import torch
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Local imports
import src.utils.logging as log_utils
from src.utils import evalution as evaluation
from .video_demo_net import VideoInferenceRunner

# We need to set the start method to 'spawn' for CUDA compatibility
# try:
#     mp.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

logger = log_utils.get_logger(__name__)

def worker_func(gpu_id, cfg, session_chunk, save_dir, data_root):
    """
    Worker function to be run in a separate process.
    """
    # Setup logging per worker
    # log_utils.setup_logging(cfg.OUTPUT_DIR) # Optional: setup distinct logs per process
    if gpu_id == 0:
        logger.info(f"Worker {gpu_id} starting processing {len(session_chunk)} videos.")

    runner = VideoInferenceRunner(cfg, gpu_id=gpu_id)
    
    local_preds = {}
    local_gts = {}
    
    # Progress bar only for the master process to avoid console clutter
    iterator = tqdm(session_chunk, position=gpu_id) if gpu_id == 0 else session_chunk
    
    for session in iterator:
        try:
            pred_scores, gt_targets, num_chunks, duration = runner.run_session(session, data_root)
            
            # Save individual result immediately
            np.save(save_dir / f"{session}.npy", pred_scores)
            
            local_preds[session] = pred_scores
            if gt_targets is not None:
                local_gts[session] = gt_targets
                
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed processing session {session}: {e}")
            continue

    # Save local results to a temp file for the master process to pick up
    worker_result_path = save_dir / f"worker_{gpu_id}_results.pkl"
    with open(worker_result_path, 'wb') as f:
        pickle.dump({'preds': local_preds, 'gts': local_gts}, f)
        
    if gpu_id == 0:
        logger.info(f"Worker {gpu_id} finished.")


def demo(cfg):
    """
    Main entry point for Distributed Inference.
    """
    log_utils.setup_logging(cfg.OUTPUT_DIR)
    
    # Determine devices
    num_gpus = min(cfg.NUM_GPUS, torch.cuda.device_count())
    if num_gpus < 1:
        raise RuntimeError("No GPUs found for inference.")
    
    logger.info(f"Starting Multi-GPU Inference on {num_gpus} GPUs.")

    # Determine Input Sessions
    if cfg.DEMO.ALL_TEST:
        sessions = getattr(cfg.DATA, "TEST_SESSION_SET")
    else:
        sessions = cfg.DEMO.INPUT_VIDEO
    
    # Sort sessions to ensure deterministic splitting
    sessions = sorted(sessions)

    # Output Setup
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = Path(cfg.OUTPUT_DIR) / f"inference_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg.DATA.PATH_TO_DATA_DIR) / cfg.DATA.PATH_PREFIX

    # Split sessions among GPUs
    chunk_size = int(np.ceil(len(sessions) / num_gpus))
    session_chunks = [sessions[i:i + chunk_size] for i in range(0, len(sessions), chunk_size)]
    
    # Handle edge case where we have more GPUs than chunks (small datasets)
    while len(session_chunks) < num_gpus:
        session_chunks.append([])

    start_time = time.time()

    # Launch Processes
    mp.spawn(
        worker_func,
        nprocs=num_gpus,
        args=(cfg, session_chunks, save_dir, data_root),
        join=True
    )

    total_time = time.time() - start_time
    logger.info(f"All workers finished in {total_time:.2f}s. Aggregating results...")

    # Aggregation
    all_preds = {}
    all_gts = {}
    
    for gpu_id in range(num_gpus):
        worker_file = save_dir / f"worker_{gpu_id}_results.pkl"
        if worker_file.exists():
            with open(worker_file, 'rb') as f:
                data = pickle.load(f)
                all_preds.update(data['preds'])
                all_gts.update(data['gts'])
            # Clean up temp file
            worker_file.unlink()

    # Global Evaluation
    if all_gts:
        logger.info('Performing Global Evaluation...')
        final_results = evaluation.eval_perframe(
            cfg,
            np.concatenate([all_gts[k] for k in sorted(all_gts.keys())], axis=0),
            np.concatenate([all_preds[k] for k in sorted(all_preds.keys())], axis=0),
        )
        logger.info(f"Global mAP: {final_results['mean_AP']}")
        
        # Save complete consolidated results
        with open(save_dir / 'full_results.pkl', 'wb') as f:
            pickle.dump({
                'cfg': cfg,
                'preds': all_preds,
                'gts': all_gts
            }, f)
            
    logger.info(f"Results saved to {save_dir}")
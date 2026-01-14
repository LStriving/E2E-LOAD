#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy as np
import os
import torch
import time
import pickle
from datetime import datetime 

from decord import VideoReader, cpu
from bisect import bisect_right
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import src.utils.checkpoint as cu
import src.utils.logging as logging
from src.models import build_model


import src.datasets.helper as helper
import src.datasets.utils as utils
from src.utils import evalution

logger = logging.get_logger(__name__)

def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
         src/config/defaults.py
    """
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Online Inference Speed Test.")
    # logger.info(cfg)
    model = build_model(cfg)
    
    # Setuo model to test mode. 
    model.eval()

    # params = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH)
    
    cu.load_test_checkpoint(cfg, model) 
    logger.info("Succeed Loading the PreTrained Parameters.")

    if cfg.DEMO.ALL_TEST:
        sessions = getattr(cfg.DATA, "TEST_SESSION_SET")
    else:
        sessions = cfg.DEMO.INPUT_VIDEO 
    
    inference_time = 0
    total_frames = 0
    found_target = False
    pred_scores = {}
    gt_targets = {}
    # 预定义 GPU 上的 Normalization (利用 Broadcasting，不需要复杂的 Transform)
    mean_tensor = torch.tensor(cfg.DATA.MEAN, device='cuda').view(1, 3, 1, 1)
    std_tensor = torch.tensor(cfg.DATA.STD, device='cuda').view(1, 3, 1, 1)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = cfg.OUTPUT_DIR + '_'
    
    # loading the video info; 
    for idx, session in enumerate(sessions):
        data_root = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.PATH_PREFIX)
        target_root = os.path.join(data_root, cfg.DATA.TARGET_FORDER)


        video_path = utils.get_video(os.path.join(data_root, cfg.DATA.VIDEO_FORDER), session, cfg.DATA.VIDEO_EXT)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        real_frame_count = len(vr)
        scale_factor = vr.get_avg_fps() / cfg.DATA.TARGET_FPS
        # "logical" length
        frame_length = int(real_frame_count / scale_factor)
        
        frame_indices = np.arange(frame_length)
        num_chunks = int(frame_length // cfg.MODEL.CHUNK_SIZE)
        total_frames += num_chunks
        
        # Load the related targets; 
        target_path = os.path.join(target_root, session + ".npy")
        if os.path.exists(target_path):
            target = np.load(target_path)
            found_target = True
        if found_target:
            # 简单的对齐检查，防止 crash
            min_len = min(num_chunks, target.shape[0])
            target = target[:min_len]
            if num_chunks != target.shape[0]:
                logger.warning(f"{session}.npy has an \
                unexpeceted shape: {target.shape[0]} (expected: {num_chunks})")
        
        # 切分 chunk
        # [Num_Chunks, Chunk_Size]
        chunk_indices = np.array(np.split(
            frame_indices[: num_chunks * cfg.MODEL.CHUNK_SIZE],
            num_chunks,
            axis=0
        ))
        
        
        # 采样 (Temporal Downsampling) chunk_indices [num_chunks, samples_per_chunk]
        if cfg.MODEL.CHUNK_SIZE > 1:
            # 中心采样，开始采样索引 CHUNK_SAMPLE_RATE // 2
            chunk_indices = chunk_indices[:, cfg.MODEL.CHUNK_SAMPLE_RATE // 2 :: cfg.MODEL.CHUNK_SAMPLE_RATE]
        else:
            chunk_indices = chunk_indices # No-op


        single_pred = []
        single_gt = []

        model.empty_cache()
        
        with torch.no_grad():
            if found_target:
                target_sampled = target[::cfg.MODEL.WORK_MEMORY_SAMPLE_RATE]
            
            for work_start, work_end in zip(range(0, num_chunks + 1),
                                            range(cfg.MODEL.WORK_MEMORY_NUM_SAMPLES, num_chunks + 1)):  
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::cfg.MODEL.WORK_MEMORY_SAMPLE_RATE]

                if work_start == 0: 
                    # 第一步：加载整个 Work Memory 的帧
                    # Flatten indices: [chunk_0_frames, chunk_1_frames, ...]
                    selected_chunks = chunk_indices[work_indices] # [n_work, n_samples]
                    frames_indices_to_load = selected_chunks.flatten()
                    if found_target:
                        current_target_batch = target_sampled[work_indices]
                    
                else:
                    # 后续步骤：只加载最新的一个 Chunk
                    # 原代码逻辑：indice = work_indices[-1]
                    last_idx = work_indices[-1]
                    frames_indices_to_load = chunk_indices[last_idx]
                    
                    if found_target:
                        current_target_batch = [target_sampled[last_idx]] 
                
                # 确保索引不越界
                frames_indices_to_load = np.clip(frames_indices_to_load, 0, real_frame_count - 1)
                
                # long
                long_end = work_start - 1 
                long_start = long_end - cfg.MODEL.LONG_MEMORY_NUM_SAMPLES * cfg.MODEL.LONG_MEMORY_SAMPLE_RATE
                long_indices = np.arange(long_start+1, long_end+1)
                ulong_indices, repeat_times = np.unique(long_indices, return_counts=True) 

                memory_key_padding_mask = np.zeros(len(long_indices))
                last_zero = bisect_right(long_indices, 0) - 1 
                if last_zero > 0:
                    memory_key_padding_mask[:last_zero] = float('-inf')
                memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0).cuda()

                # Load images
                raw_frames = vr.get_batch(frames_indices_to_load).asnumpy()
                work_frames = torch.tensor(raw_frames).cuda(non_blocking=True).float() # [T, H, W, C]
                work_frames = work_frames / 255.0
                # Permute: [T, H, W, C] -> [C, T, H, W] (For Normalize/Crop)
                # Normalize (Manual implementation on GPU is faster than creating transforms)
                work_frames = (work_frames - mean_tensor) / std_tensor
                work_frames = work_frames.permute(3, 0, 1, 2)

                # Load the images;
                min_scale = cfg.DATA.TEST_CROP_SIZE
                max_scale = cfg.DATA.TEST_CROP_SIZE
                crop_size = cfg.DATA.TEST_CROP_SIZE
                work_frames = utils.spatial_sampling(
                    work_frames,
                    spatial_idx=1, 
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=False, 
                    inverse_uniform_sampling=False, 
                    aspect_ratio=None, 
                    scale=None, 
                    motion_shift=False, 
                ) # C, NF, H, W; 

                work_frames = work_frames.cuda(non_blocking=True).unsqueeze(0)
                
                start = time.time()
                score = model.stream_inference(work_frames, ulong_indices, repeat_times, memory_key_padding_mask)
                delta = time.time() - start
                inference_time += delta
                
                score = score[0].softmax(dim=-1).cpu().numpy() # torch.Size([1, 32, 22])
                

                if work_start == 0: 
                    single_pred.extend(list(score))
                else:
                    single_pred.extend([list(score[-1])])
                
                if found_target:
                    single_gt.extend(current_target_batch)
        
        # assert {len(single_pred), num_chunks, len(single_gt)}
        
        # save the predicted results(no gt);
        single_pred = np.array(single_pred)
        save_dir = save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, session+".npy")
        np.save(save_path, single_pred)

        
        # performing the single test
        if found_target:
            result = evalution.eval_perframe(cfg, single_gt, single_pred)
            logger.info(f'Process: {idx+1}/{len(sessions)} | Video: {session} | mAP: {result["mean_AP"]:.4f}')
            pred_scores[session] = single_pred
            gt_targets[session] = single_gt

    if found_target:
        logger.info('Performing the Last Test.')
        results = evalution.eval_perframe(
                cfg,
                np.concatenate(list(gt_targets.values()), axis=0),
                np.concatenate(list(pred_scores.values()), axis=0),
        )

    logger.info("All Inference Time, {}".format(inference_time))
    logger.info("Num Chunks, {}".format(total_frames))
    logger.info("FPS, {}".format(float(total_frames/inference_time)))
    if found_target:
        logger.info("mAP, {}".format(results["mean_AP"]))

    
    if cfg.DEMO.ALL_TEST and found_target:
        logger.info('Saving Predicted Files to: {}'.format(save_path))

        pickle.dump({
            'cfg': cfg,
            'perframe_pred_scores': pred_scores,
            'perframe_gt_targets': gt_targets,
        }, open(save_path + '.pkl', 'wb'))

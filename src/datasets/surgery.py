import json
from tqdm import tqdm
from decord import VideoReader, cpu

from .check_len_consistency import check_session
from .thumos import *

@DATASET_REGISTRY.register()
class Surgery(torch.utils.data.Dataset):
    '''
    Video-based Dataset (on-the-fly loading mode)
    '''
    def __init__(self, cfg, mode, num_retries=2):
        assert mode in ['train', 'val', 'test'], f'Mode "{mode}" not supported for Surgery'
        self.mode = mode
        self.cfg = cfg

        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB 
        self.data_anno = self.cfg.DATA.ANNO
        self.data_info = self.cfg.DATA.DATA_INFO
        self.num_classes = self.cfg.DATA.NUM_CLASSES
        self.class_names = self.cfg.DATA.CLASS_NAMES
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0

        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1  
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Surgery {}...".format(mode))
        self._construct_loader()
        self.aug = False  
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0  
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:  
                self.rand_erase = True

        self.aug_transform = transform.create_random_augment_tensor(
            auto_augment_str=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION
        )
        # 预先初始化 Erase Transform
        self.erase_transform = None
        if self.rand_erase:
            self.erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu", # 如果是在 CPU 做预处理，保持 cpu
            )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch 

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.samples)

    def _construct_target_tolocal(self):
        """
        Construct target for initialization
        """
        # check if exists already
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root, exist_ok=True)
        exist_targets = os.listdir(self.target_root)
        if len(exist_targets) == len(self.cfg.DATA.TRAIN_SESSION_SET) + len(self.cfg.DATA.TEST_SESSION_SET):
            # TODO: santity check
            # Gather sessions
            sessions = self.cfg.DATA.TRAIN_SESSION_SET + self.cfg.DATA.TEST_SESSION_SET
            for session in sessions:
                ok, _ = check_session(session, self.video_root, self.target_root, self.cfg.MODEL.CHUNK_SIZE)
                if not ok:
                    break
                else:
                    return
        self.__create_target()
        
    def __create_target(self):
        assert self.data_anno is not None, f'Annotation file not provided.'
        with open(self.data_anno, 'r') as f:
            data = json.load(f)

        for video_id, video_anno in tqdm(data.items(), desc="Constructing target"):
            if  video_id not in self.cfg.DATA.TRAIN_SESSION_SET and \
                video_id not in self.cfg.DATA.TEST_SESSION_SET:
                continue
            # real_frame_count = video_anno['frame_count']
            # real_fps = video_anno['fps']
            video_path = utils.get_video(self.video_root, video_id, self.cfg.DATA.VIDEO_EXT)
            real_frame_count, real_fps = utils.get_frame_count_and_fps(video_path)
            scale_factor = real_fps / self.cfg.DATA.TARGET_FPS
            # "logical" length
            frame_len = int(real_frame_count / scale_factor)
            length = int(frame_len // self.cfg.MODEL.CHUNK_SIZE)
            target_shape = (length, self.num_classes)
            target = np.zeros(target_shape, dtype=np.int32)

            for ann in video_anno['annotations']:
                real_start_frame, real_end_frame = ann['segment(frames)']
                label_id = int(ann['label_id'])
                start_frame, end_frame = real_start_frame / scale_factor, real_end_frame / scale_factor
                start_idx = int(start_frame // self.cfg.MODEL.CHUNK_SIZE)
                end_idx = int(end_frame // self.cfg.MODEL.CHUNK_SIZE) + 1
                target[start_idx:end_idx, label_id] = 1

            # Set background where no action
            for i in range(length):
                if np.sum(target[i, :-1]) == 0:
                    target[i, -1] = 1
            
            target_path = os.path.join(self.target_root, f'{video_id}.npy')
            np.save(target_path, target)
    
        return
    
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self.data_root = self.data_root = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.PATH_PREFIX
        ) # "data" "Surgery"

        ## Video Path
        self.video_root = os.path.join(
            self.data_root, self.cfg.DATA.VIDEO_FORDER
        ) # data/Surgery videos

        # TODO: remove target
        self.target_root = os.path.join(
            self.data_root, self.cfg.DATA.TARGET_FORDER
        )
        self._construct_target_tolocal()
        
        # self.sessions contains all the {train/val} sessions;
        self.sessions = getattr(
            self.cfg.DATA,
            ("train" if self.mode == "train" else "test").upper()
            + "_SESSION_SET",
        ) 

        self.cur_iter = 0
        self.epoch = 0.0

        # --- 2. Load Samples (New Logic) ---
        # Instead of getting separated lists, we get a list of VideoSampleInfo objects.
        # Ensure load_video_samples_decord is imported or defined in the file.
        raw_samples_list = helper.load_video_samples_decord(
            self.cfg,
            self.sessions,
            self.video_root,
            self.target_root,
            self.mode,
            return_list=True,
        )

        # --- 3. Handle Multi-Clip Expansion (for Testing/Validation) ---
        # If we need multiple clips (e.g. 3 spatial crops) per video, we simply duplicate
        # the sample object reference. This is lightweight and efficient.
        
        if self._num_clips > 1:
            self.samples = list(
                chain.from_iterable(
                    [[x] * self._num_clips for x in raw_samples_list]
                )
            )
        else:
            self.samples = raw_samples_list

        # --- 4. Construct Spatial-Temporal Indices ---
        # This tells __getitem__ which specific crop (0, 1, or 2) to take for each duplicated sample.
        # We iterate over the *unique* raw samples count.
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(raw_samples_list))
                ]
            )
        ) 

        # --- 5. Cleanup & Verification ---
        logger.info(
            "Constructing Surgery dataloader (size: {}).".format(
                len(self.samples),
            )  
        ) 
        
        assert (
            len(self.samples) > 0
        ), "Failed to load Surgery dataset"

    def _load_raw_frames_decord(self, vr, indices, time_diff_prob, gaussian_prob):
        """
        Helper function to load frames using Decord and apply raw level augmentations
        (Time Difference / Gaussian Blur) to match original helper.load_frames logic.
        """
        # 1. Decord Loading
        # vr.get_batch returns (T, H, W, C) in uint8 (CPU)
        # We immediately convert to numpy to interface with Torch
        buffer = vr.get_batch(indices).asnumpy()
        buffer = buffer[..., ::-1].copy() # make it the same as cv2
        frames = torch.from_numpy(buffer).float() # (T, H, W, C)

        # 2. Raw Augmentations
        return helper.aug_raw_frames(time_diff_prob, gaussian_prob, frames)
        

    def __getitem__(self, index):
        # 0. Retrieve Pre-calculated Sample Info
        sample = self.samples[index] # VideoSampleInfo Object
        
        # 1. Initialize Video Reader (Efficient IO: Open once)
        try:
            # ctx=cpu(0) is best for Dataloader multiprocessing
            vr = VideoReader(sample.video_path, ctx=cpu(0),num_threads=1)
        except Exception as e:
            # Fallback or strict error handling
            raise RuntimeError(f"Failed to open video {sample.video_path}: {e}")

        # 2. Setup Augmentation Params
        p_dt = self.p_convert_dt if self.mode in ["train"] else 0.0
        p_gauss = 0.0
        if self.mode in ["train"]:
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            
        elif self.mode in ["val"]:
            spatial_sample_index = -2
            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE
            crop_size = self.cfg.DATA.TEST_CROP_SIZE
            
        elif self.mode in ["test"]:
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(f"Does not support {self.mode} mode")

        # 3. Load Work Memory Frames
        # Replaces helper.load_frames logic
        work_frames, work_time_diff_aug = self._load_raw_frames_decord(
            vr, 
            sample.work_indices, 
            time_diff_prob=p_dt, 
            gaussian_prob=p_gauss
        )

        # 4. Load Long Memory Frames (If enabled)
        long_frames = None
        long_key_padding_mask = None
        
        if self.cfg.MODEL.LONG_MEMORY_ENABLE and sample.long_memory_indices is not None:
            long_indices, long_key_padding_mask = sample.long_memory_indices

            # Optimization: 
            # We fetch directly using the indices. The list `long_indices` already contains 
            # the full sequence (including repeats), so we DO NOT need the complex 
            # repeat_interleave/einops logic from the old code.
            long_frames, _ = self._load_raw_frames_decord(
                vr, 
                long_indices, 
                time_diff_prob=p_dt, 
                gaussian_prob=p_gauss
            )
            # long_frames is now (Total_Long_Frames, H, W, C)

        # 5. Prepare Output Containers
        num_aug = (
            self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
            if self.mode in ["train"] else 1
        )
        
        f_out_work = [None] * num_aug
        f_out_long = [None] * num_aug if self.cfg.MODEL.LONG_MEMORY_ENABLE else None

        # 6. Apply Spatial/Color Augmentations (Loop over num_aug)
        # This part remains largely similar to preserve training recipe exactness
        for idx in range(num_aug):
            
            # --- Process Long Frames ---
            if self.cfg.MODEL.LONG_MEMORY_ENABLE:
                # Clone for this specific augmentation view
                curr_long = long_frames.clone() 
                curr_long = curr_long / 255.0 # Normalize 0-1

                # SSL Color Jitter
                if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                    curr_long = transform.color_jitter_video_ssl(
                        curr_long,
                        bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                        hue=self.cfg.DATA.SSL_COLOR_HUE,
                        p_convert_gray=self.p_convert_gray,
                        moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                        gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                        gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                    )

                # Auto Augment
                if self.aug and self.cfg.AUG.AA_TYPE:
                    curr_long = self._apply_auto_augment(curr_long) # Extracted logic below

                # Normalization
                curr_long = utils.tensor_normalize(curr_long, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                curr_long = curr_long.permute(3, 0, 1, 2) # T H W C -> C T H W

                # Spatial Sampling (Crop/Resize)
                curr_long = self._apply_spatial_sampling(
                    curr_long, spatial_sample_index, min_scale, max_scale, crop_size
                )

                # Random Erasing & Mask Gen
                curr_long = self._apply_erase_and_mask(curr_long)
                
                f_out_long[idx] = curr_long

            # --- Process Work Frames ---
            curr_work = work_frames.clone()
            curr_work = curr_work / 255.0

            if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                 curr_work = transform.color_jitter_video_ssl(
                    curr_work,
                    bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                    hue=self.cfg.DATA.SSL_COLOR_HUE,
                    p_convert_gray=self.p_convert_gray,
                    moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                    gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                    gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                )

            if self.mode in ["train"] and self.aug and self.cfg.AUG.AA_TYPE:
                curr_work = self._apply_auto_augment(curr_work)

            curr_work = utils.tensor_normalize(curr_work, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
            curr_work = curr_work.permute(3, 0, 1, 2) # T H W C -> C T H W

            curr_work = self._apply_spatial_sampling(
                curr_work, spatial_sample_index, min_scale, max_scale, crop_size
            )
            
            curr_work = self._apply_erase_and_mask(curr_work)
            
            f_out_work[idx] = curr_work

        # 7. Finalize Return Structures
        final_work = f_out_work[0] if num_aug == 1 else f_out_work
        
        # Handle Labels
        labels = sample.label # Directly from object
        if num_aug > 1 and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            labels = [labels] * num_aug
            # If labels was numpy, list multiplication works. 
            # If it was tensor, you might want torch.stack or similar depending on downstream.
            # Assuming list of numpy arrays based on original code logic.

        if self.cfg.MODEL.LONG_MEMORY_ENABLE:
            final_long = f_out_long[0] if num_aug == 1 else f_out_long
            
            # Mask Processing (Original logic preserved)
            if self.cfg.DATA.ZERO_MASK:
                long_key_padding_mask_ = np.repeat(long_key_padding_mask, 3) # Why 3? Maybe for channels?
                long_key_padding_mask_[long_key_padding_mask_ == 0] = 1
                long_key_padding_mask_[long_key_padding_mask_ < 0] = 0
                
                # Apply mask broadcasting
                # Assuming final_long is Tensor (C, T, H, W) or list of Tensors
                # If list (num_aug > 1), this needs to be applied to each
                if isinstance(final_long, list):
                    final_long = [fl * long_key_padding_mask_[None, :, None, None] for fl in final_long]
                else:
                    final_long = final_long * long_key_padding_mask_[None, :, None, None]
                
                # final_long type ensure
                # Note: list doesn't have .to(), so only apply if tensor
                if not isinstance(final_long, list):
                    final_long = final_long.to(torch.float32)
            return final_work, final_long, long_key_padding_mask, labels
        else:
            return final_work, labels

    # --- Refactored Augmentation Blocks to keep __getitem__ clean ---
    
    # def _apply_auto_augment(self, frames):
    #     # frames: T H W C
    #     aug_transform = create_random_augment(
    #         input_size=(frames.size(1), frames.size(2)),
    #         auto_augment=self.cfg.AUG.AA_TYPE,
    #         interpolation=self.cfg.AUG.INTERPOLATION,
    #     )
    #     frames = frames.permute(0, 3, 1, 2) # T H W C -> T C H W
    #     list_img = self._frame_to_list_img(frames)
    #     list_img = aug_transform(list_img)
    #     frames = self._list_img_to_frames(list_img)
    #     frames = frames.permute(0, 2, 3, 1) # Back to T H W C
    #     return frames

    def _apply_auto_augment(self, frames):
        # 1. 维度调整：Augmentation 需要 (T, C, H, W) 或 (C, H, W)
        frames = frames.permute(0, 3, 1, 2)  # T H W C -> T C H W
        
        # 2. 类型转换：v2 transform 通常需要 Float 或 uint8 Tensor
        # 建议先转 float / 255.0 的操作如果不在这里做，
        # v2.RandAugment 也能处理 uint8，但为了数值稳定，建议保持和你原代码一致的 dtype
        # 这里假设输入已经是处理好的 float tensor 或者 uint8 tensor
        
        # 4. 直接应用变换 (极速！C++ 底层实现，无 Python 循环，无 PIL 转换)
        frames = self.aug_transform(frames)
        
        # 5. 还原维度
        frames = frames.permute(0, 2, 3, 1) # T C H W -> T H W C
        
        return frames

    def _apply_spatial_sampling(self, frames, spatial_idx, min_s, max_s, crop_s):
        # Setup relatives
        scl = self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE
        asp = self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
        
        rel_s = scl if (self.mode in ["train"] and len(scl) > 0) else None
        rel_a = asp if (self.mode in ["train"] and len(asp) > 0) else None
        
        return utils.spatial_sampling(
            frames,
            spatial_idx=spatial_idx,
            min_scale=min_s,
            max_scale=max_s,
            crop_size=crop_s,
            random_horizontal_flip=(self.cfg.DATA.RANDOM_FLIP if self.mode in ["train"] else False),
            inverse_uniform_sampling=(self.cfg.DATA.INV_UNIFORM_SAMPLE if self.mode in ["train"] else False),
            aspect_ratio=rel_a,
            scale=rel_s,
            motion_shift=(self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode in ["train"] else False),
        )

    def _apply_erase_and_mask(self, frames):
        """
        frames: Tensor with shape (C, T, H, W) based on previous context
        """
        # 1. 应用 Random Erasing
        if self.erase_transform is not None:
            # RandomErasing 通常设计用于处理 (N, C, H, W) 或 (C, H, W)
            # 对于视频，我们通常把 时间维度 T 当作 Batch 维度 N 来处理，
            # 这样可以对每帧独立进行 Erasing (或者根据 RandomErasing 实现细节保持一致)
            
            # (C, T, H, W) -> (T, C, H, W)
            frames = frames.permute(1, 0, 2, 3)
            
            # 直接调用预初始化的对象
            frames = self.erase_transform(frames)
            
            # (T, C, H, W) -> (C, T, H, W)
            frames = frames.permute(1, 0, 2, 3)

        # 2. 处理 Mask
        if self.cfg.AUG.GEN_MASK_LOADER:
            mask = self._gen_mask() # 假设这里返回的是 numpy array
            
            # 性能优化：立刻转为 Tensor，避免后续 collation 时再转
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float() # 或 .long() 根据需求
            
            # 逻辑修复：
            # 原代码 `frames + [torch.Tensor(), mask]` 意图可能是返回一个列表 [frames, dummy, mask]
            # 或者是想把 mask 拼接到 frames 上？
            # 假设你的 DataLoader collate_fn 期望的是分开的返回值：
            return frames, torch.empty(0), mask  # 返回 Tuple: (Frames, Dummy, Mask)
            
        # 如果没有 mask，只返回 frames (或者为了接口统一，返回 None)
        # 根据你原本的逻辑，这里似乎直接返回 frames 即可
        return frames
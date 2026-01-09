import decord
from decord import VideoReader, cpu
import torchvision as tv
from datasets.transform import GaussianBlurVideo, temporal_difference
from .thumos import *

@DATASET_REGISTRY.register()
class Surgery(torch.utils.data.Dataset):
    '''
    Video-based Dataset (on-the-fly loading mode)
    '''
    def __init__(self, cfg, split, num_retries=2):
        assert split in ['train', 'val', 'test'], f'Split "{split}" not supported for Surgery'
        self.split = split
        self.cfg = cfg

        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB 
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0

        self.dummy_output = None
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.split in ["train"]:
            self._num_clips = 1  
        elif self.split in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )  

        logger.info("Constructing Surgery {}...".format(split))
        self._construct_loader()
        self.samples = helper.load_video_samples_decord(
            cfg, self.sessions, self.video_root, self.target_root, split)
        self.aug = False  
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0  
        self.cur_epoch = 0

        if self.split == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:  
                self.rand_erase = True
        
    
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
        
        self.sessions = getattr(
            self.cfg.DATA,
            ("train" if self.split == "train" else "test").upper()
            + "_SESSION_SET",
        )

    def _load_raw_frames_decord(self, vr, indices, time_diff_prob, gaussian_prob):
        """
        Helper function to load frames using Decord and apply raw level augmentations
        (Time Difference / Gaussian Blur) to match original helper.load_frames logic.
        """
        # 1. Decord Loading
        # vr.get_batch returns (T, H, W, C) in uint8 (CPU)
        # We immediately convert to numpy to interface with Torch
        buffer = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(buffer).float() # (T, H, W, C)

        # 2. Raw Augmentations
        return helper.aug_raw_frames(time_diff_prob, gaussian_prob, frames)
        

    def __getitem__(self, index):
        # 0. Retrieve Pre-calculated Sample Info
        sample = self.samples[index] # VideoSampleInfo Object
        
        # 1. Initialize Video Reader (Efficient IO: Open once)
        try:
            # ctx=cpu(0) is best for Dataloader multiprocessing
            vr = VideoReader(sample.video_path, ctx=cpu(0))
        except Exception as e:
            # Fallback or strict error handling
            raise RuntimeError(f"Failed to open video {sample.video_path}: {e}")

        # 2. Setup Augmentation Params
        p_dt = self.p_convert_dt if self.split in ["train"] else 0.0
        p_gauss = 0.0
        if self.split in ["train"]:
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            
        elif self.split in ["val"]:
            spatial_sample_index = -2
            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE
            crop_size = self.cfg.DATA.TEST_CROP_SIZE
            
        elif self.split in ["test"]:
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
            raise NotImplementedError(f"Does not support {self.split} mode")

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
            long_indices, long_mask = sample.long_memory_indices
            long_key_padding_mask = torch.from_numpy(long_mask)

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
            if self.split in ["train"] else 1
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
                if self.split in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
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

            if self.split in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                 curr_work = transform.color_jitter_video_ssl(
                    curr_work,
                    bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                    hue=self.cfg.DATA.SSL_COLOR_HUE,
                    p_convert_gray=self.p_convert_gray,
                    moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                    gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                    gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                )

            if self.split in ["train"] and self.aug and self.cfg.AUG.AA_TYPE:
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
    
    def _apply_auto_augment(self, frames):
        # frames: T H W C
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        frames = frames.permute(0, 3, 1, 2) # T H W C -> T C H W
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1) # Back to T H W C
        return frames

    def _apply_spatial_sampling(self, frames, spatial_idx, min_s, max_s, crop_s):
        # Setup relatives
        scl = self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE
        asp = self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
        
        rel_s = scl if (self.split in ["train"] and len(scl) > 0) else None
        rel_a = asp if (self.split in ["train"] and len(asp) > 0) else None
        
        return utils.spatial_sampling(
            frames,
            spatial_idx=spatial_idx,
            min_scale=min_s,
            max_scale=max_s,
            crop_size=crop_s,
            random_horizontal_flip=(self.cfg.DATA.RANDOM_FLIP if self.split in ["train"] else False),
            inverse_uniform_sampling=(self.cfg.DATA.INV_UNIFORM_SAMPLE if self.split in ["train"] else False),
            aspect_ratio=rel_a,
            scale=rel_s,
            motion_shift=(self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.split in ["train"] else False),
        )

    def _apply_erase_and_mask(self, frames):
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            # C T H W -> T C H W for RandomErasing usually, or keeping as is?
            # Original: frames.permute(1, 0, 2, 3) -> Erase -> permute back
            # Original input was C T H W. 
            # Permute(1,0,2,3) makes it T C H W.
            frames = erase_transform(frames.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        
        if self.cfg.AUG.GEN_MASK_LOADER:
            mask = self._gen_mask()
            # Append mask channel? Original logic: frames + [Tensor, mask]
            # This looks like it returns a list? [frames_tensor, mask_tensor]
            return frames + [torch.Tensor(), mask] # Keeping original weird logic
            
        return frames
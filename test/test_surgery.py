import os
import sys
import unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from torch.utils.data.dataloader import DataLoader, Sampler

from src.datasets.surgery import Surgery, Thumos
from src.config.defaults import get_cfg, assert_and_infer_cfg

def visualize_comparison(tensor_old, tensor_new, save_dir="debug_vis", index=0):
    """
    Robust visualization that handles (T, H, W, C), (T, C, H, W), or (B, T, ...) inputs.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 0. Safety: Detach and move to CPU
    tensor_old = tensor_old.detach().cpu()
    tensor_new = tensor_new.detach().cpu()
    
    print(f"\n[Visual Debug] Input Shapes -> Old: {tensor_old.shape}, New: {tensor_new.shape}")

    # 1. Handle Batch Dimension: If 5D (B, T, ...), take the first sample
    if tensor_old.ndim == 5: 
        tensor_old = tensor_old[0]
        tensor_new = tensor_new[0]

    # 2. Standardize to (T, C, H, W)
    # Case A: (T, H, W, C) -> Permute to (T, C, H, W)
    if tensor_old.ndim == 4 and tensor_old.shape[-1] in [1, 3]:
        tensor_old = tensor_old.permute(0, 3, 1, 2)
        tensor_new = tensor_new.permute(0, 3, 1, 2)
    elif tensor_old.ndim == 4 and tensor_old.shape[0] in [1, 3]:
        tensor_old = tensor_old.permute(1, 0, 2, 3)
        tensor_new = tensor_new.permute(1, 0, 2, 3)
    
    
    # Case B: (T, H, W) Grayscale/Squeezed -> Unsqueeze to (T, 1, H, W)
    elif tensor_old.ndim == 3:
        print("[Visual Debug] Input is 3D (T, H, W), adding channel dim...")
        tensor_old = tensor_old.unsqueeze(1)
        tensor_new = tensor_new.unsqueeze(1)

    # 3. Select Frames to Visualize
    T = tensor_old.shape[0]
    indices_to_show = [0, T//2, min(T-1, T)]
    indices_to_show = sorted(list(set(indices_to_show))) # Remove duplicates if T is small

    for t in indices_to_show:
        img_old = tensor_old[t] # (C, H, W)
        img_new = tensor_new[t] # (C, H, W)

        # 4. Calculate Diff (Amplified for visibility)
        # Ensure dimensions match for subtraction
        if img_old.shape != img_new.shape:
            print(f"[Error] Frame {t} shape mismatch: {img_old.shape} vs {img_new.shape}")
            # Try to resize or crop strictly for viz (optional fallback)
            min_h = min(img_old.shape[1], img_new.shape[1])
            min_w = min(img_old.shape[2], img_new.shape[2])
            img_old = img_old[:, :min_h, :min_w]
            img_new = img_new[:, :min_h, :min_w]
        
        diff = torch.abs(img_old - img_new) * 5.0

        # 5. Stack and Save
        # make_grid expects a LIST of 3D tensors (C, H, W)
        try:
            grid = make_grid([img_old, img_new, diff], nrow=3, padding=2)
            save_image(grid, f"{save_dir}/sample_{index}_frame_{t}.png")
            print(f"Saved: {save_dir}/sample_{index}_frame_{t}.png")
        except Exception as e:
            print(f"[Error] make_grid failed for frame {t}. Shape: {img_old.shape}. Error: {e}")

def test_tensor_content_similarity(loader_old, loader_new, num_samples=3):
    print(f"\n{'Sample':<10} | {'MSE':<10} | {'PSNR (dB)':<10} | {'Result'}")
    print("-" * 50)

    iter_old = iter(loader_old)
    iter_new = iter(loader_new)

    for i in range(num_samples):
        try:
            batch_old = next(iter_old)
            batch_new = next(iter_new)
        except StopIteration:
            break
            
        # Unpack: Assuming format (frames, labels, ...)
        # Adjust index [0] if your loader returns something else
        frames_old = batch_old[1] 
        frames_new = batch_new[1]
        
        # Calculate Metrics
        mse = torch.mean((frames_old - frames_new) ** 2).item()
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
            
        status = "PASS" if psnr > 10.0 else "FAIL"
        print(f"{i:<10} | {mse:.6f}   | {psnr:.2f}       | {status}")
        
        # Always visualize failures, or first sample for sanity
        if status == "FAIL" or i == 0:
            print(f">> Visualizing sample {i}...")
            # Pass the WHOLE batch item (make_grid handles the unpacking inside)
            visualize_comparison(frames_old, frames_new, index=i)

# Call this in your test loop
# visualize_comparison(frames_old, frames_new)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
# Run the test
# test_tensor_content_similarity(old_loader, new_loader)

class ReverseSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Generate indices from len-1 down to 0
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self):
        return len(self.data_source)

class TestSurgeryDataset(unittest.TestCase):

    def setUp(self):
        """
        Set up the configuration and dataset for testing.
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file("configs/Surgery/MVITv2_S_16x4.yaml")
        self.cfg = assert_and_infer_cfg(self.cfg)
        self.cfg.DATA.TEST_SAMPLE_RATE = 2
        self.cfg.DATA.TRAIN_SESSION_SET = ["360"]
        self.cfg.DATA.TEST_SESSION_SET = ['360']
        self.cfg.MODEL.LONG_MEMORY_ENABLE = True
        self.cfg.AUG.ENABLE = True
        self.mode = "test"
        # self.mode = "train"
        self.dataset = Surgery(self.cfg, mode=self.mode)
        self.cfg.DATA.VIDEO_FOLDER = "frames"
        self.dataset1 = Thumos(self.cfg, mode=self.mode)
        self.loader = DataLoader(self.dataset, sampler=ReverseSampler(self.dataset))
        self.loader1= DataLoader(self.dataset1, sampler=ReverseSampler(self.dataset1))

    @unittest.skip("passed")
    def test_dataset_length(self):
        """
        Test if the dataset length is correct.
        """
        self.assertGreater(len(self.dataset), 0, "Dataset length should be greater than 0.")
        self.assertEqual(len(self.dataset), len(self.dataset1))

    @unittest.skip("passed")
    def test_sample_loading(self):
        """
        Test if a sample can be loaded correctly.
        """
        sample = self.dataset[0]
        sample_com = self.dataset1[0]
        self.assertIsInstance(sample, tuple, "Sample should be a tuple.")
        if not self.cfg.MODEL.LONG_MEMORY_ENABLE:
            self.assertEqual(len(sample), 2, "Sample should contain frames and labels.")
        else:
            self.assertEqual(len(sample), 4, "Sample shape is wrong")

        frames = sample[0]
        frames_ori = sample_com[0]
        self.assertIsInstance(frames, torch.Tensor, "Frames should be a torch.Tensor.")
        print(frames.shape) # C, T, H, W
        self.assertEqual(frames.shape[1], frames_ori.shape[1], "Number of frames should be matched.")

    # def test_normalization(self):
    #     """
    #     Test if the frames are normalized correctly.
    #     """
    #     sample = self.dataset[0]
    #     frames = sample[0]
    #     print(torch.max(frames), torch.min(frames))
    #     self.assertTrue(torch.all(frames >= -1.0) and torch.all(frames <= 1.0), "Frames should be normalized to [-1, 1].")

    def test_tensor_content_similarity(self):
        """
        Test if the tensor contents from the two datasets are close enough.
        """
        sample_surgery = self.dataset[1280]
        sample_thumos = self.dataset1[1280]

        label_surgery = sample_surgery[-1]
        label_thumos = sample_thumos[-1] 

        print(label_thumos.shape)
        self.assertTrue(
            np.allclose(label_surgery, label_thumos, atol=1e-5),
            "Tensor contents are not close enough between Surgery and Thumos datasets."
        )

        test_tensor_content_similarity(self.loader, self.loader1, 3000)

    def test_return_type(self):
        '''
        Test if the return types of each item are the same
        '''
        res = self.dataset[0]
        res1 = self.dataset1[0]

        for a,b in zip(res, res1):
            print(type(a), type(b))
            self.assertTrue(a.shape == b.shape)
    

if __name__ == "__main__":
    unittest.main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

if __name__ == '__main__':
    import os
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(ROOT)
    sys.path.append(os.path.join(os.path.dirname(ROOT), 'slowfast','slowfast'))
    print(sys.path[-2:])

import src.utils.logging as logging

from build import MODEL_REGISTRY
import stem_helper
from src.models.modules import SMViT, TransientExtractor, DualStreamMamba, DPPE_Head


logger = logging.get_logger(__name__)
@MODEL_REGISTRY.register()
class Pro2Mamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 参数获取
        embed_dim = cfg.MVIT.EMBED_DIM
        print('embed_dim:', embed_dim)
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        assert cfg.MVIT.SPATIAL.AGGREGATION.TYPE in ['meanP', 'cls_token']
        self.pool_type = cfg.MVIT.SPATIAL.AGGREGATION.TYPE # 'cls' or 'avg'
        self.enable_transient = getattr(cfg.MODEL, 'ENABLE_TRANSIENT', True)

        # 2. 模块初始化
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = (
            cfg.MVIT.PATCH_2D
        )
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,  # 3 7 7
            stride=cfg.MVIT.PATCH_STRIDE,  #
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch, 
        )  
        self.spatial_mvit = SMViT(cfg)
        
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
        if self.enable_transient:
            self.trans_extractor = TransientExtractor(cfg)
        else:
            self.trans_extractor = None
        self.tess_encoder = DualStreamMamba(cfg)
        self.head = DPPE_Head(cfg)
        
        self.norm = nn.LayerNorm(cfg.MVIT.TEMPORAL.EMBED_DIM)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def empty_cache(self):
        """清空所有推理状态"""
        self.trans_extractor.empty_cache()
        self.tess_encoder.empty_cache()

    def _extract_spatial(self, x):
        """
        Input: (B, C, T, H, W)
        Output: Tokens (B, T, N, D)
        """
        work_inputs, bcthw_work = self.patch_embed(x, keep_spatial=True)  
        B, C, T_work, H, W = list(bcthw_work) 
        work_inputs = einops.rearrange(work_inputs, "b c t h w -> (b t) (h w) c")  # torch.Size([64, 3136, 96])

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                (B * T_work), -1, -1  
            )  # Expand for each frames; 
            work_inputs = torch.cat((cls_tokens, work_inputs), dim=1)
        
        work_inputs, _ = self.spatial_mvit(work_inputs, bcthw_work) # bt hw c
        work_inputs = einops.rearrange(work_inputs, "(b t) hw c-> b t hw c", b=B, t=T_work) 
        return work_inputs

    def forward(self, x, labels=None):
        """
        x: (B, C, T, H, W)
        labels: (B, T) Optional, for Training Context (Teacher Forcing)
        """
        tokens = self._extract_spatial(x) # (B, T, N, D)
        
        # 1. Long-term Feature: Aggregation
        if self.pool_type == 'cls_token' and self.cls_embed_on:
            x_long = tokens[:, :, 0, :]
        else:
            # Avg Pool (Exclude CLS if exists)
            start_idx = 1 if self.cls_embed_on else 0
            x_long = tokens[:, :, start_idx:, :].mean(dim=2)
        x_long = self.norm(x_long)
        
        # 2. Transient Feature
        if self.enable_transient:
            x_trans = self.trans_extractor.forward_train(tokens)
            x_trans = self.norm(x_trans)
        else:
            # Pass Zeros to Transient Stream
            x_trans = torch.zeros_like(x_long)
        
        # 3. Mamba Temporal
        z = self.tess_encoder(x_long, x_trans) # (B, T, D)
        
        # 4. Head
        prev_probs = None
        if labels is not None:
            # Construct Shifted One-Hot Context
            one_hot = F.one_hot(labels, num_classes=self.head.num_classes).float()
            prev_probs = torch.roll(one_hot, shifts=1, dims=1)
            prev_probs[:, 0, :] = 0
            
        logits, dyn_z, p_base = self.head(z, prev_probs)
        
        return {
            "logits": logits,
            "dynamic_z": dyn_z,
            "prototypes": p_base
        }

    def stream_inference(self, x_chunk, prev_probs=None):
        """
        在线推理模式：输入一个 Chunk，但在内部串行处理。
        
        Args:
            x_chunk: (B, C, T, H, W) - 当前收到的视频片段
            prev_probs: (B, C) - 上一个 Chunk 结束时的预测概率 (用于 DPPE 上下文)
            
        Returns:
            chunk_logits: (B, T, C) - 当前 Chunk 每一帧的预测结果
            last_probs: (B, C) - 当前 Chunk 最后一帧的概率 (传给下一个 Chunk)
        """
        # B, C, T, H, W = x_chunk.shape
        
        # 1. 批量提取空间特征 (Batch Spatial Extraction)
        # 这样比在循环里做 embedding 更快，因为 PatchEmbed 是并行的
        tokens_seq = self._extract_spatial(x_chunk) # (B, T, N, D)
        chunk_logits_list = []
        
        # 2. 内部时间步循环 (Internal Step-by-Step Loop)
        for t in range(tokens_seq.shape[1]):
            # === Slice Current Frame ===
            x_curr_tokens = tokens_seq[:, t, :, :] # (B, N, D)
            
            # A. Long-term Feature
            if self.pool_type == 'cls_token' and self.cls_embed_on:
                x_long = x_curr_tokens[:, 0, :]
            else:
                start_idx = 1 if self.cls_embed_on else 0
                x_long = x_curr_tokens[:, start_idx:, :].mean(dim=1)
            x_long = self.norm(x_long) # (B, D)
            
            # B. Transient Feature (Updates Internal Buffer)
            if self.enable_transient:
                x_trans = self.trans_extractor.forward_inference(x_curr_tokens)
                x_trans = self.norm(x_trans)
            else:
                x_trans = torch.zeros_like(x_long)
            
            # C. Mamba Step (Updates Internal SSM State)
            # step 接受 (B, D)
            z_step = self.tess_encoder.step(x_long, x_trans) # (B, D)
            
            # D. Head (DPPE Evolution)
            # prev_probs 初始来自参数，后续来自上一步 loop 的输出
            logits_step, _, _ = self.head(z_step, prev_probs) # (B, C)
            
            chunk_logits_list.append(logits_step)
            
            # Update prev_probs for next step (t+1)
            prev_probs = F.softmax(logits_step, dim=-1)
            
        # 3. 堆叠结果
        chunk_logits = torch.stack(chunk_logits_list, dim=1) # (B, T, C)
        
        return chunk_logits, prev_probs

# Example Usage
if __name__ == "__main__":
    from src.config.defaults import get_cfg, assert_and_infer_cfg
    from losses import Pro2Loss
    cfg = get_cfg()
    cfg.merge_from_file("configs/Surgery/MVITv2_S_16x4_stream_dataset_test.yaml")
    cfg = assert_and_infer_cfg(cfg)
    # Mock Config
    # tmp = {
    #     'img_size': 224,
    #     'patch_size': 16,
    #     'embed_dim': 192,
    #     'num_classes': 7,
    #     'dilation_rates': [1, 6],
    #     'cls_embed_on': True,
    #     'pool_type': 'avg'
    # }
    
    T = 3
    t = T // (cfg.MODEL.CHUNK_SIZE // cfg.MODEL.CHUNK_SAMPLE_RATE)
    # cfg.update(tmp) 
    patch_stride = cfg.MVIT.PATCH_STRIDE 
    cfg.MVIT.SPATIAL.PATCH_DIMS_WORK = cfg.MVIT.SPATIAL.PATCH_DIMS_LONG = [cfg.MODEL.WORK_MEMORY_NUM_SAMPLES, 
                                                                           cfg.DATA.TRAIN_CROP_SIZE // patch_stride[1],
                                                                           cfg.DATA.TRAIN_CROP_SIZE // patch_stride[2]]
    model = Pro2Mamba(cfg)
    model.empty_cache()
    # Train
    dummy_input = torch.randn(2, 3, T, 224, 224) # B, C, T, H, W
    dummy_target = torch.randint(0, cfg.MODEL.NUM_CLASSES - 1, (2, t))
    out = model(dummy_input, dummy_target)
    print("Train Logits:", out['logits'].shape)
    
    # Loss
    loss_fn = Pro2Loss(cfg)
    loss, log = loss_fn(out, dummy_target)
    print("Loss:", loss.item())
    
    # Inference Step
    model.eval()
    model.empty_cache()
    frame_chunk = torch.randn(1, 3, 3, 224, 224)
    prev_prob = torch.zeros(1, cfg.MODEL.NUM_CLASSES)
    step_logits, _ = model.stream_inference(frame_chunk, prev_prob) # need to load 3 frames!
    print("Inference Logits:", step_logits.shape, step_logits)
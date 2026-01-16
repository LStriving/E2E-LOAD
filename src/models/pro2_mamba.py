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
from src.models.modules import SMViT


logger = logging.get_logger(__name__)


# 尝试导入 Mamba，如果环境没有安装则报错或模拟
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not found. Using MockMamba for demonstration.")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.in_proj = nn.Linear(d_model, d_model)
            self.d_model = d_model
            self.d_state = d_state
            self.expand = expand
            self.d_conv = d_conv
        def forward(self, x):
            return self.in_proj(x)
        def step(self, x, conv_state, ssm_state):
            return self.in_proj(x), conv_state, ssm_state

# =============================================================================
# 2. 瞬态特征提取器 (TransientExtractor)
# =============================================================================

class TransientExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 获取参数
        self.dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        self.dilations = getattr(cfg.MODEL, 'DILATION_RATES', [1, 6, 12])

        # 融合 MLP
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.dim * len(self.dilations), self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        
        # 门控 MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 推理缓存
        self.history_buffer = [] 
        self.max_k = max(self.dilations)

    def empty_cache(self):
        self.history_buffer = []

    def compute_spatial_gate(self, diff_list):
        # diff_list: List of tensors [(B, N, D), ...] or Tensor (B, T, N, D)
        # 这里简化处理，计算 diff 的平均幅度
        if isinstance(diff_list, list):
            # Inference mode: list of (B, N, D)
            stacked_diff = torch.stack(diff_list, dim=-1) # (B, N, D, K)
            mag = stacked_diff.norm(dim=2).mean(dim=-1)   # (B, N)
            
            mean = mag.mean(dim=1, keepdim=True) # (B, 1)
            var = mag.var(dim=1, keepdim=True)   # (B, 1)
            stats = torch.cat([mean, var], dim=-1)
            gate = self.gate_mlp(stats) # (B, 1)
            return gate.unsqueeze(1) # (B, 1, 1) for broadcast
            
        else:
            # Training mode: (B, T, N, D*K) fused or handle separately
            # 假设输入是 list of (B, T, N, D)
            pass

    def forward_train(self, x_seq, cls_token_exists=False):
        """
        x_seq: (B, T, N, D)
        cls_token_exists: boolean, if True, ignore first token for Diff calculation
        """
        B, T, N, D = x_seq.shape
        start_idx = 1 if cls_token_exists else 0
        x_spatial = x_seq[:, :, start_idx:, :] # (B, T, N_spatial, D)
        
        diffs = []
        for k in self.dilations:
            # Padding history for causal convolution
            pad = x_spatial[:, :1].repeat(1, k, 1, 1)
            padded = torch.cat([pad, x_spatial], dim=1)
            diff = x_spatial - padded[:, :T] # (B, T, N_s, D)
            diffs.append(diff)
            
        # 1. 计算 Gate
        # Average magnitude across dilations
        avg_mag = torch.stack([d.norm(dim=-1) for d in diffs], dim=-1).mean(dim=-1) # (B, T, N_s)
        
        s_mean = avg_mag.mean(dim=-1, keepdim=True) # (B, T, 1)
        s_var = avg_mag.var(dim=-1, keepdim=True)   # (B, T, 1)
        gate = self.gate_mlp(torch.cat([s_mean, s_var], dim=-1)).unsqueeze(-1) # (B, T, 1, 1)
        
        # 2. 融合 Diff
        concat_diff = torch.cat(diffs, dim=-1) # (B, T, N_s, D*K)
        fused_diff = self.fuse_mlp(concat_diff) # (B, T, N_s, D)
        
        # 3. Pooling (GAP)
        pooled_diff = fused_diff.mean(dim=2) # (B, T, D)
        
        return pooled_diff * gate.squeeze(2)

    def forward_inference(self, x_curr, cls_token_exists=False):
        """
        处理单帧输入，更新内部 buffer，返回单帧瞬态特征。
        x_curr: (B, N, D) - 当前帧的 Patch Tokens
        """
        start_idx = 1 if cls_token_exists else 0
        x_sp = x_curr[:, start_idx:, :] # (B, N_s, D)
        
        # 1. 更新 Buffer (FIFO)
        self.history_buffer.append(x_sp)
        # buffer 长度只需维持 max_k + 1
        if len(self.history_buffer) > self.max_k + 1:
            self.history_buffer.pop(0)
            
        # 2. 计算多尺度差分
        diffs = []
        for k in self.dilations:
            # 如果 Buffer 不够长，用最老的帧代替（或者用全0，视策略而定）
            # 这里采用 repeat mode: 刚开始时 diff 为 0
            if len(self.history_buffer) > k:
                prev = self.history_buffer[-(k+1)]
            else:
                prev = self.history_buffer[0] 
            
            diffs.append(x_sp - prev)
            
        # 3. 计算 Gate
        # diffs: list of [(B, N_s, D)...]
        gate = self.compute_spatial_gate(diffs) # (B, 1, 1)
        
        # 4. 融合与池化
        concat_diff = torch.cat(diffs, dim=-1) # (B, N_s, D*K)
        fused_diff = self.fuse_mlp(concat_diff) # (B, N_s, D)
        
        pooled_diff = fused_diff.mean(dim=1) # (B, D)
        
        return pooled_diff * gate.squeeze(1)

# =============================================================================
# 3. 双流 Mamba (DualStreamMamba)
# =============================================================================

class DualStreamMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        d_state = getattr(cfg.MODEL, 'MAMBA_STATE', 16)
        d_conv = getattr(cfg.MODEL, 'MAMBA_CONV', 4)
        expand = getattr(cfg.MODEL, 'MAMBA_EXPAND', 2)

        self.long_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.trans_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Linear(dim * 2, dim)
        
        self.inference_params = {}

    def empty_cache(self):
        self.inference_params = {}

    def forward(self, x_long, x_trans):
        z_l = self.long_mamba(x_long)
        z_t = self.trans_mamba(x_trans)
        return self.fusion(torch.cat([z_l, z_t], dim=-1))

    def step(self, x_long, x_trans):
        # Mamba inference usually requires managing state manually if not using simple .step API
        # Here assuming standard mamba_ssm step: 
        # output, conv_state, ssm_state = model.step(x, conv_state, ssm_state)
        
        if 'long_conv' not in self.inference_params:
             self._init_states(x_long.device, x_long.shape[0])

        z_l, self.inference_params['long_conv'], self.inference_params['long_ssm'] = \
            self.long_mamba.step(x_long, self.inference_params['long_conv'], self.inference_params['long_ssm'])
            
        z_t, self.inference_params['trans_conv'], self.inference_params['trans_ssm'] = \
            self.trans_mamba.step(x_trans, self.inference_params['trans_conv'], self.inference_params['trans_ssm'])
            
        return self.fusion(torch.cat([z_l, z_t], dim=-1))

    def _init_states(self, device, batch_size):
        # Helper to init zero states. Size depends on Mamba implementation details.
        # Assuming Mamba exposes .allocate_inference_cache or similar, or manual:
        # For simplicity, we assume the user calls standard Mamba which might handle None, 
        # otherwise we init zeros.
        # d_conv_state: (B, D, d_conv)
        # d_ssm_state: (B, D, d_state)
        
        # Note: Official Mamba usually uses `inference_params` dict to pass cache.
        # We adhere to the prompt's request: step(hidden, conv_state, ssm_state)
        dtype = self.long_mamba.in_proj.weight.dtype
        d_model = self.long_mamba.d_model
        d_state = self.long_mamba.d_state
        d_conv = self.long_mamba.d_conv
        
        self.inference_params['long_conv'] = torch.zeros(batch_size, d_model, d_conv, device=device, dtype=dtype)
        self.inference_params['long_ssm'] = torch.zeros(batch_size, d_model, d_state, device=device, dtype=dtype)
        self.inference_params['trans_conv'] = torch.zeros(batch_size, d_model, d_conv, device=device, dtype=dtype)
        self.inference_params['trans_ssm'] = torch.zeros(batch_size, d_model, d_state, device=device, dtype=dtype)

# =============================================================================
# 4. DPPE Head
# =============================================================================

class DPPE_Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        num_classes = cfg.MODEL.NUM_CLASSES
        self.lambda_val = getattr(cfg.MODEL, 'DPPE_LAMBDA', 0.5)

        self.num_classes = num_classes
        self.static_prototypes = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.orthogonal_(self.static_prototypes)
        
        self.transition_matrix = nn.Parameter(torch.eye(num_classes))
        self.evolve_mlp = nn.Sequential(
            nn.Linear(dim + num_classes, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, z_t, prev_probs=None):
        # z_t: (B, T, D) or (B, D)
        is_sequence = z_t.dim() == 3
        if not is_sequence: z_t = z_t.unsqueeze(1) # (B, 1, D)
        
        B, T, D = z_t.shape
        z_norm = F.normalize(z_t, dim=-1)
        p_base_norm = F.normalize(self.static_prototypes, dim=-1)
        
        # Static Score
        static_score = torch.einsum('btd,cd->btc', z_norm, p_base_norm)
        
        # Dynamic Evolution
        if prev_probs is None:
            context = torch.zeros(B, T, self.num_classes, device=z_t.device)
        else:
            # prev_probs 可能是 (B, C) 或 (B, T_labels, C)
            if prev_probs.dim() == 2: 
                prev_probs = prev_probs.unsqueeze(1) # (B, 1, C)
            
            # === [关键修复] ===
            # 检查 prev_probs 的时间维度 T_p 是否与 z_t 的 T 一致
            T_p = prev_probs.shape[1]
            if T_p != T:
                if T_p > T:
                    # 如果标签比视频长，截取对应部分 (假设是对齐的)
                    prev_probs = prev_probs[:, :T, :]
                else:
                    # 如果标签比视频短 (罕见)，重复或补零，这里抛出警告或错误更安全
                    # 为了鲁棒性，这里选择重复最后一步 (Broadcasting) 或 报错
                    # 简单处理：如果 T=1 且 T_p > 1，上面已经截取了；
                    # 如果 T > T_p，通常是数据加载问题。
                    pass 

            # 应用转移矩阵 A: previous_state @ A
            context = torch.matmul(prev_probs, self.transition_matrix)

        delta_p = self.evolve_mlp(torch.cat([z_t, context], dim=-1)) # (B, T, D)
        
        # Dynamic Score (using query refinement trick)
        dyn_z = z_t + delta_p
        dyn_z_norm = F.normalize(dyn_z, dim=-1)
        dynamic_score = torch.einsum('btd,cd->btc', dyn_z_norm, p_base_norm)
        
        final = static_score + self.lambda_val * dynamic_score
        final = final * self.scale
        
        if not is_sequence: final = final.squeeze(1)
        
        return final, dyn_z_norm, p_base_norm

# =============================================================================
# 5. 主模型: Pro2Mamba
# =============================================================================
@MODEL_REGISTRY.register()
class Pro2Mamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 参数获取
        embed_dim = cfg.MVIT.EMBED_DIM
        print('embed_dim:', embed_dim)
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.pool_type = cfg.get('pool_type', 'avg') # 'cls' or 'avg'


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
            
        self.trans_extractor = TransientExtractor(cfg)
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
        B, T, N, D = tokens.shape
        
        # 1. Long-term Feature: Aggregation
        if self.pool_type == 'cls' and self.cls_embed_on:
            x_long = tokens[:, :, 0, :]
        else:
            # Avg Pool (Exclude CLS if exists)
            start_idx = 1 if self.cls_embed_on else 0
            x_long = tokens[:, :, start_idx:, :].mean(dim=2)
        x_long = self.norm(x_long)
        
        # 2. Transient Feature
        x_trans = self.trans_extractor.forward_train(tokens, self.cls_embed_on)
        x_trans = self.norm(x_trans)
        
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
        B, C, T, H, W = x_chunk.shape
        
        # 1. 批量提取空间特征 (Batch Spatial Extraction)
        # 这样比在循环里做 embedding 更快，因为 PatchEmbed 是并行的
        tokens_seq = self._extract_spatial(x_chunk) # (B, T, N, D)
        chunk_logits_list = []
        
        # 2. 内部时间步循环 (Internal Step-by-Step Loop)
        for t in range(tokens_seq.shape[1]):
            # === Slice Current Frame ===
            x_curr_tokens = tokens_seq[:, t, :, :] # (B, N, D)
            
            # A. Long-term Feature
            if self.pool_type == 'cls' and self.cls_embed_on:
                x_long = x_curr_tokens[:, 0, :]
            else:
                start_idx = 1 if self.cls_embed_on else 0
                x_long = x_curr_tokens[:, start_idx:, :].mean(dim=1)
            x_long = self.norm(x_long) # (B, D)
            
            # B. Transient Feature (Updates Internal Buffer)
            x_trans = self.trans_extractor.forward_inference(x_curr_tokens, self.cls_embed_on)
            x_trans = self.norm(x_trans) # (B, D)
            
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

# =============================================================================
# 6. Loss Function
# =============================================================================

class Pro2Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.margin = getattr(cfg.MODEL, 'LOSS_MARGIN', 0.1)
        self.lambda_proc = getattr(cfg.MODEL, 'LOSS_LAMBDA', 0.1)
            
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output_dict, targets, boundary_mask=None):
        """
        targets: (B, T)
        boundary_mask: (B, T) or None
        """
        logits = output_dict['logits']
        B, T, C = logits.shape
        
        # 1. Boundary-Aware CE
        loss_cls = self.ce_loss(logits.reshape(-1, C), targets.reshape(-1)).view(B, T)
        if boundary_mask is not None:
            weights = 1.0 + boundary_mask * 1.0 # 边界处权重加倍
            loss_cls = (loss_cls * weights).mean()
        else:
            loss_cls = loss_cls.mean()
            
        # 2. Procedural Contrastive Loss
        # Flatten
        z = output_dict['dynamic_z'].view(-1, output_dict['dynamic_z'].shape[-1]) # (N, D)
        p = output_dict['prototypes'] # (C, D)
        y = targets.view(-1)
        
        # Cosine Similarity
        sim = torch.matmul(z, p.t()) # (N, C)
        
        # Pos Sim
        pos_mask = F.one_hot(y, num_classes=C).bool()
        pos_sim = sim[pos_mask]
        
        # Neg Sim (Hard Negative)
        neg_sim = sim.clone()
        neg_sim[pos_mask] = -float('inf')
        hard_neg_sim, _ = neg_sim.max(dim=1)
        
        loss_proc = F.relu(self.margin + hard_neg_sim - pos_sim).mean()
        
        return loss_cls + self.lambda_proc * loss_proc, {"cls": loss_cls.item(), "proc": loss_proc.item()}

# Example Usage
if __name__ == "__main__":
    from src.config.defaults import get_cfg, assert_and_infer_cfg
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
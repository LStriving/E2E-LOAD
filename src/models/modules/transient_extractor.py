import torch
import torch.nn as nn

class TransientExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 获取参数
        self.dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        self.dilations = getattr(cfg.MODEL, 'DILATION_RATES', [1, 6, 12])
        self.gating_mode = getattr(cfg.MODEL, 'TRANSIENT_GATING', 'spatial_stat')

        # Fusion Layer
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.dim * len(self.dilations), self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        
        # Gating Network (Conditional Init)
        self.gate_mlp = None
        if self.gating_mode == 'spatial_stat':
            # Input: Mean + Var = 2 dim
            self.gate_mlp = nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )
        elif self.gating_mode == 'mlp':
            # Input: Feature Dim (Magnitude)
            self.gate_mlp = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )

        self.history_buffer = [] 
        self.max_k = max(self.dilations) if self.dilations else 0

    def empty_cache(self):
        self.history_buffer = []

    def compute_gate(self, diff_list_or_tensor, mode='train'):
        """
        diff_list: List of tensors [(B, N, D), ...] or Tensor (B, T, N, D)
        """
        if self.gating_mode == 'none':
            if mode == 'train': 
                B, T, N, _ = diff_list_or_tensor.shape # approximate
                return torch.ones(B, T, 1, 1, device=diff_list_or_tensor.device)
            else:
                B, N, _ = diff_list_or_tensor[0].shape
                return torch.ones(B, 1, 1, device=diff_list_or_tensor[0].device)

        # Logic for 'spatial_stat' (Proposed)
        if self.gating_mode == 'spatial_stat':
            if mode == 'train':
                # Tensor: (B, T, N, D*K) -> norm -> (B, T, N) -> mean/var
                # Simplified: assume input is pre-calculated mag or handle raw
                # TODO
                pass # Logic is inside forward for efficiency
            else:
                # Inference: list of (B, N, D)
                stacked = torch.stack(diff_list_or_tensor, dim=-1) # (B, N, D, K)
                mag = stacked.norm(dim=2).mean(dim=-1) # (B, N)
                stats = torch.cat([mag.mean(dim=1, keepdim=True), mag.var(dim=1, keepdim=True)], dim=-1)
                return self.gate_mlp(stats).unsqueeze(1) # (B, 1, 1)

        # Logic for 'mlp' (Baseline)
        if self.gating_mode == 'mlp':
            # Simply use magnitude of the fused difference
            pass # Implemented in forward inline
        return 1.0

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
            # 逻辑：diff[t] = x[t] - x[t-k]
            # 当 t < k 时，我们假设 x[t-k] = x[0] (Padding 逻辑)
            
            # 1. 正常部分 (t >= k): 使用切片，零拷贝
            # x_spatial[:, k:, ...]  减去  x_spatial[:, :-k, ...]
            diff_main = x_spatial[:, k:, :] - x_spatial[:, :-k, :] 
            
            # 2. 头部部分 (t < k): 需要减去 x[0]
            # x_spatial[:, :k, ...] 减去 x_spatial[:, 0:1, ...] (广播)
            diff_head = x_spatial[:, :k, :] - x_spatial[:, 0:1, :]
            
            # 3. 拼接 (只在 T 维度拼接一次，比 cat(repeat) 快)
            diff = torch.cat([diff_head, diff_main], dim=1)
            diffs.append(diff)
            
        # 1. Gating Ablation
        gate = 1.0
        if self.gating_mode != 'none':
            avg_mag = torch.stack([d.norm(dim=-1) for d in diffs], dim=-1).mean(dim=-1) # (B, T, N)
            
            if self.gating_mode == 'spatial_stat':
                s_mean = avg_mag.mean(dim=-1, keepdim=True)
                s_var = avg_mag.var(dim=-1, keepdim=True)
                gate = self.gate_mlp(torch.cat([s_mean, s_var], dim=-1)).unsqueeze(-1)
            
            elif self.gating_mode == 'mlp':
                # Global Average Magnitude -> MLP
                global_mag = avg_mag.mean(dim=-1, keepdim=True) # (B, T, 1)
                gate = self.gate_mlp(global_mag).unsqueeze(-1)

        # 2. Fusion
        fused = self.fuse_mlp(torch.cat(diffs, dim=-1))
        return fused.mean(dim=2) * gate.squeeze(2)

    def forward_inference(self, x_curr, cls_token_exists=False):
        """
        处理单帧输入，更新内部 buffer，返回单帧瞬态特征。
        x_curr: (B, N, D) - 当前帧的 Patch Tokens
        """
        start_idx = 1 if cls_token_exists else 0
        x_sp = x_curr[:, start_idx:, :]
        
        self.history_buffer.append(x_sp)
        if len(self.history_buffer) > self.max_k + 1: self.history_buffer.pop(0)
        
        diffs = []
        for k in self.dilations:
            prev = self.history_buffer[-(k+1)] if len(self.history_buffer) > k else self.history_buffer[0]
            diffs.append(x_sp - prev)
            
        gate = 1.0
        if self.gating_mode == 'spatial_stat':
            stacked = torch.stack(diffs, dim=-1)
            mag = stacked.norm(dim=2).mean(dim=-1)
            stats = torch.cat([mag.mean(dim=1, keepdim=True), mag.var(dim=1, keepdim=True)], dim=-1)
            gate = self.gate_mlp(stats).unsqueeze(1)
        
        fused = self.fuse_mlp(torch.cat(diffs, dim=-1))
        return fused.mean(dim=1) * gate.squeeze(1)
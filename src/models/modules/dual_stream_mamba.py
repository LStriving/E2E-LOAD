import torch
import torch.nn as nn

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

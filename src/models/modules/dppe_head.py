import torch
import torch.nn as nn
import torch.nn.functional as F

class DPPE_Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dim = cfg.MVIT.TEMPORAL.EMBED_DIM
        num_classes = cfg.MODEL.NUM_CLASSES
        self.lambda_val = getattr(cfg.MODEL, 'DPPE_LAMBDA', 0.5)
        # Ablation C: Head Type ('static', 'dynamic', 'dual')
        self.head_mode = cfg.get('head_mode', 'dual')
        assert self.head_mode in ['static', 'dynamic', 'dual']
        # Ablation C: Prior Type ('none', 'learnable')
        self.prior_mode = cfg.get('prior_mode', 'learnable')
        assert self.prior_mode in [None, 'none', 'None', 'learnable']

        self.num_classes = num_classes
        # Static Prototypes (Always needed for 'static' and 'dual')
        self.static_prototypes = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.orthogonal_(self.static_prototypes)
        
        # Dynamic Components (Only init if needed)
        if self.head_mode in ['dynamic', 'dual']:
            self.transition_matrix = nn.Parameter(torch.eye(self.num_classes)) 
            self.evolve_mlp = nn.Sequential(
                nn.Linear(dim + self.num_classes, dim),
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
        static_score = torch.einsum('b...d,cd->b...c', z_norm, p_base_norm)
        
        if self.head_mode == 'static':
            return static_score * self.scale, z_norm, p_base_norm

        # 2. Dynamic Evolution
        B = z_t.shape[0]
        T = z_t.shape[1] if z_t.dim() == 3 else 1
        
        # Prior Context Ablation
        if prev_probs is None or self.prior_mode == 'none':
            context = torch.zeros(B, T, self.num_classes, device=z_t.device)
        else:
            # Ensure shape alignment (handling the T mismatch error)
            if prev_probs.dim() == 2: prev_probs = prev_probs.unsqueeze(1)
            if prev_probs.shape[1] != T: prev_probs = prev_probs[:, :T, :] # Slice
            
            if self.prior_mode == 'learnable':
                context = torch.matmul(prev_probs, self.transition_matrix)
            else:
                context = prev_probs # Identity prior

        delta_p = self.evolve_mlp(torch.cat([z_t, context], dim=-1))
        dyn_z = z_t + delta_p
        dyn_z_norm = F.normalize(dyn_z, dim=-1)
        dynamic_score = torch.einsum('b...d,cd->b...c', dyn_z_norm, p_base_norm)
        
        # 3. Dual Path Ablation
        if self.head_mode == 'dynamic':
            final = dynamic_score
        else: # 'dual'
            final = static_score + self.lambda_val * dynamic_score
        if not is_sequence: final = final.squeeze(1)
        return final * self.scale, dyn_z_norm, p_base_norm

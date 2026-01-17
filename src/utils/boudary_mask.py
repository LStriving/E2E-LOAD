import torch
import torch.nn.functional as F

def generate_boundary_mask(targets, boundary_width=3):
    """
    生成边界掩码，用于 Boundary-Aware Loss。
    
    Args:
        targets (torch.Tensor): Ground Truth 标签，形状 (B, T)
        boundary_width (int): 边界窗口半径。例如 width=2，则边界前后各2帧都会被标记。
                              总窗口大小 = 2 * width + 1。等于0时，只有边界会被标记
    
    Returns:
        torch.Tensor: Mask Tensor，形状 (B, T)，边界区域为 1.0，非边界为 0.0
    """
    # 1. 确保 targets 是 Long 类型
    if targets.dim() == 3:
        targets = targets.argmax(dim=-1)
    targets = targets.long()
    B, T = targets.shape
    
    # 2. 计算相邻帧差异 (B, T-1)
    # 如果 targets[t] != targets[t+1]，则 diff[t] 为 True
    diff = targets[:, 1:] != targets[:, :-1]
    
    # 3. 初始化 Mask (B, T)
    mask = torch.zeros_like(targets, dtype=torch.float)
    
    # 4. 标记边界点
    # 将发生变化的两个相邻帧都标记为边界 (t 和 t+1)
    mask[:, 1:][diff] = 1.0
    mask[:, :-1][diff] = 1.0
    
    # 5. 如果需要更宽的窗口，进行膨胀 (Dilation) 操作
    if boundary_width > 0:
        # 增加 Channel 维度以适配 max_pool1d: (B, 1, T)
        mask = mask.unsqueeze(1)
        
        # Kernel Size = 2 * width + 1保证覆盖左右
        kernel_size = 2 * boundary_width + 1
        
        # 使用 MaxPool1d 进行形态学膨胀 (Dilation)
        # stride=1, padding=width 保证输出长度不变且中心对齐
        mask = F.max_pool1d(mask, kernel_size=kernel_size, stride=1, padding=boundary_width)
        
        # 恢复形状 (B, T)
        mask = mask.squeeze(1)
        
    return mask



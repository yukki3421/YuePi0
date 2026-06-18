import torch
class AdaptiveRMSNorm:
    pass

def build_blockwise_causal_mask(
    num_vlm_tokens: int, 
    num_proprio_tokens: int, 
    num_action_tokens: int,
    batch_size: int, 
    dtype = torch.float32,
    device = None
) -> torch.Tensor:
    '''返回(B, 1, T, T)的attention mask, 可见处为0, 屏蔽处为 -inf'''
    T_total = num_vlm_tokens + num_proprio_tokens + num_action_tokens
    mask_pre = torch.zeros(T_total, T_total, dtype=dtype)
    mask_pre[:num_vlm_tokens, num_vlm_tokens:] = torch.finfo(dtype).min
    # proprio屏蔽action
    mask_pre[num_vlm_tokens:num_vlm_tokens+num_proprio_tokens, num_vlm_tokens+num_proprio_tokens:] = torch.finfo(dtype).min
    # action内部加causal   
    A = num_action_tokens
    action_start = num_vlm_tokens + num_proprio_tokens
    action_sub = torch.zeros(A, A, dtype=dtype)
    upper = torch.triu(torch.ones(A, A), diagonal=1).bool()
    action_sub.masked_fill_(upper, torch.finfo(dtype).min)
    mask_pre[action_start:,action_start:] = action_sub
    # 拓展
    mask = mask_pre[None, None, :, :].expand(batch_size, 1, T_total, T_total)
    return mask.to(device)
import torch
# 用于实现配对旋转, 对半分的配对旋转方法
def rotate_half(x):
    dim = x.shape[-1]
    x_2 = x[...,  dim//2:] # python的省略号切片, 就是前面所有维度全取
    x_1 = x[..., :dim//2]
    return torch.cat((-x_2, x_1), dim=-1)

# 输出旋转后的q, k
# 输入q, k 的形状: (B， H, T, D)
# cos，sin的形状：（B, T， D)
def apply_rotary_pos_emb(q, k, cos, sin):
    # 增加一个H维度
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    # 旋转公式
    q_rot = q*cos + rotate_half(q)*sin
    k_rot = k*cos + rotate_half(k)*sin
    return q_rot, k_rot
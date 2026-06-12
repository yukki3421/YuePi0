import torch
import pytest
from model.vla.grouped_query_attention import GroupedQAttention
from model.vla.kvcache import KVCache

def test_kv_cache():
    torch.manual_seed(0)
    B, T, D = 2, 8, 512
    num_heads = 32
    num_kv_heads = 8
    head_dim = 16
    rope_theta = 10000

    gqa = GroupedQAttention(D, num_heads, num_kv_heads, head_dim, rope_theta, 0)
    gqa.eval()

    x_full = torch.randn(B, T, D)
    pos_full = torch.arange(T).unsqueeze(0).expand(B, T)
    mask_full = torch.triu(torch.ones(T, T), diagonal=1) * (-1e9)

    # 方式一：一次性forward
    with torch.no_grad():
        y_full, _ = gqa(x_full, mask_full, pos_full)
    
    # 方式二：分两步用kv cache
    T1 = 5
    cache = KVCache()

    x1 = x_full[:, :T1]
    pos1 = pos_full[:, :T1]
    mask1 = mask_full[:T1, :T1]

    x2 = x_full[:, T1:]
    pos2 = pos_full[:, T1:]
    mask2 = mask_full[T1:, :] #注意这个mask！！应该为（3， 8）而不是(3, 3)

    with torch.no_grad():
        y1, _ = gqa(x1, mask1, pos1, cache)
        y2, _ = gqa(x2, mask2, pos2, cache)
    
    y_cached = torch.cat([y1, y2], dim=1)
    diff = (y_full - y_cached).abs().max().item()
    assert torch.allclose(y_full, y_cached, atol=1e-5), \
    f"KV Cache 输出不一致, max_diff={diff:.2e}"

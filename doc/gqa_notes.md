# Day 2 - GQA Attention 复现笔记

## 1. GQA 是什么

**Grouped Query Attention**：让 Q head 数量多于 KV head 数量，多个 Q heads 共享一组 KV。

| | MHA | MQA | GQA |
|---|---|---|---|
| Q heads | H | H | H |
| KV heads | H | 1 | G |
| KV 显存 | O(H·T) | O(T) | O(G·T) |
| 表达能力 | 最强 | 最弱 | 中等 |

pi-zero 用的是 **H=32, G=8**，即 32 个 Q heads，8 个 KV heads，每 4 个 Q 共享 1 套 KV。

## 2. num_kv_groups 的计算

```python
self.num_kv_groups = num_heads // num_kv_heads  # 必须是整除，用 //
```

`//` 得到整数（4），`/` 得到 float（4.0），后者会导致后续 reshape 形状出错。

## 3. repeat_kv 的原理

把 (B, H_KV, T, D_h) 扩展成 (B, H_Q, T, D_h)：

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 在 head 维插入 size-1，然后用 expand 广播，最后 reshape 合并
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_kv_heads * n_rep, slen, head_dim
    )
```

`n_rep = num_kv_groups = num_heads // num_kv_heads`

**注意**：`torch.repeat_interleave` 是另一种 API，但这里是 `expand + reshape`，效果相同。

## 4. GQA forward 流程

```
hidden_states (B, T, D)
  ↓ q_proj  → (B, T, num_heads * head_dim)
  ↓ k_proj  → (B, T, num_kv_heads * head_dim)
  ↓ v_proj  → (B, T, num_kv_heads * head_dim)
  ↓ view + transpose → Q: (B, num_heads, T, head_dim)
                     K/V: (B, num_kv_heads, T, head_dim)
  ↓ RoPE
  ↓ repeat_kv → K/V: (B, num_heads, T, head_dim)
  ↓ Q @ K^T / sqrt(head_dim) → (B, num_heads, T, T)
  ↓ + mask
  ↓ softmax(dim=-1, dtype=hidden_states.dtype)
  ↓ @ V → (B, num_heads, T, head_dim)
  ↓ transpose + view → (B, T, hidden_size)
  ↓ o_proj → (B, T, hidden_size)
```

## 5. softmax 的 dtype 问题（踩坑）

```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=hidden_states.dtype)
```

**必须指定 `dtype=hidden_states.dtype`**。

原因：bf16 下 softmax 累加精度差很多，原版用的是 `dtype=hidden_states.dtype`。漏掉这步会导致 allclose 失败。

## 6. Linear 层必须 bias=False（踩坑）

原版 GemmaAttention 的 Q/K/V/O 四个线性层全部 `bias=False`：

```python
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
```

如果 bias=True，你的 state_dict key 会多出 `.bias`，`load_state_dict` 时 key 对不上，权重根本没拷进去，输出自然全错。

## 7. 为什么 attention 要除以 √head_dim

Q 和 K 的向量维度是 `head_dim`，不是 `hidden_size`。

当 `head_dim` 大时（128），点积值方差大，softmax 后梯度接近 0（饱和）。除以 `√head_dim` 让方差稳定。

## 8. verify 的关键细节

1. **必须 `load_state_dict(strict=False)`**：因为自定义类和原版类的 `state_dict` keys 可能不完全一致（rotary_emb 等子模块不同），strict=True 会报错
2. **mask 用 `-1e9` 而不是 `1e-9`**：后者是正数，加上去会让不该 mask 的位置值变大；前者才是 -inf
3. **FakeConfig 逐字段赋值**：class body 内无法引用外部局部变量，必须先定义空 class，再逐个赋值

## 9. 自检 Q&A

> **Q1: 为什么 GQA 比 MQA 显存大但表达能力更强？**
> MQA 只有 1 个 KV head，所有 Q 共享同一套 K/V，特征太单一。GQA 有 G 个 KV head（8 个），每个子空间有不同的 key 表达，特征更丰富。

> **Q2: `expand` 和 `repeat` 的区别？**
> `expand` 不复制数据，只在指定维度广播（维度大小必须是 1 或相同）；`repeat` 实际复制数据。repeat_kv 用 expand+reshape 是为了避免实际复制，利用广播机制省显存。

> **Q3: GQA 的 O _proj 输入维度是 `num_heads * head_dim` 而不是 `hidden_size`？**
> 两者数值相等（num_heads * head_dim = hidden_size），但语义不同。o_proj 从 attention 输出（已经是 head 维度展开后的形式）映射回 hidden_size。

> **Q4: KV Cache 的核心思想是什么？**
> 把之前 token 的 K/V 存起来，只算新 token 的 K/V，然后拼接。这样 attention 复杂度从 O(T²) 变成 O(T)（第一步）和 O(1)（后续步）。推理速度大大加快。

## 10. 下一步

Day 3：**KV Cache** —— 实现 `MyKVCache`，验证两步生成与一次性生成输出一致。
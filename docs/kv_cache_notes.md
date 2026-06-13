# Day 3 - KV Cache 复现笔记

## 1. KV Cache 是什么

**一句话**：把已经算过的 token 的 K 和 V 存起来，下次生成新 token 时直接复用，不用重新算。

只在 **推理（autoregressive generation）** 时使用，**训练时不用**。

## 2. 训练 vs 推理的计算模式

| | 训练 | 推理（无 cache） | 推理（有 cache） |
|---|---|---|---|
| 输入方式 | 整个序列一次性 | 每次输入 1 个新 token | 每次输入 1 个新 token |
| K/V 计算 | 一次算完 T 个 | 每步重新算前 0..t 个 | 每步只算第 t 个，cat 到旧的 |
| 复杂度 | O(T²) 一次 | O(T²) × T = O(T³) | O(T) × T = O(T²) |

推理为什么慢？因为生成是**串行**的（要等上一个 token 才能算下一个），无 cache 时每步都重复计算前面所有 token 的 K/V。

## 3. KVCache 数据结构

```python
class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []    # 每个元素是一层的 K
        self.value_cache: List[torch.Tensor] = []  # 每个元素是一层的 V
```

**为什么是 List？** 因为模型有多层 decoder（pi-zero 有 18 层），每层的 K/V 不一样，要分开存：

```
key_cache[0] = layer 0 的 K, shape (B, H_KV, T, D_h)
key_cache[1] = layer 1 的 K, shape (B, H_KV, T, D_h)
...
key_cache[17] = layer 17 的 K
```

`layer_idx` 参数就是告诉 cache："我是第几层"。

## 4. update 方法的两个分支

```python
def update(self, key_states, value_states, layer_idx):
    if len(self.key_cache) <= layer_idx:
        # 分支 1：第一次（prefill 阶段）
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    else:
        # 分支 2：后续（decode 阶段）
        # 在 seq_len 维（dim=-2）拼接旧的和新的
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], key_states], dim=-2
        )
        self.value_cache[layer_idx] = torch.cat(
            [self.value_cache[layer_idx], value_states], dim=-2
        )
    return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

**分支 1** 用 `append`，因为 `cat` 一个空 list 会报错。
**分支 2** 在 `dim=-2`（seq_len 维度）拼接。

## 5. attention 里集成 cache 的位置

```python
# RoPE 之后
cos, sin = self.rotary_emb(query, position_ids)
query, key = apply_rotary_pos_emb(query, key, cos, sin)

# ===== KV Cache =====
if kv_cache is not None:
    key, value = kv_cache.update(key, value, self.layer_idx)
# =====================

# repeat_kv 之前
key = repeat_kv(key, self.num_kv_groups)
value = repeat_kv(value, self.num_kv_groups)
```

**关键：cache 必须在 RoPE 之后、repeat_kv 之前**

| 时机 | 原因 |
|---|---|
| RoPE 之后 | cache 存的应该是带位置编码的 K，下次拼接才正确（不然要重新旋转） |
| repeat_kv 之前 | cache 只存 H_KV 份就够了，省显存（用的时候再扩展） |

## 6. 推理两阶段（prefill + decode）

```
prefill 阶段：把 prompt 一次性喂进去（建立 cache）
  输入: x_prompt (B, T_prompt, D)
  cache: 从空 → 存了 T_prompt 个 token

decode 阶段：每次喂 1 个 token，反复调用
  输入: x_new (B, 1, D)
  cache: T_prompt → T_prompt + 1 → T_prompt + 2 → ...
```

测试时把 `T1=5` 当 prefill，把后 3 个当 decode（一次喂多个，简化代码）。

## 7. mask 的形状（最大踩坑点）

测试时三种 mask：

| 阶段 | Q 长度 | K 长度 | mask 形状 | 切片方法 |
|---|---|---|---|---|
| 一次性（baseline） | 8 | 8 | (8, 8) | `mask_full` |
| step 1（prefill） | 5 | 5 | (5, 5) | `mask_full[:5, :5]` |
| step 2（decode） | 3 | 8 | (3, 8) | `mask_full[5:, :]` |

**为什么 step 2 mask 是 (3, 8) 而不是 (3, 3)？**
- Q 只有 3 个新 token（位置 5、6、7）
- K 有 **完整 8 个**（cache 里 5 个 + 新的 3 个）
- attention 矩阵 = `Q @ K^T = (3, head_dim) @ (head_dim, 8) = (3, 8)`
- 所以 mask 必须是 (3, 8)

它正好是完整 (8, 8) 因果 mask 的 **最后 3 行**：

```
row 5: [0, 0, 0, 0, 0, 0,    -inf, -inf]   ← 位置 5 看 0..5
row 6: [0, 0, 0, 0, 0, 0,    0,    -inf]   ← 位置 6 看 0..6
row 7: [0, 0, 0, 0, 0, 0,    0,    0   ]   ← 位置 7 看 0..7
```

## 8. mask 必须用 -inf（再次踩坑）

```python
# 错（mask 不够强，softmax 还是会"轻微看"未来）
mask = torch.triu(torch.ones(T, T), diagonal=1) * (-1)

# 对（接近 -inf，softmax 后变 0）
mask = torch.triu(torch.ones(T, T), diagonal=1) * (-1e9)
```

`-1` 不是有效 mask！只让 attention 稍微减弱，没真正屏蔽。导致 prefill 阶段的 softmax 分母不同，结果就和 baseline 对不上。

## 9. 验证方法

模拟两步推理 vs 一次性推理：

```python
# 方式 1：一次性 forward（baseline）
y_full = gqa(x_full, mask_full, pos_full)

# 方式 2：分两步用 cache
cache = KVCache()
y1 = gqa(x_full[:, :5], mask_full[:5, :5], pos_full[:, :5], cache)
y2 = gqa(x_full[:, 5:], mask_full[5:, :], pos_full[:, 5:], cache)
y_cached = torch.cat([y1, y2], dim=1)

# 必须 allclose
assert torch.allclose(y_full, y_cached, atol=1e-5)
```

## 10. 自检 Q&A

> **Q1: KV Cache 为什么能加速推理？**
> 不用重复计算前面所有 token 的 K/V，每步只算 1 个新 token。复杂度从 O(T³) 降到 O(T²)。

> **Q2: 为什么训练不用 KV Cache？**
> 训练时整个序列一次性输入，**并行**计算所有 token 的 attention。如果用 cache，反而退化成串行。

> **Q3: cache 在 RoPE 之前还是之后存？**
> **之后**。cache 存的应该是已经带位置编码的 K，否则下次取出来还要重新旋转，错位且浪费。

> **Q4: cache 在 repeat_kv 之前还是之后存？**
> **之前**。cache 只存 H_KV 份就够了（少存 G 倍数据），用的时候再 repeat_kv 扩展。

> **Q5: 为什么 cache 是 List 而不是 Dict？**
> layer_idx 是连续整数 0..N-1，List 索引最简洁。Dict 也行但是多余。

> **Q6: prefill 阶段 mask 是 (T, T)，decode 阶段 mask 是 (T_new, T_total)，为什么？**
> attention 矩阵 = Q @ K^T，shape 是 `(T_q, T_k)`。decode 时 Q 只有新的几个，K 有 cache 里所有 token，所以 T_q ≠ T_k。

> **Q7: 为什么 mask 用 -1 不行？**
> mask 的目的是让 softmax 输出该位置为 0，需要加 -inf（实际用 -1e9 近似）。-1 只让 logit 减 1，softmax 后还是有非零值，等价于"轻微关注未来 token"，属于信息泄漏。

> **Q8: 多层 attention 共享同一个 KVCache 实例吗？**
> 是。整个模型 forward 时所有 layer 共用一个 KVCache 实例，靠 `layer_idx` 区分存到哪个 list 元素。

## 11. 下一步

Day 4：**SigLIP Patch Embedding + Encoder** —— 视觉编码器，把图片切 patch 后用 ViT 编码。
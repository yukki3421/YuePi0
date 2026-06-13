# Day 1 - RoPE 复现笔记

## 1. RoPE 是什么

一句话：**把 Q 和 K 向量在二维平面内旋转，角度由位置决定，让 attention 点积天然只依赖相对位置。**

## 2. 三种位置编码的演进

| 方法 | 怎么做 | 缺点 |
|---|---|---|
| 绝对位置编码 | 每个位置学一个固定向量，加到 embedding 上 | 外推差：训练 512，测试 1024 崩 |
| 可学习位置编码 | 每个位置一个可学习向量 | 同上，长度写死 |
| **RoPE** | 旋转 Q 和 K，点积自带相对位置信息 | 几乎无缺点，成了事实标准 |

## 3. RoPE 旋转的数学

### 3.1 二维旋转

向量 $[x_1, x_2]$ 旋转角度 $\theta$ 后：

$$
[x_1\cos\theta - x_2\sin\theta,\quad x_1\sin\theta + x_2\cos\theta]
$$

两个旋转后的向量做点积，只依赖相对角度 $(\alpha - \beta)$，与绝对角度无关。

### 3.2 RoPE 的做法：每对维度一个二维平面

D 维向量分成 D/2 对，每对独立旋转：

```
维度 (0, 32) 配对 → 角度 θ0
维度 (1, 33) 配对 → 角度 θ1
维度 (2, 34) 配对 → 角度 θ2
...
维度 (31, 63) 配对 → 角度 θ31
```

不同对用不同频率：高频管局部细节，低频管远距离感知。

## 4. inv_freq 公式

$$
\text{inv\_freq}_i = \frac{1}{\text{base}^{2i/\dim}}, \quad i = 0, 1, ..., \dim/2-1
$$

注意公式里有 **2i/dim**，不是 i/dim。**最容易写错这里。**

## 5. 为什么 cos/sin 要分开两步算

RoPE 实现分两个组件：

| 组件 | 职责 | 输出形状 |
|---|---|---|
| `RotaryEmbedding.forward` | 把位置 id 变成 cos/sin | cos/sin: `(B, T, D)` |
| `apply_rotary_pos_emb` | 把 cos/sin 应用到 Q 和 K 上 | q_rot/k_rot: `(B, H, T, D)` |

分开的原因：
- Q 和 K 共用同一份 cos/sin，算一次复用
- 有些变体只旋 Q 不旋 K，分开更灵活
- cos/sin 是纯位置的，跟内容无关，可以预计算

## 6. forward 的形状流程

```
inv_freq:              (dim/2,)       = (32,)
    ↓ [None, :, None] + expand
inv_freq_expanded:     (B, dim/2, 1)  = (2, 32, 1)
position_ids:          (B, T)         = (2, 10)
    ↓ [:, None, :]
position_ids_expanded: (B, 1, T)      = (2, 1, 10)
    ↓ matmul
freqs:                 (B, dim/2, T) = (2, 32, 10)
    ↓ transpose(1, 2)
freqs:                 (B, T, dim/2) = (2, 10, 32)
    ↓ cat([freqs, freqs], dim=-1)
emb:                   (B, T, D)     = (2, 10, 64)
    ↓ cos / sin
cos, sin:              (B, T, D)     = (2, 10, 64)
```

## 7. rotate_half 的作用

把向量"后半部分取负号挪到前面"：

```
x = [x0, x1, x2, x3 | x4, x5, x6, x7]
rotate_half(x) = [-x4, -x5, -x6, -x7, x0, x1, x2, x3]
```

配合 `x * cos + rotate_half(x) * sin`，恰好实现每对维度的 2D 旋转。

## 8. apply_rotary_pos_emb 的形状广播

cos/sin 形状 `(B, T, D)`，q/k 形状 `(B, H, T, D)`。

通过 unsqueeze 在 H 维插入 size-1，PyTorch 自动广播：

```python
cos = cos.unsqueeze(1)   # (B, 1, T, D)
q_rot = q * cos + rotate_half(q) * sin  # (B, H, T, D)
```

## 9. register_buffer vs nn.Parameter

| | `nn.Parameter` | `register_buffer` |
|---|---|---|
| optimizer 更新 | ✅ | ❌ |
| 进入 state_dict | ✅ | ✅（persistent=True） |
| 随 .cuda() 迁移 | ✅ | ✅ |
| 用途 | weight, bias | 固定查找表（inv_freq、Sinusoidal 编码） |

inv_freq 是固定常数，不需要学习，用 `register_buffer` 正确。

## 10. 遇到的问题

| 问题 | 原因 | 修法 |
|---|---|---|
| `__init__` 漏 `super().__init__()` | PyTorch 参数注册机制不启动 | 加上 |
| `inv_freq` 公式少写 2 | `2i/dim` 写成了 `i/dim` | 公式对照原版检查 |
| `angles` 在 `__init__` 里提前算好 | 没用到，forward 里白算 | 只存 inv_freq，forward 里实时算 |
| forward 里形状混淆 | 不确定每一步是几维 | 画形状图，每一步跟着走 |
| `rotate_half(v)` 写成字母 v | 抄代码时笔误，v 应该是 k | 检查变量名 |
| `apply_rotary_pos_emb` 缺 return | 函数算完没返回 | 加上 `return q_rot, k_rot` |

## 11. 自检 Q&A

> **Q1: 为什么旋转能让 attention 点积只依赖相对位置？**
> 位置 m 的 Q 旋转 mθ，位置 n 的 K 旋转 nθ，点积 = |q||k|cos((m-n)θ)。角度差 (m-n) 就是相对位置，绝对位置 m 和 n 被抵消了。

> **Q2: 为什么要用不同频率（高频+低频）？**
> 高频分辨局部细节（近距离），低频感知远距离依赖。类似傅里叶变换，多频率叠加才能覆盖不同尺度的模式。

> **Q3: `rotate_half` 为什么这样设计？**
> 它让向量前一半和后一半形成配对，前一半用 `x` 当实部、`-x2` 当虚部，配合 cos/sin 完成 2D 旋转。不是显然的，但数学上严格等价。

> **Q4: `...` 省略号切片是什么意思？**
> `x[..., :D//2]` 等价于 `x[:, :, :, :D//2]`（假设4维）。省略号代表"前面所有维度不动，只操作最后一个维度"，适配任意形状。

> **Q5: 为什么 inv_freq 用 buffer 而不是 Parameter？**
> inv_freq 是固定的数学常数，不参与训练。Parameter 会被 optimizer 更新，buffer 不会。两者都会进 state_dict 并随设备迁移。

> **Q6: `expand` 和 `view` 有什么区别？**
> `view` 改变形状但共享底层存储（不复制数据），要求总元素数不变。`expand` 在指定维度复制数据（广播），返回新视图，不改变总元素数。inv_freq 从 (dim/2,) expand 到 (B, dim/2, 1) 是在第0维复制。

> **Q7: 为什么 cos/sin 要包 fp32 计算？**
> 三角函数对精度敏感，bf16 下 cos/sin 计算误差大。GemmaRotaryEmbedding 用 `@torch.no_grad()` + `.float()` 确保全用 fp32。

> **Q8: `unsqueeze(1)` 和 `transpose(1,2)` 有什么区别？**
> `unsqueeze(1)` 在第1维插入 size-1 维度，(B,T,D) → (B,1,T,D)。`transpose` 交换两个维度，(B,T,D) transpose(1,2) → (B,D,T)。用途不同：unsqueeze 是为了广播，transpose 是为了调整维度顺序。

## 12. 下一步

Day 2：**GQA Attention（Grouped Query Attention）**。

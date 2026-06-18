# Day 11-12 - Flow Matching / Sinusoidal Embedding / adaLN 笔记

## 一、Flow Matching 是什么

### 1.1 问题：怎么生成连续动作

机器人动作是 **连续向量**（比如 7 维关节角度）。怎么训练一个模型从噪声生成动作？

| 方法 | 特点 |
|------|------|
| 离散化（切格子） | 精度差 |
| GAN | 训练不稳定 |
| Diffusion | 主流，但推理 50-1000 步 |
| **Flow Matching** | 本质同 Diffusion，但更简洁，推理 5-10 步 |

### 1.2 核心思想：学一条「流」

把"从噪声到动作"想象成一条**路径**：

```
t=0                                          t=1
[纯噪声] ──────────────────────────────→ [真实动作]
   x₀                                          x₁
```

模型不直接预测 $x_1$，而是预测**速度场**：

$$
v(x, t) = \frac{dx}{dt}
$$

告诉你"在位置 $x$、时间 $t$ 的瞬间，应该往哪个方向走多快"。

类似导航：不是直接告诉司机终点坐标，而是每秒告诉他当前该开多快、什么方向。

### 1.3 训练流程

每一步训练只做这几件事：

```
1. 拿一条真实动作       x₁ ← dataset
2. 采样一个噪声         x₀ ~ N(0, I)
3. 采样一个时间         t ~ Uniform(0, 1)
4. 线性插值得到中间状态  ψ_t = (1-t)·x₀ + t·x₁
5. 真实速度             v_target = x₁ - x₀
6. 模型预测速度         v_pred = model(ψ_t, t, condition)
7. Loss = MSE(v_pred, v_target)
```

数学公式：

$$
\psi_t = (1-t)\,x_0 + t\,x_1
$$

$$
\frac{d\psi_t}{dt} = x_1 - x_0
$$

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t,\,x_0,\,x_1} \left\| v_\theta(\psi_t,\,t) - (x_1 - x_0) \right\|^2
$$

**就这么简单**——没有 KL divergence，没有变分推断，没有 score matching。

### 1.4 推理流程（Euler 法）

从噪声 $x_0$ 出发，分 $N$ 步走到 $x_1$：

```
x = sample_noise()                     # x₀
for k in range(N):                     # N = 10 步
    t = k / N
    v = model(x, t, condition)
    x = x + (1/N) * v
return x                               # ≈ x₁
```

数学上：

$$
x_{k+1} = x_k + \frac{1}{N}\,v_\theta(x_k,\,t_k),\quad t_k = \frac{k}{N}
$$

每步只调一次 model，10 步生成一个动作。

### 1.5 为什么是直线？

$\psi_t = (1-t)\,x_0 + t\,x_1$ 是 $x_0$ 和 $x_1$ 之间的**直线**。沿着这条直线，速度 $\dot{\psi}_t = x_1 - x_0$ **是常数**（不随 $t$ 变化）。

**Flow Matching 的目标函数**：让模型在每一点的预测速度，等于这条直线的方向。

如果模型完美学到这一点，从 $x_0$ 出发沿模型预测的速度走，就能走到 $x_1$。

### 1.6 与 Diffusion 的对比

| 维度 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 训练目标 | 预测噪声 $\epsilon$ 或 score | 预测速度 $v$ |
| 中间路径 | 复杂（噪声 schedule） | 简单（直线） |
| 推理步数 | 50-1000 | 5-10 |
| 数学复杂度 | KL 散度、变分推断 | 纯 MSE |
| 直观性 | 抽象 | 几何（沿直线走） |

---

## 二、Sinusoidal Embedding：把标量 t 变成向量

### 2.1 为什么需要

模型的输入：

| 输入 | 维度 |
|------|------|
| $\psi_t$（当前状态向量） | 已经高维 |
| $t$（时间） | **标量** |
| condition（图像、文本、proprio） | 已经高维 |

但 Transformer 的输入必须是高维向量。**怎么把标量 $t$ 变成 1024 维？**

### 2.2 错误做法：直接 Linear

```python
nn.Linear(1, 1024)(t)   # ❌
```

问题：Linear 是线性变换，$t=0.1$ 和 $t=0.5$ 的输出只差一个常数倍，模型很难区分不同时间。

### 2.3 Sinusoidal 编码

把 $t$ 用**多个不同频率**的正弦/余弦编码：

$$
\text{emb}(t) = \big[\,\sin(t \omega_0),\ \cos(t \omega_0),\ \sin(t \omega_1),\ \cos(t \omega_1),\ \ldots\big]
$$

频率按几何级数递减：

$$
\omega_i = \frac{1}{P^{2i / d}} = P^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2} - 1
$$

其中 $P$ 是 `max_period`（通常 10000），$d$ 是 embedding 维度。

`max_period=10000` 时：
- $\omega_0 = 1$（最快指针）
- $\omega_{d/2-1} \approx 1/10000$（最慢指针）

### 2.4 钟表类比

类比：钟表为什么有 3 根针？

| 指针 | 频率 | 看什么尺度 |
|------|------|-----------|
| 时针 | 低频，慢 | 大尺度（小时级） |
| 分针 | 中频 | 中尺度（分钟级） |
| 秒针 | 高频，快 | 小尺度（秒级） |

**任何时刻只有 3 根针位置组合起来，才能唯一确定时间。**

每个 $\sin(t\omega)$ 就像一根指针，$\omega$ 决定它转得多快：

$$
\text{周期} = \frac{2\pi}{\omega}
$$

- $\omega = 1$ → 周期 $\approx 6.28$，**快指针**
- $\omega = 1/10000$ → 周期 $\approx 62800$，**慢指针**

### 2.5 高频 vs 低频的分工

#### 高频指针擅长「微小变化」

比较 $t=0.1$ 和 $t=0.2$：

```
ω = 1（高频）:
  sin(0.1) ≈ 0.0998
  sin(0.2) ≈ 0.1987
  差值 ≈ 0.099   ← 区分明显 ✅

ω = 0.0001（低频）:
  sin(0.00001) ≈ 0.00001
  sin(0.00002) ≈ 0.00002
  差值 ≈ 1e-5   ← 几乎一样 ❌
```

#### 低频指针擅长「大尺度差异」

比较 $t=10$ 和 $t=10000$：

```
ω = 1（高频）:
  sin(10)    ≈ -0.544
  sin(10000) ≈ -0.305
  差值随机，因为高频转了 1591 圈，方向乱 ❌

ω = 0.0001（低频）:
  sin(0.001) ≈ 0.001
  sin(1)     ≈ 0.841
  差值 ≈ 0.84   ← 清楚区分 ✅
```

### 2.6 单频率的「周期性盲区」

任何 $\sin$ 都是**周期性**的：

$$
\sin(0) = \sin(2\pi) = \sin(4\pi) = 0
$$

**只用一个频率**，模型会把不同的 $t$ 当成同一个 $t$。

但**多个频率联合**就能消除歧义——只要频率不是简单倍数关系（几何级数保证了这一点），不同 $t$ 组合出来的向量唯一。

```
t=0:     [sin(0·1), sin(0·0.5), sin(0·0.0001)]      = [0,    0,    0]
t=2π:    [sin(2π·1), sin(2π·0.5), sin(2π·0.0001)]   = [0,   ~1,  ~0.0006]
                                                            ↑     ↑
                                                       慢指针救场，区分开了
```

### 2.7 为什么用几何级数

$$
\omega_i = P^{-2i/d}
$$

代入 $i=0,1,\ldots,d/2-1$（取 $d=8, P=10000$）：

| $i$ | $\omega_i$ |
|-----|-----------|
| 0 | $1$ |
| 1 | $\approx 0.1$ |
| 2 | $0.01$ |
| 3 | $\approx 0.001$ |

每个 $\omega$ 是前一个的约 $1/10$。这保证了：

1. **频率范围广**：覆盖 4 个数量级
2. **频率不重叠**：每个频率独立，不浪费
3. **均匀覆盖各尺度**：任何时间尺度都有合适的指针

---

## 三、PyTorch 实现

### 3.1 SinusoidalPosEmb

```python
import math
import torch
from torch import nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        输入 t: (B,)  标量批次
        输出  : (B, dim)
        """
        half = self.dim // 2
        # freqs: (half,) 几何递减 1 → 1/max_period
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        # args: (B, half) = t.unsqueeze(-1) * freqs.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        # emb: (B, dim)
        return torch.cat([args.sin(), args.cos()], dim=-1)
```

### 3.2 形状变化

```
t.shape           = (B,)
t.unsqueeze(-1)   = (B, 1)
freqs.shape       = (half,) = (dim/2,)
args.shape        = (B, half)
sin(args), cos(args) = (B, half) each
cat                = (B, dim)
```

### 3.3 ActionEncoder

把动作向量映射到 action expert 的隐藏空间：

$$
\text{action\_token} = W_a \cdot a + b_a, \quad W_a \in \mathbb{R}^{d_h \times d_a}
$$

```python
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_size)

    def forward(self, actions):
        return self.proj(actions)
```

输入输出：
- 输入：$(B, \text{horizon}, d_a)$，例如 $(B, 4, 7)$
- 输出：$(B, \text{horizon}, d_h)$，例如 $(B, 4, 1024)$

镜像的 `ActionDecoder` 直接用 `nn.Linear(hidden_size, action_dim)` 即可，无需单独写类。

---

## 四、adaLN：把时间注入每一层

### 4.1 为什么需要 adaLN

`TimeEncoder` 输出 $(B, d_t)$ 的时间嵌入，怎么进入 Transformer？

| 方案 | 描述 | 缺点 |
|------|------|------|
| 拼成 token | 把 `time_emb` 当 prefix token 拼到序列前 | 占 token 位、效率低 |
| **adaLN** | 在每一层用 `time_emb` 调制 hidden states | ✅ 不占 token 位，每层都注入 |

### 4.2 普通 RMSNorm vs adaLN

**普通 RMSNorm**（参数固定）：

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}} \odot w
$$

权重 $w$ 是训练后的固定参数，与 batch 内容无关。

**adaLN**（参数由时间嵌入动态生成）：

$$
[\gamma, \beta] = \text{MLP}(c)
$$

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}} \odot (1 + \gamma) + \beta
$$

其中 $c$ 是 `time_emb`，$\gamma \in \mathbb{R}^{B \times d}$ 是 scale，$\beta \in \mathbb{R}^{B \times d}$ 是 shift。

**关键点**：是 $1 + \gamma$ 而不是 $\gamma$——保证 $\gamma=0$ 时退化到纯 RMSNorm（不影响初始化）。

### 4.3 adaLN-Zero（DiT 改进）

DiT 论文（图像生成）发现：把生成 $\gamma, \beta$ 的 MLP **最后一层权重 + bias 全初始化为 0**，效果更好。

初始时：

$$
\gamma = 0,\ \beta = 0 \implies y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}} = \text{pure RMSNorm}(x)
$$

每层从「恒等通路 + 归一化」开始训练，避免 action expert 初始化噪声扰乱 VLM 的 attention，训练更稳定。

### 4.4 几何直观

把 $\gamma, \beta$ 想象成"调音师"：

```
x  ──[RMSNorm]──> 各维度归一化（音量统一）
                     │
                     └── 乘 (1+γ) ── 按当前 t 的需要重新调音量
                            │
                            └── 加 β ── 加一个偏置（音色）
                                  │
                                  └── y
```

每个 t 对应不同的 $(\gamma, \beta)$，所以**每个时间步的 hidden states 调制方式都不同**。模型由此知道"现在去噪到了哪一步"。

### 4.5 在 Transformer 层里的位置

每个 DecoderLayer 有 **2 个** adaLN（替代原 RMSNorm）：

```
hidden_states
    │
    ├── AdaptiveRMSNorm(hidden, time_emb)  ← Norm 1 (attn 前)
    ├── Self-Attention
    ├── + residual
    │
    ├── AdaptiveRMSNorm(hidden, time_emb)  ← Norm 2 (mlp 前)
    ├── MLP
    └── + residual
```

只有 **action expert** 用 adaLN；VLM 和 Proprio 还是普通 RMSNorm（它们不依赖时间）。

### 4.6 PyTorch 实现

```python
class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, time_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale_proj = nn.Linear(time_dim, hidden_size)
        self.shift_proj = nn.Linear(time_dim, hidden_size)

        # adaLN-Zero 初始化：开始等价于纯 RMSNorm
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def _norm(self, x):
        # x: (B, T, hidden_size)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x, time_emb):
        # x:        (B, T, hidden_size)
        # time_emb: (B, time_dim)
        x_norm = self._norm(x)
        scale = self.scale_proj(time_emb).unsqueeze(1)  # (B, 1, hidden_size)
        shift = self.shift_proj(time_emb).unsqueeze(1)  # (B, 1, hidden_size)
        return x_norm * (1 + scale) + shift
```

### 4.7 设计选择对比

| 实现细节 | 选项 1 | 选项 2（推荐） |
|---------|--------|---------------|
| 计算 scale/shift | 一个 `Linear(time_dim, 2*hidden)` + `chunk` | 两个独立 `Linear` |
| 初始化 | 普通 init | **zero init**（adaLN-Zero） |
| 调制公式 | $y = \gamma \odot x_\text{norm} + \beta$ | $y = (1+\gamma) \odot x_\text{norm} + \beta$ |

本项目用**选项 2**：可读性更好、训练更稳。

---

## 五、Flow Matching 在 PiZero 中的串联

### 训练时

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 从 dataset 拿 (image, text, proprio, action)             │
│ 2. 采样 t ∈ [0,1], x₀ ∈ N(0, I)                             │
│ 3. ψ_t = (1-t)·x₀ + t·action                                │
│ 4. SinusoidalPosEmb(t) → time_emb (1024 维)                 │
│ 5. ActionEncoder(ψ_t) → action_token                        │
│                                                              │
│ 6. 喂给 PiZero:                                             │
│      image+text → VLM expert                                 │
│      proprio    → Proprio expert                             │
│      action_token (+ time_emb 注入) → Action expert          │
│                                                              │
│ 7. Action expert 输出预测速度 v_pred                          │
│ 8. Loss = MSE(v_pred, action - x₀)                          │
└─────────────────────────────────────────────────────────────┘
```

### 推理时

```
┌─────────────────────────────────────────────────────────────┐
│ x = sample_noise()              # x₀                        │
│ for t in [0, 0.1, ..., 0.9]:                                │
│     time_emb = SinusoidalPosEmb(t)                          │
│     v = model(x, time_emb, image, text, proprio)            │
│     x = x + 0.1 * v                                         │
│ return x                        # 最终 action                │
└─────────────────────────────────────────────────────────────┘
```

### 关键观察

1. **$t$ 在每一层都被注入**（通过 adaLN，详见第四节）
2. **训练和推理用同一个网络**（Flow Matching 没有"训练专用"的 KL 项）
3. **VLM 和 Proprio 不依赖 $t$**，只有 Action expert 需要时间信息（所以也只有 Action expert 用 adaLN）

---

## 六、关键收获

1. **Flow Matching = 学速度场 + 直线插值**，训练目标就是 $\text{MSE}(v_\text{pred},\, x_1 - x_0)$，简单到不可思议。
2. **Sinusoidal = 把标量 $t$ 编成多频率向量**，让模型能分辨不同时间。
3. **多频率组合消除单频率的周期性盲区**——这是钟表 3 根针的智慧。
4. **几何级数频率**保证从大尺度到小尺度均匀覆盖。
5. **adaLN = 用时间嵌入动态生成 RMSNorm 的 scale/shift**，把时间信息注入每一层 hidden states。
6. **adaLN-Zero（zero init）**让 action expert 从「恒等通路」开始训练，避免初始化噪声扰乱预训练的 VLM。
7. **Flow Matching 的优雅在于解耦**：condition（image+text+proprio）和 noise schedule（t、x₀、ψ_t）完全独立，模块化清晰。

---

## 七、复现进度对照（Day 11-12）

| 模块 | 落点 | 状态 |
|------|------|------|
| `TimeEncoder` (Sinusoidal) | `vla/modules.py` | ✅ |
| `ActionEncoder` | `vla/modules.py` | ✅ |
| `AdaptiveRMSNorm` (adaLN-Zero) | `vla/modules.py` | ✅ |
| 对应测试 | `tests/model/vla/test_modules.py` | ✅ |
| Flow Matching forward (Day 13) | `vla/yuepi0.py::forward` | ⏳ 下一步 |
| Flow Matching infer (Day 14) | `vla/yuepi0.py::infer_action` | ⏳ |

懂了之后再去看 `pizero.py:forward` 那 50 行代码，会发现"啊，就这？"。这就是 Flow Matching 的魅力——简洁。

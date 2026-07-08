# Flow Matching 笔记

## 常见疑问（Q&A）

**Q1: forward 里 t 为什么是随机 torch.rand(B) 而不是固定值？**

每个 batch 样本独立采样 t ~ Uniform(0,1)，目的是让模型在**所有噪声水平上**都练过。

```text
同一个样本，不同 epoch 见到不同的 t:
  epoch 1: t=0.23 → 加 77% 噪声，学粗略方向
  epoch 2: t=0.81 → 加 19% 噪声，学精修
  epoch 3: t=0.05 → 加 95% 噪声，学从极端噪声出发

一个 batch 16 个样本 = 16 个不同 t = 一次覆盖 16 个噪声水平
```

**Q2: forward 里为什么没有 10 步去噪流程？**

因为 flow matching 的训练和推理是**完全分开的两个阶段**：

```text
训练 = 单步（single step）
  采样一个随机 t → 造一个 x_t → 一次 forward 输出 v_pred → 算一次 loss → backward
  没有循环，不需要多步

推理 = 迭代（multi-step）  
  从纯噪声出发 → 循环 10 次 → 每次调模型预测 v → 欧拉法走一步
  需要多步因为早期 v 不准，要逐步纠正
```

训练时为什么不需要循环？因为模型的核心能力是"在任意 t 下预测去噪方向"。训练时采到哪个 t 就在哪个 t 上练，积累足够多不同 t 的经验后，推理时才能连续 10 步走准。

**Q3: 所有 flow matching / diffusion 都是训练单步、推理多步吗？**

**是的。** Diffusion 也是这个模式，只是去噪步数更多（1000 步）。Flow matching 的路径是直线，所以 10 步就够了。

**Q4: 插值公式 x_t = (1-t)·noise + t·action 是干什么的？**

训练时你只有真实 action 和随机 noise，要造出"中间状态"给模型练习。插值就是这个人造工具：

```text
t=0:   x_t = 纯噪声   → 模型学"从极端噪声出发的大致方向"
t=0.5: x_t = 半噪声半动作 → 模型学"中等噪声的去噪"
t=1:   x_t = 纯动作   → 模型学"精修"
```

推理时不插值，因为根本不知道 action——靠模型学到的 v 一步步"走"到 action。

---

## 解决的问题

直接回归（behavior cloning）遇到多模态动作分布时，会取平均，导致**动作崩溃**。

Flow matching 通过从噪声到动作的去噪路径，生成多样且合理的动作。

---

## 训练目标

```
不再直接预测 action，而是预测速度场 v
v = action - noise    (训练时已知，直接算)
```

模型学的是：给定 context、带噪动作 x_t、噪声水平 t，输出去噪方向 v。

---

## 训练流程（一次 forward，不需要多步）

```
1. 采样 t ~ Uniform(0,1)       # 随机噪声水平
2. 采样 noise ~ N(0, I)         # 随机噪声
3. 插值造中间状态 x_t:
   x_t = (1-t)·noise + t·action    # 从噪声到真实动作的线性路径
4. 算真实速度场:
   v_target = action - noise
5. 模型预测 v_pred:
   v_pred = model(context, x_t, t)
6. Loss = MSE(v_pred, v_target)
```

为什么 t 每次随机 Uniform(0,1)：

- 同一个样本在不同 epoch 见到不同的 t，在各种各样的噪声水平上都练过
- 这样推理时从纯噪声出发，模型知道怎么一步步走

---

## 推理流程（需要多步去噪）

```
1. 从纯噪声开始: x = noise ~ N(0, I)
2. 分 N=10 步走:
   for step in range(10):
       t = step * dt                    # 从 0 到 1
       v = model(context, x, t)
       x = x + dt * v                   # 欧拉法去噪
3. 返回 x ≈ 真实 action
```

训练和推理用的是同一个主干，只是推理时循环调用 10 次。

---

## 为什么 10 步而不是 1 步

t=0 时 x 是纯噪声，v 是 E[action | noise]，方差极大（几乎不含信息）。

1 步从噪声走到真实，方向定错，下面再也救不回来。

多步让模型逐步调整方向：早期走小步（错了还能改），后期 x 接近真实时 v 越来越精确。

---

## v_target 公式说明（重要）

```
标准 flow matching:
  x_t = (1-t)·noise + t·action
  v_target = action - noise

OpenVLA 风格（加 sigma_min 避免除零）:
  x_t = (1 - (1-sig)·t)·noise + t·action
  v_target = action - (1-sig)·noise

两种都自洽，只要保持训练和推理一致。
```

---

## t 的采样策略：Uniform vs Beta（π0 论文）

训练流程第 1 步"采样 t"有两种策略，由 config 的 `flow_sampling` 字段控制。Q1 里举的 uniform 例子是为了讲"为什么 t 要随机"，实际 π0 用的是下面的 beta 策略。

### Uniform（torch.rand）

t 在 [0,1] 均匀分布，E[t] = 0.5，模型在所有噪声水平花一样精力，没有侧重。

原版 uniform 分支还用 low-discrepancy 采样降低小 batch 方差（不是简单 `torch.rand(bsz)`）：

```python
t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
```

拆三段看这个公式：

- `torch.arange(bsz) / bsz`：等分网格，bsz=4 时是 [0, 0.25, 0.5, 0.75]，点等距排开
- `torch.rand(1)`：一个标量共享偏移，所有点加同一个值，整体平移网格
- `% (1 - eps)`：超过 1 的绕回来，eps=1e-5 防止取到 1.0 边界

bsz=4 代两个偏移看效果：

```text
r=0.1: [0, 0.25, 0.5, 0.75] + 0.1 = [0.1, 0.35, 0.6, 0.85]   均匀散开
r=0.9: [0, 0.25, 0.5, 0.75] + 0.9 = [0.9, 1.15, 1.4, 1.65]
       取模后 = [0.9, 0.15, 0.4, 0.65]                        还是均匀

对比 torch.rand(4): 可能 = [0.1, 0.12, 0.85, 0.87]          前两后两扎堆
```

核心：网格保证覆盖均匀（每段 1/bsz 必有一个点），共享偏移保证整体随机，两者结合既均匀又随机。

关键别写错：偏移必须是 `torch.rand(1)` 一个标量。一旦写成 `torch.rand(bsz)` 每点独立偏移，网格结构就垮了，退化回普通 torch.rand(bsz)，白费。

为什么小 batch 下尤其明显：bsz=4 时 torch.rand 很容易扎堆；bsz=256 时大数定律自然铺匀，收益就小。

注意和 beta 的区别：low-discrepancy 是让均匀覆盖更稳（分布形状不变），beta 是故意偏置分布形状（偏向噪声端）。两码事。

### Beta(1.5, 1) 翻转（π0 论文做法）

```python
z = Beta(1.5, 1).sample((bsz,))   # 直接采的是 z，偏大
t = t_max * (1 - z)               # 翻转后 t 偏小
```

Beta(α, β) 是 [0,1] 上的连续分布，两个形状参数控制偏向：

- α 越大越偏向 1，β 越大越偏向 0
- α=β=1 时退化为均匀分布（等价 torch.rand）
- Beta(1.5, 1)：α=1.5 > β=1，PDF ∝ sqrt(z) 在 [0,1] 单调增，z 偏大
- 均值 E[z] = α/(α+β) = 1.5/2.5 = 0.6

翻转 t = t_max·(1-z)：

- z 偏大 → (1-z) 偏小 → t 偏小
- E[t] = t_max·(1-E[z]) = 0.999×0.4 ≈ 0.4

对照 uniform 的 0.5，beta 把训练重心往噪声端挪了 0.1。

### t_max = 1 - flow_sig_min

flow_sig_min 默认 0.001，所以 t_max = 0.999。作用是卡住 t 上限，防止 t 正好等于 1 的数值奇点。不影响"谁偏大谁偏小"的判断。

### 为什么偏向噪声端

t 是去噪进度：t=0 是纯噪声（像电视雪花，啥也看不出），t=1 是真实动作。
模型任务：给一个中间状态，猜"该往哪个方向走一步"。
噪声端（小 t）模型看到的半成品和真实动作差最远，猜方向最难；接近真实动作（大 t）时一眼能看出，太简单学不到。

所以 π0 让模型多在噪声端花时间。这叫"非均匀时刻采样"，diffusion 类模型的常见技巧。

### 关键：偏小 ≠ 只取小值

Beta 采样是"小值多采、大值少采但也要采"，整个 [0, t_max] 都覆盖。
不能图省事写 `t = torch.rand(B) * 0.5`，那样 t 永远进不了 0.5 以上，模型在大 t 区间完全不训练，去噪走到后半程就瞎了。

### 分布归属（易混淆点）

- 直接从 Beta(1.5,1) 采的是中间变量 z
- t = t_max·(1-z) 是算出来的，不是直接采
- 忽略 t_max≈1，则 1-t ≈ z，所以 1-t（近似）才是偏大的 Beta(1.5,1)
- t 是它的镜像，偏小（噪声端）

### 代码位置（对齐 open-pi-zero）

- 采样逻辑在 train.py 的 `sample_fm_time`，不在 model.forward 内部
- forward 只接收 t、不负责采样（原版 pizero.py:735 forward 签名有 t 形参）
- config 字段：`flow_sampling` / `flow_alpha` / `flow_beta` / `flow_sig_min`

---

## 与 Diffusion 对比

| 特性 | Diffusion | Flow Matching |
|------|-----------|---------------|
| 训练 | 复杂（noise schedule、KL） | 简单（一行插值 + MSE） |
| 推理步数 | 100-1000 步 | 10 步 |
| 路径形状 | 弯曲马尔可夫链 | 直线常速 |
| 多模态生成 | ✓ | ✓ |

---

## Sinusoidal Time Embedding（TimeEncoder）

输入 t: 标量 ∈ [0, 1]
输出 time_emb: (B, time_dim) 稠密向量

为什么不用单层 Linear(t)：
- 线性层对 0.1 和 0.2 只能学到线性关系，表达能力差
- sin/cos 编码把连续的 t 映射成高维傅里叶特征，模型更容易捕捉不同 t 下的不同行为

公式：
```
freqs[i] = exp(-log(max_period) * i / half_dim)    # i = 0, 1, ..., half_dim-1
angles[t, i] = t * freqs[i]
time_emb = [sin(angles), cos(angles)]               # dim 维向量
```

其中 max_period 控制频率范围。max_period 越大，低频成分越多，对应更宽的 t 敏感性范围。

---

## 关键概念区分

| 概念 | 含义 | 场景 |
|------|------|------|
| 训练插值 | noise → action 的线性路径 | 训练时造中间状态 |
| 推理多步 | 从噪声出发，逐步去噪 | 推理时生成动作 |
| 采集频率 | 5Hz 录制物理数据 | 数据采集 |
| 执行频率 | 50-100Hz 控制机器人 | 控制器层 |
| 物理时间步 | action[0], action[1] 在真实时间里的先后 | 动作 chunk 内部 |
| Flow Matching t ∈ [0,1] | 噪声水平的标量参数 | 训练/推理的数学框架 |
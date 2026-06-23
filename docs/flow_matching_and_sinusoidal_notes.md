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
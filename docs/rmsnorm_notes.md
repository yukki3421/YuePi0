# Day 1 - RMSNorm 复现笔记

## 1. RMSNorm 是什么

一句话：**把向量除以它自己的"均方根"，让向量长度归一，然后乘一个可学习的缩放。**

公式：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

跟 LayerNorm 对比，砍掉了两件事：

| 步骤 | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 $\mu$ | ✅ 有 | ❌ 没有 |
| 除标准差/RMS | ✅ 有 | ✅ 有（用 RMS） |
| 乘 $\gamma$ | ✅ 有 | ✅ 有 |
| 加 $\beta$ | ✅ 有 | ❌ 没有 |

RMSNorm 论文核心 claim：**重新中心化（减均值）不是必要的，重新缩放（除 RMS）才是关键。** 实验上效果几乎不变，速度更快，Llama / Gemma / PaLM 都用它。

## 2. 维度问题

数据形状 `(B, T, D)`，RMSNorm 沿 **最后一维 D**（hidden dim）归一。

```python
x.pow(2).mean(dim=-1, keepdim=True)
```

为什么是 D？因为每个 token 自己的"幅度"才是要稳定的对象，token 之间、样本之间不需要统一幅度。

`keepdim=True` 不能漏：保留维度方便后续除法广播回 `(B, T, D)`。

## 3. Gemma 的小特色

Gemma 写成 `(1 + weight)` 形式，且 `weight` 初始化为 0：

$$
\text{out} = \frac{x}{\text{RMS}(x)+\epsilon} \cdot (1 + \gamma)
$$

初始时 $(1+0)=1$，等价于"啥也不学"，纯归一化。这是常见的"初始恒等映射"trick，让训练初期更稳。

## 4. 数值精度问题（最重要的踩坑）

### 4.1 三种 dtype 对比

| dtype | 显存 | 速度 | 精度 | 用途 |
|---|---|---|---|---|
| fp32 | 大 | 慢 | ~7 位有效数字 | 数值稳定的"基准" |
| bf16 | 一半 | 快一倍 | **~2-3 位有效数字** | 训练主流 |

bf16 的核心特点：**范围跟 fp32 一样大（不容易溢出），但精度极差**。

### 4.2 "大数加小数"的灾难

bf16 在 5.0 附近的"台阶"约 0.04。所以：

```
bf16 里：5.00 + 0.01 = 5.00   ← 0.01 被吃掉
fp32 里：5.00 + 0.01 = 5.01   ← 正常
```

### 4.3 这跟 RMSNorm 有什么关系

`mean(x²)` 要把 D 个数加起来（D 在 Gemma 里是 4096）。bf16 累加过程：

```
sum = 0.01
sum += 0.01  → 0.02
...
sum 累到 5 左右
sum += 0.01  → 还是 5.00  ❌
sum += 0.01  → 还是 5.00  ❌
（往后全废）
```

最终 `mean(x²)` 可能比真值少 10%~30%，整个 RMSNorm 输出就崩了。

### 4.4 解决方案

临时升 fp32 算，算完降回原 dtype：

```python
def forward(self, x):
    input_dtype = x.dtype       # 记原 dtype
    output = self._norm(x.float())   # 升 fp32 算
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)    # 降回原 dtype
```

代价：这一层显存翻倍（整体可忽略）。收益：精度不崩。

### 4.5 什么时候需要 fp32

只要这步里"很多数加起来"，就要升：

| 是否需要 fp32 | 操作 |
|---|---|
| ✅ 需要 | RMSNorm / LayerNorm（加 D 个） |
| ✅ 需要 | Softmax（加 T 个，且有指数运算） |
| ✅ 需要 | Loss 计算（加 batch 内所有） |
| ✅ 需要 | AdamW 的 m, v 状态 |
| ❌ 不用 | 矩阵乘法（GPU tensor core 硬件帮你处理） |
| ❌ 不用 | 激活函数（GELU、SiLU 等，没累加） |
| ❌ 不用 | embedding lookup（纯查表） |

**经验法则**：累加 ≥ 几百个数，就要 fp32。

## 5. 实现细节踩坑

| 坑 | 后果 | 修法 |
|---|---|---|
| `nn.module` 写成小写 | `AttributeError` | `nn.Module`（大写 M） |
| `super().self.__init__()` | 语法错误 | `super().__init__()` |
| 用普通 `torch.zeros` 而不是 `nn.Parameter` | 参数不被注册，无梯度，`load_state_dict` 失败 | 必须 `nn.Parameter(torch.zeros(dim))` |
| 创建了 `self.gamma` 但 forward 里没用它 | 权重白建，allclose 必挂 | forward 里要乘 `(1 + self.gamma)` |
| forget `keepdim=True` | shape 不匹配，广播错误 | `mean(dim=-1, keepdim=True)` |
| 用 `1/sqrt` 而不是 `rsqrt` | 慢、数值稳定性稍差 | `torch.rsqrt` |
| eps 加在 sqrt 外面 | 数值不稳定 | `rsqrt(mean + eps)`，eps 在里面 |
| 没升 fp32 | bf16 输入时精度爆炸 | `.float()` 进，`.type_as(x)` 出 |

## 6. 测试方法

两套互补的测试：

### 6.1 数学性质测试（不依赖参考实现）

| 性质 | 怎么测 |
|---|---|
| 输出 RMS = 1（weight=0 时） | `y.pow(2).mean(-1).sqrt()` 应该全 ≈ 1 |
| 缩放不变性 | `mine(x)` 和 `mine(x*10)` 输出应该一样 |
| 形状保持 | `y.shape == x.shape` |
| weight 起作用 | weight 全 1 时输出应该是 weight 全 0 时的 2 倍 |
| 梯度可传 | `y.sum().backward()` 后 `mine.weight.grad` 非 None |
| dtype 保持 | bf16 输入 → bf16 输出 |

### 6.2 对齐参考实现（bit-level）

```python
mine = GemmaRMSNorm(dim)
orig = GemmaRMSNorm(dim)
orig.weight.data.normal_()                  # 灌随机权重
mine.load_state_dict(orig.state_dict())     # 拷给 mine
torch.allclose(mine(x), orig(x), atol=1e-6)
```

**关键**：必须给 `orig.weight` 灌非零值。两个 weight 都是 0 → 乘 `(1+0)=1` → 输出相同但毫无意义，是"假对齐"。

## 7. 自检 Q&A

> **Q1: RMSNorm 比 LayerNorm 砍掉了什么？为什么？**
> 砍掉了减均值和 bias。RMSNorm 论文证明：重新中心化不是必要的，只要重新缩放就够。少一次 reduce、少一个参数，更快。

> **Q2: 为什么沿最后一维归一？**
> 最后一维是 hidden dim，每个 token 自己的"幅度"才是要稳定的对象。沿 batch / seq_len 归一没有意义——不同样本、不同位置的语义本身就应该不同。

> **Q3: 为什么 weight 初始化为 0 而不是 1？**
> Gemma 用 `(1 + weight)` 而不是直接 `weight`。weight=0 时等价于乘 1，输出就是纯归一化结果。这是"初始恒等映射"trick，训练初期更稳，等模型自己慢慢偏离 0。

> **Q4: 输入乘 10 后输出会变吗？**
> 不变。`(10x) / RMS(10x) = (10x) / (10·RMS(x)) = x / RMS(x)`，10 倍被分母抵消。这就是 RMSNorm 的"缩放不变性"。

> **Q5: bf16 为什么"大数加小数会丢"？**
> bf16 只有 7 位尾数，相邻两个数的台阶约为数值本身的 0.78%。在 5 附近台阶 ≈ 0.04，加 0.01 比台阶小，被舍入掉。

> **Q6: 为什么 norm 层一定要升 fp32？**
> `mean(x²)` 要累加 D 个数（D=4096）。bf16 累到几之后，每次加的小数全被吃掉，最终 sum 比真值少 10%~30%，整个输出崩。

> **Q7: 矩阵乘法为什么不用升 fp32？**
> GPU tensor core 硬件在 bf16 矩阵乘内部用 fp32 累加，自动处理了精度问题。我们手写的 norm 层没有这个硬件加速，必须自己 upcast。

> **Q8: `nn.Parameter` 和普通 `torch.zeros` 的区别？**
> `nn.Parameter` 会自动注册到模块的 `.parameters()` 里，会被 optimizer 更新、会进入 `state_dict`、梯度会被计算。普通 tensor 啥都没有。

> **Q9: `keepdim=True` 有什么用？**
> 形状从 `(B, T, D)` reduce 到 `(B, T)` 还是 `(B, T, 1)` 的差别。后者可以正确广播回 `(B, T, D)` 做除法，前者会报错或广播错。

> **Q10: `rsqrt` 和 `1/sqrt` 有区别吗？**
> `rsqrt` 是单条 CUDA 指令，比"先 sqrt 再倒数"快约 2 倍，数值稳定性也稍好。所有现代 norm 层都用 `rsqrt`。

## 8. 下一步

Day 1 Part B：**RoPE 旋转位置编码**。

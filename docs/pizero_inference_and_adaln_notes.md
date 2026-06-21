# Day 13 - PiZero 串通 / adaLN-Zero / infer_action 设计笔记

今天的主线:
1. 把 `PiZero.forward()` 串通,smoke test 跑过(✅ 完成)
2. 搞懂 `adaLN` / `adaLN-Zero` 是什么、为什么
3. 设计 `infer_action()` 的参数和结构(待实现)

---

## 一、adaLN 和 adaLN-Zero 是什么

### 1.1 背景:为什么普通 LayerNorm 不够

普通 LayerNorm:

```python
y = (x - mean) / std * γ + β     # γ, β 是常数,所有 sample 共享
```

但在 Flow Matching / Diffusion 里,**同一个 `x_t` 在不同 `t` 下应该被不同地处理**(噪声水平不同)。
于是把 `γ, β` 改成"随 `t` 变化"。

### 1.2 adaLN(Adaptive LayerNorm)

`γ, β` 由 `time_emb` 生成:

```python
scale = Linear(time_emb)    # (B, hidden) ← 每个 sample 自己的 γ
shift = Linear(time_emb)    # (B, hidden) ← 每个 sample 自己的 β

y = norm(x) * (1 + scale) + shift
```

这就是 `modules.py` 里 `AdaptiveRMSNorm` 干的事:

```python
def forward(self, x, time_emb):
    x_norm = self._norm(x)
    scale = self.scale_proj(time_emb).unsqueeze(1)
    shift = self.shift_proj(time_emb).unsqueeze(1)
    return x_norm * (1+scale) + shift   # ← adaLN
```

### 1.3 adaLN-Zero(DiT 推荐版,零初始化)

在 adaLN 基础上,**每个残差分支末端额外乘一个 `α(t)`,并且 `α` 初始化为 0**。

```
                    ┌── adaLN ──┐    ┌── α_attn(t) ──┐
                    │           │    │  (初始=0)     │
x ──────────────► LayerNorm ──► Attn ─► × α ────────► + ──► out
│                                                     ▲
└─────────────────────────────────────────────────────┘
                       残差(直接加 x)
```

公式:

```python
out = x + α_attn(t) * Attention(adaLN(x, t))
                ▲
                └── 由 t 生成,零初始化
```

### 1.4 为什么 α 要零初始化(关键 trick)

把 `α=0` 代进去:

```python
out = x + 0 * f(x, t) = x   # ← 整个 block 等于 identity
```

**训练开始的瞬间,每个 Transformer block 都是 identity,什么也不干**,这带来三大好处:

| 好处 | 解释 |
|------|------|
| 保护预训练权重 | PaliGemma 是预训练好的强 VLM,一开始就让随机初始化的子层扰动它会破坏好特征 |
| 梯度通畅 | `∂out/∂x = 1 + α·∂f/∂x`,`α=0` 时梯度完美穿过 |
| 自学条件强度 | 模型自己决定哪些 block 需要大 α、哪些不需要,而不是被迫吃满条件信号 |

### 1.5 三种模式总览

| 模式 | LayerNorm | 残差缩放 | t 注入位置 |
|------|-----------|----------|------------|
| `None` (`time_cond=True`) | 普通 RMSNorm | 无 | 在 `ActionEncoder` 里 cat 进去 |
| `adaLN` | `AdaptiveRMSNorm` | 无 | 每层 norm 的 γ, β |
| `adaLN-Zero` | `AdaptiveRMSNorm` | `AdaptiveLayerscale`(零初始化) | norm 的 γ, β + 残差分支 α |

Pi0 原论文只用了最朴素的 `time_cond=True`,`open-pi-zero` 额外提供 `adaLN` / `adaLN-Zero` 选项。

### 1.6 `AdaptiveLayerscale` 怎么实现

```python
class AdaptiveLayerscale(nn.Module):
    def __init__(self, hidden_size, time_dim):
        super().__init__()
        self.proj = nn.Linear(time_dim, hidden_size)
        # 关键:零初始化
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, time_emb):
        # x: (B, T, hidden), time_emb: (B, time_dim)
        alpha = self.proj(time_emb).unsqueeze(1)   # (B, 1, hidden)
        return alpha * x                            # 初始 α=0 → 输出 0
```

---

## 二、调 bug:`time_hidden_size` vs `action_hidden_size`

### 2.1 报错

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x1280 and 2048x1024)
```

报错位置:`ActionEncoder.forward` 的 `self.linear_2(emb)`。

### 2.2 追 shape

```
ActionEncoder 隐含假设:  time_emb_dim == hidden_size = 1024
你实际的情况:           time_emb_dim = 256 (来自 yaml 的 time_hidden_size)
                                       hidden_size = 1024 (action_hidden_size)
```

cat 出来:`(B, T, 1024+256) = (B, T, 1280)`
`linear_2` 期望:`nn.Linear(2*1024, 1024)` → 输入最后一维必须是 `2048`
不匹配 → 💥

### 2.3 原版怎么解决

`open-pi-zero/src/model/vla/pizero.py` 第 78-96 行:

```python
if cfg.action_expert_adaptive_mode:        # adaLN / adaLN-Zero 模式
    self.action_encoder = ActionEncoder(..., time_cond=False)
    self.time_embedding = SinusoidalPosEmb(
        cfg.time_hidden_size,              # ← 此时才用 256
        cfg.time_max_period,
    )
else:                                       # time_cond=True 模式
    self.action_encoder = ActionEncoder(..., time_cond=True)
    self.time_embedding = SinusoidalPosEmb(
        self.action_hidden_size,           # ← 关键:用 1024,不是 256
        cfg.time_max_period,
    )
```

**核心结论**:
> `cfg.time_hidden_size`(256)**只在 adaLN 模式下生效**(用于生成 norm 的 scale/shift)。
> 当 `time_cond=True` 时,`time_embedding` 必须建成 `action_hidden_size`(1024),因为要和 action_emb 拼接。

### 2.4 修法(我们采取的)

`yuepi0.py` 的 `PiZero.__init__`:

```python
# 旧(错):
self.time_encoder = TimeEncoder(config.time_hidden_size)   # 256

# 新(对):
self.time_encoder = TimeEncoder(config.action_hidden_size) # 1024
```

修完 smoke test 通过 ✅

### 2.5 教训

> `ActionEncoder` 的 `linear_2 = nn.Linear(2*hidden, hidden)` **隐式假设 time_dim == hidden_size**。
> 这个一致性 **不由 `ActionEncoder` 自己保证,而由 `PiZero.__init__` 在外面保证**。
> 看模块时要看上一层怎么用它,光看模块本身往往看不出隐式假设。

---

## 三、`infer_action()` 设计

训练学的是速度场 `v_θ(x_t, t)`,推理时要从纯噪声出发用 Euler 法走 N 步积到 `t=1`,得到预测动作。

### 3.1 参数清单

跟 `forward(batch)` 比,**少了 `action`**(那是 ground truth,推理时不知道):

```python
@torch.no_grad()
def infer_action(self, batch, num_inference_steps: int = 10):
    """
    batch:
        input_ids:      (B, max_image_text_tokens)
        pixel_values:   (B, 3, 224, 224)
        attention_mask: (B, max_image_text_tokens)
        proprio:        (B, cond_steps, proprio_dim)
    返回:
        action_pred:    (B, horizon_steps, action_dim)
    """
```

### 3.2 函数骨架

```python
@torch.no_grad()
def infer_action(self, batch, num_inference_steps: int = 10):
    input_ids      = batch['input_ids']
    pixel_values   = batch['pixel_values']
    attention_mask = batch['attention_mask']
    proprio        = batch['proprio']

    device = pixel_values.device
    dtype  = pixel_values.dtype
    B = pixel_values.size(0)

    # === 1) 跟 forward 一样准备条件 ===
    vlm_emb     = self.embedder(input_ids, pixel_values)
    proprio_emb = self.proprio_encoder(proprio)
    causal_mask, vlm_pos, proprio_pos, action_pos = \
        self.build_mask_and_position_ids(attention_mask, dtype)

    # === 2) 从纯噪声出发 ===
    x = torch.randn(B, self.num_action_tokens, self.action_dim,
                    device=device, dtype=dtype)

    # === 3) Euler 积分 ===
    dt = 1.0 / num_inference_steps
    t  = torch.zeros(B, device=device, dtype=dtype)
    for _ in range(num_inference_steps):
        time_emb   = self.time_encoder(t)
        action_emb = self.action_encoder(x, time_emb)

        out = self.joint(
            causal_mask,
            {"vlm": vlm_pos, "proprio": proprio_pos, "action": action_pos},
            {"vlm": vlm_emb.clone(),         # ← 必须 clone
             "proprio": proprio_emb.clone(),
             "action": action_emb},
        )
        v = self.action_decoder(out['action'])    # (B, T_a, A)

        x = x + dt * v
        t = t + dt

    return x
```

### 3.3 几个关键点

| 细节 | 为什么 |
|------|--------|
| `@torch.no_grad()` | 推理不要梯度,省一半显存,快 |
| `x = torch.randn(...)`(不是 `torch.rand`) | FM 推理起点是 **高斯噪声** `N(0,1)`,不是均匀分布 |
| `t = torch.zeros(B,)` 不是 `0.0` | `time_encoder` 期望 (B,) 形状的输入 |
| **不采样 t、不造 x_t** | 训练才采样 t;推理是沿着学到的速度场从 t=0 一路积到 t=1 |
| `vlm_emb.clone()` / `proprio_emb.clone()` | `JointModel` 内部会对 embeds dict 里的 tensor 做 **in-place** 修改;循环 10 次必须每次喂新鲜的 |
| `num_inference_steps=10` | FM 推理 5-10 步就够,Diffusion 要 50-1000 步,这是 FM 最大的卖点 |

### 3.4 训练 vs 推理对照

|  | 训练 `forward` | 推理 `infer_action` |
|---|----------------|---------------------|
| 输入 | input_ids, pixel_values, mask, proprio, **action** | input_ids, pixel_values, mask, proprio |
| t 来源 | `torch.rand(B,)` 随机采样 | `0, 1/N, 2/N, ..., (N-1)/N` 离散步 |
| x_t 来源 | `(1-t)·noise + t·action` 已知 GT 构造 | 从 noise 出发,Euler 一步步推 |
| forward 次数 | 1 次 | N 次(默认 10) |
| 模型输出 | v_pred → MSE loss | v_pred → Euler step `x += dt*v` |
| 用 grad 吗 | 是 | 否(`@torch.no_grad()`) |

### 3.5 smoke test

```python
def test_pizero_infer_action_shape():
    cfg = _load_cfg()
    model = PiZero(cfg).eval()
    B = 2
    batch = _make_batch(cfg, B)   # 同训练 fixture,batch['action'] 不会被用到

    actions = model.infer_action(batch, num_inference_steps=3)

    assert actions.shape == (B, cfg.horizon_steps, cfg.action_dim)
    assert torch.isfinite(actions).all()
```

`num_inference_steps=3` 是测试 trick——只验证 shape 和数值有限,不在乎效果。

### 3.6 进阶版本(暂不实现)

原版还有个 `infer_action`(非 naive)用 **KV cache**:vlm/proprio 部分每次循环结果都一样,可以只在第一步算,后面 9 步只跑 action expert。
**第一版用 `naive` 写法重新算就行**,跑通了再考虑优化。

---

## 四、TODO

- [ ] 实现 `PiZero.infer_action()`(本笔记 §3.2 骨架)
- [ ] 在 `PiZero.__init__` 加 `self.action_dim = config.action_dim`(方便引用)
- [ ] smoke test 验证 shape + finite(本笔记 §3.5)
- [ ] (中期)对齐原版加 `flow_sig_min`:`x_t = (1 - (1-σ)·t)·noise + t·action`
- [ ] (中期)把 adaLN 接入 `MixtureDecoderLayer.forward`
- [ ] (中期)实现 `AdaptiveLayerscale`(本笔记 §1.6)完成 adaLN-Zero

---

## 五、今天的关键 takeaway

1. **训练 ≠ 推理**:训练采样 `t` + 构造 `x_t` 算 loss;推理从噪声出发 Euler 积分。
2. **`time_cond=True` 模式下,time_dim 必须等于 action_hidden_size**,因为要 cat。这个约束 `ActionEncoder` 不强制,要在外层 `PiZero.__init__` 自觉对齐。
3. **adaLN-Zero 的灵魂是 α 零初始化**:让网络从 identity 开始学,保护预训练权重 + 梯度通畅 + 自学条件强度。
4. **`in-place` 修改的 tensor 在推理循环里必须 clone**——这种坑只能靠原版注释或踩坑发现。

# Phase 1 笔记：加载 PaliGemma 权重 + bf16 + 冻结 VLM 训练

> 日期: 2026-06-24
> 目标: 把 YuePi0 从"小 fake 模型 overfit"扩展到"加载真实 PaliGemma-3B 权重，单卡 24GB 跑得动"

---

## 一、本日完成的修复 (代码 Bug)

### 1. `yuepi0.py`: 训练/推理调用 action_encoder 不一致
**问题**:  训练时 `forward` 无条件传 `time_emb`，推理时 `infer_action` 用 `if self.adaptive_mode` 分支。当前 `adaptive_mode=None` 不触发 bug，但属于"睡眠 bug"。

**修复** (yuepi0.py:142-146):
```python
time_emb = self.time_encoder(t)
if self.adaptive_mode:
    action_emb = self.action_encoder(x_t)            # adaLN: time 走 norm
else:
    action_emb = self.action_encoder(x_t, time_emb)  # 朴素: time 和 action cat
```

### 2. `yuepi0.py`: bf16 训练下 `torch.rand` 默认 fp32 污染
**问题**: 报错 `mat1 and mat2 must have the same dtype (Float vs BFloat16)`。
原因是 `t = torch.rand(B, device=action.device)` 默认 fp32，进入 `x_t = (1-(1-σ)t)·noise + t·action` 后把 x_t 提升到 fp32，再喂给 bf16 的 `action_encoder.linear_1`。

**修复** (yuepi0.py:135):
```python
t = torch.rand(B, device=action.device, dtype=action.dtype)
```

**教训**: `torch.rand/zeros/ones/arange/full` 这类"凭空创建"的 API 默认 fp32，不管你模型是什么 dtype。bf16 训练里 **凡是新建 floating tensor 都要显式传 dtype**。

### 3. `paligemma/modules.py`: GemmaRoPE 没把 cos/sin 转回输入 dtype
**问题**: 报错 `expected scalar type BFloat16 but found Float`。
原因是 RoPE 内部用 fp32 算 cos/sin（精度需要），但 return 前忘了转回 `input_dtype`，结果 q_rot 被升到 fp32，跟 bf16 的 v 在 attention 里冲突。

**修复** (modules.py:51):
```python
def forward(self, qk, position_ids):
    input_dtype = qk.dtype
    # ... fp32 计算 ...
    return cos.to(input_dtype), sin.to(input_dtype)
```

**对照**: `GemmaRMSNorm` 末尾用 `output.type_as(x)` 是正确范式；RoPE 之前写了一半。

### 4. `yuepi0.py`: 添加 `final_action_clip_value`
**目的**: 跟 open-pi-zero 对齐，推理 Euler 积分后裁剪到 `[-clip, clip]`。

**修复** (yuepi0.py __init__ + infer_action):
```python
self.final_action_clip_value = config.get("final_action_clip_value", None)
# ...
# Euler loop 之后
if self.final_action_clip_value is not None:
    x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
```

**为什么 `torch.clamp` 不可微也没问题**: 在 `@torch.no_grad()` 下没有反向；即使要可微，clamp 的零梯度区域不会引发崩溃，只是没有梯度流过被截断的元素。

---

## 二、PaliGemma 权重加载

### Key 映射
HF 的 PaliGemma safetensors 总共 **603 个 key**，分三类前缀：
- `language_model.*` (164)
- `vision_tower.*` (437)
- `multi_modal_projector.*` (2)

YuePi0 的 state_dict 总共 **938 个 key**（含 action expert / proprio expert / encoder / decoder / 各 mixture vlm 层等）。

映射规则 (`src/model/utils.py: hf_key_to_yuepi0_key`):
```
language_model.model.embed_tokens.weight      → embedder.embed_tokens.weight
language_model.model.layers.[i].*             → joint.mixtures.vlm.layers.[i].*
language_model.model.norm.weight              → 丢弃 (use_final_norm=False)
multi_modal_projector.*                       → embedder.multi_modal_projector.*
vision_tower.*                                → embedder.vision_tower.*
```

加载结果: `loaded=602, skipped=1 (norm), shape_mismatch=0, unloaded=336 (剩下都是 action/proprio expert，随机初始化训练即可)`。

### 实现细节
- 用 `safetensors.safe_open(..., framework="pt")` 逐 key 读，零拷贝 mmap，不需要 pickle。
- 用 `param.data.copy_(tensor)` 写入（`.copy_()` 是 in-place，`=` 只是改字典指针）。
- 用 `torch.device("meta")` 可以零内存窥视模型 state_dict 结构（验证脚本里就这么用）。

### 验证脚本
- `scripts/inspect_paligemma_keys.py`: 看 HF safetensors 都有什么 key
- `scripts/inspect_pizero_keys.py`: 看 PiZero state_dict 都有什么 key
- `scripts/load_paligemma.py`: 跑映射，报告 loaded/skipped/mismatch
- `scripts/test_paligemma_loaded.py`: CPU sanity check，验证加载后 forward 能跑（loss=1.4422, vlm_emb std=0.0341，后者是因为 image 经过 `/sqrt(2048)` 缩放）

---

## 三、单卡 24GB 显存预算

### 不加任何技巧 (FP32 + AdamW)
- 权重 fp32: 14.2GB
- 梯度 fp32: 14.2GB
- AdamW state (fp32 m+v): 28.4GB
- → 总计 ~50GB ❌ 完全装不下

### 第一阶段方案: **冻结 VLM + bf16 + AdamW (普通)**
- 权重 bf16: 7.1GB (全模型)
- 可训练梯度 bf16: 1.25GB (只有 626M 训练参数)
- AdamW state fp32: 2.5GB
- 激活 + workspace: 3-5GB
- → 总计 ~13-15GB ✅ 单卡 24GB 装得下，**实测峰值 7.11GB**(仍处于训练早期)

> pi-zero 论文也是冻 VLM 的，所以这步并不是退化方案，是正路。

### 冻结代码
```python
def freeze_vlm(model):
    modules_to_freeze = [
        model.embedder.embed_tokens,
        model.embedder.vision_tower,
        model.embedder.multi_modal_projector,
        model.joint.mixtures['vlm'],
    ]
    for m in modules_to_freeze:
        for p in m.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    # 实测: 626.1M / 3549.6M = 17.6%
    return trainable
```

---

## 四、bf16 训练的一些坑（汇总）

PyTorch 的 dtype 提升规则: **fp32 + bf16 → fp32**。任何"漏网"的 fp32 张量都会污染整条链路。

容易翻车的点:
1. `torch.rand/zeros/ones/arange/full` 默认 fp32（修复 #2）
2. RoPE / LayerNorm 等内部用 fp32 但忘了转回 (修复 #3)
3. AdamW state（m, v）默认 fp32 — 这个我们留着，不动
4. attention_mask 用 `torch.finfo(dtype).min` 填，dtype 必须跟 attn_scores 一致 — 这个 `build_mask_and_position_ids` 已经传 dtype 进去了

通用模式: **任何"凭空创建张量"的代码都显式传 dtype**；任何 `.float()` 升精度的内部计算都要在 return 前 `.to(input_dtype)`。

---

## 五、训练入口结构 (`src/agent/train.py`)

```
1. OmegaConf.load("config/realdataTrain.yaml") + resolve
2. dataset / loader / tokenizer / processor 准备 (跟 fakedatatrain 一样)
3. model = PiZero(config)
4. load_paligemma_weights(model, hf_path)        ← 新加
5. model = model.to(torch.bfloat16).to(device)   ← 整模型 bf16
6. trainable_params = freeze_vlm(model)          ← 新加
7. optimizer = AdamW(trainable_params, lr=...)   ← 只优化可训练参数
8. GPU memory self-check (打印峰值)
9. training while loop:
   - inputs = to_device_bf16(inputs, device)     ← 注意要传 device！
   - loss = model(inputs)
   - loss.backward(); optimizer.step()
10. eval block (overfit ratio)
```

### `to_device_bf16` 辅助函数 (utils.py)
```python
def to_device_bf16(inputs: dict, device) -> dict:
    out = {}
    for k, v in inputs.items():
        v = v.to(device)
        if v.is_floating_point():   # int64 不要转 bf16
            v = v.to(torch.bfloat16)
        out[k] = v
    return out
```

---

## 六、第一阶段成果

500 步训练 (Phase 1 PaliGemma + bf16 + frozen VLM):
- Run 1: loss 1.91 → 0.18 (144s)
- Run 2: loss 1.73 → 0.12 (136s)

跟之前的小 fake 模型 overfit (1.85 → 0.2) 数值差不多。**这是符合预期的**: fake 数据的 pixel 是随机噪声，PaliGemma 的视觉特征也提取不出语义，所以加大模型并不能让 fake-overfit 任务变更难/更易。**真正的差异要等切到 Bridge 真实数据集才会显现**。

---

## 七、训练 vs 推理为什么不一样

| 阶段 | 时间步 t | 前向次数 | 用途 |
|------|----------|----------|------|
| 训练 | 随机采一个 `t ∈ U(0,1)` | 1 次 | 算速度场预测损失 |
| 推理 | 从 t=0 到 t=1 离散化 | 10 次 (Euler) | 积分还原 action |

Flow Matching 训练只需要"在随机 t 上拟合速度场"，**不需要每个样本走完完整 10 步去噪**。10 步去噪只在推理时跑 Euler。

---

## 八、待办

- [ ] 修复 train.py eval block 的 `to_device_bf16(inputs)` 漏传 device → 改成 `to_device_bf16(inputs, device)`
- [ ] 跑完 Phase 1 完整 eval，对比新 ratio vs 旧 0.265
- [ ] Phase 2: 加 FSDP 双卡支持（CLI flag 切 `freeze_vlm` / `fsdp`）
- [ ] Phase 3 (主目标): 切到 Bridge RLDS 真实数据集

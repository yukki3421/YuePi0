# PaliGemma 权重加载

## 为什么必须加载预训练权重

```
PiZero 总参数 ~3B
  - PaliGemma 部分: ~2.8B  (vision + language)
  - action expert: ~250M
  - proprio/action/time encoder: 几百万

从零训练:
  bridge 数据集 ~2M transitions
  3B 参数从零训练需要数百 GPU 天

加载 PaliGemma 预训练:
  视觉+语言能力直接拿来用
  只训练 action expert + 微调 VLM
  10-20 小时 1 张 A100 见效
```

## 加载的本质

加载权重 ≠ 改变张量内容
加载权重 = 把 HF 的张量按"对应关系"塞到 PiZero 对应位置

不是改张量,是**改 key 名字然后 copy_ 进去**。

## HF PaliGemma 包含什么 / 不包含什么

```
HF PaliGemma 包含:
  language_model (Gemma transformer 18 层)  → VLM expert
  vision_tower (SigLIP ViT)                 → embedder.vision_tower
  multi_modal_projector                     → embedder.multi_modal_projector
  embed_tokens (词嵌入)                     → embedder.embed_tokens

HF PaliGemma 不包含:
  proprio expert       (随机初始化, 训练时学)
  action expert        (随机初始化, 训练时学)
  proprio_encoder      (随机初始化)
  action_encoder       (随机初始化)
  action_decoder       (随机初始化)
  time_encoder         (sin/cos, 没参数)
```

## Key 映射规则

```python
# 规则 1: 词嵌入
HF:    "language_model.model.embed_tokens.weight"
YuePi: "embedder.embed_tokens.weight"

# 规则 2: 18 层 transformer
HF:    "language_model.model.layers.[i].*"
YuePi: "joint.mixtures.vlm.layers.[i].*"
做法:  替换前缀

# 规则 3: 最后那个 final norm
HF:    "language_model.model.norm.weight"
YuePi: None  (你 Mixture 没 final norm, 主动丢弃)

# 规则 4: 多模态 projector
HF:    "multi_modal_projector.*"
YuePi: "embedder.multi_modal_projector.*"
做法:  加 "embedder." 前缀

# 规则 5: 视觉塔
HF:    "vision_tower.*"
YuePi: "embedder.vision_tower.*"
做法:  加 "embedder." 前缀
```

## 加载流程

```python
# 1. 读所有 safetensors shard 到一个大 dict
hf_state = {}
for shard in sorted(hf_path.glob("*.safetensors")):
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            hf_state[k] = f.get_tensor(k)

# 2. 获取 PiZero 的 state_dict (引用,不是 copy)
own_state = model.state_dict()

# 3. 逐个翻译 + 拷贝
for hf_k, tensor in hf_state.items():
    yp_k = hf_key_to_yuepi0_key(hf_k)
    if yp_k is None: continue              # 主动丢弃
    if yp_k not in own_state: raise        # 翻译错了
    if shape 不匹配: skip                  # 配置没对齐
    own_state[yp_k].copy_(tensor)          # 原地拷贝数据
```

`.copy_()` 是 in-place 数据级覆盖,会真的改 model 内部张量。
`=` 赋值不会改 model,只改 dict 引用。

## 加载后的状态检查

```text
loaded:           几百个 key (PaliGemma 主干)
skipped:          1 个 (final norm, 主动丢弃)
shape_mismatch:   理想 0 个 (有就是 config 维度对不上 HF)
unloaded:         几十个 (action/proprio expert, encoders)
                  保持随机初始化, 训练时学
```

## 训练时哪些权重应该 fine-tune / freeze

```text
策略 1: 全量 fine-tune (需要 80GB GPU)
  所有权重都训, 包括 PaliGemma 主干
  显存占用: 模型 6GB + grad 6GB + optim state 12GB ≈ 24GB+
  3090 24GB 单卡跑不动, 需要 batch_size=1 或两卡 DDP

策略 2: 只训 action expert (24GB 够)
  Freeze: embedder, joint.mixtures.vlm
  Train:  joint.mixtures.{proprio,action},
          time_encoder, action_encoder,
          proprio_encoder, action_decoder
  显存大幅降低, 但效果可能差

策略 3: LoRA fine-tune
  PaliGemma 主干加 LoRA adapter (rank=32)
  只训 adapter + action expert
  显存中等, 效果接近全量
```

## 加载验证

```python
# 加载完检查
model.eval()
with torch.no_grad():
    batch = make_dummy_batch()
    loss = model(batch)
    assert torch.isfinite(loss), "加载后 forward 应该不炸"
```

如果加载后 forward 输出 NaN/Inf,说明:
1. 某个 key 形状不匹配但没被检测到
2. 某个 key 翻译错了
3. dtype 不匹配 (fp32 vs bf16)

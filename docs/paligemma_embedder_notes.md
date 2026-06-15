# Day N+1 - PaliGemma 图文拼接

> 复现对象:`open-pi-zero/src/model/vla/pizero.py` 的 `_forward_siglip_and_text_embedding`
> 我的复现:`YuePi0/src/model/vla/yuepi0.py` 的 `PaliGemmaEmbedder`
> 笔记:`YuePi0/docs/paligemma_embedder_notes.md`

---

## 一、核心问题:图片怎么塞进文本序列里

### 1.1 答案:用特殊 token 占位

`pizero.py:36`:
```python
self.image_token_index = cfg.image_token_index  # = 257152
```

`257152` 是 `<image>` 这个特殊 token 的 id。

**图文序列的完整结构** (processing.py:24):
```
[<image>×256] [<bos>] [prompt...] [\n] [padding...]
```

注意 **BOS 和换行符 `\n` 也是固定的**,不是可选项。

### 1.2 processor 做了什么

`PaliGemmaProcessor.__call__` 的流程:

```python
# 1. 读图 → 归一化 → pixel_values [B, 3, 224, 224]
pixel_values = process_images(images, ...)

# 2. 给 prompt 前面拼接 256 个 <image> token
input_strings = [f"<image>" * 256 + "<bos>" + prompt + "\n" for prompt in text]

# 3. tokenizer 把字符串转成 input_ids
inputs = tokenizer(input_strings)
return {"pixel_values": pixel_values, **inputs}
```

**`input_ids` 里天然就包含了 `<image>` 占位符序列**,模型只需要把占位符替换成真实的图片 embedding。

---

## 二、`_forward_siglip_and_text_embedding` 逐行解析

核心代码在 `pizero.py:408-468`。分 5 步:

### Step 1: 文本 lookup (第 432 行)

```python
inputs_embeds = self.embed_tokens(input_ids)
# [B, seq_len] → [B, seq_len, hidden=2048]
# 注意:<image> 占位符位置的 embed 也会被查出来,后面会被覆盖
```

### Step 2: 图片 encode + project (第 436-438 行)

```python
selected_image_feature = self.vision_tower(pixel_values)  # [B, 256, 1152]
image_features = self.multi_modal_projector(selected_image_feature)  # [B, 256, 2048]
```

`multi_modal_projector` 是单层 `nn.Linear(1152, 2048, bias=True)` (siglip.py:33)。

### Step 3: 图片 embedding scale ⭐ (第 443 行) — 隐蔽 bug

```python
scaled_image_features = image_features / (self.image_text_hidden_size ** 0.5)
# = image_features / sqrt(2048)
```

**必须 scale**:因为 `embed_tokens` 的初始化方差 ~ 1/sqrt(2048),不 scale 的话图片 embedding 量级会远大于文本,影响 attention。漏掉 scale 是隐蔽性高的 bug——shape 测试看不出。

### Step 4: 用 pad_token_id 初始化 final_embedding (第 449-451 行)

```python
final_embedding = torch.full(
    (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
)
# trick:用 pad_token_id 填初值,后面 text_mask 和 image_mask 会覆盖所有真正使用位置,
# 剩下只有 padding 位置的值不影响后续(被 attention mask 屏蔽)
```

### Step 5: 两类 mask + 赋值 (第 455-467 行)

```python
text_mask  = (input_ids != image_token_index) & (input_ids != pad_token_id)
image_mask = (input_ids == image_token_index)

# 文本:直接布尔索引赋值,不用 for 循环
final_embedding[text_mask] = inputs_embeds[text_mask]

# 图片:for 循环逐 batch 处理(原因见第三节)
for i in range(bsz):
    image_indices = image_mask[i].nonzero(as_tuple=True)[0]
    num_image_tokens = len(image_indices)
    final_embedding[i, image_indices] = scaled_image_features[i, :num_image_tokens]
```

---

## 三、为什么图片赋值必须 for 循环,文本不用

### 文本为什么不用 for 循环

```python
inputs_embeds.shape = [B, seq_len, hidden]   # 和 input_ids shape 完全一致
final_embedding.shape = [B, seq_len, hidden]

final_embedding[text_mask] = inputs_embeds[text_mask]
# 两边 shape 相同,布尔索引后 batch 维度自动保留,不同 batch 赋值自动配对
```

### 图片为什么必须 for 循环

```python
scaled_image_features.shape = [B, 256, hidden]   # 固定 256 个 patch
final_embedding.shape        = [B, seq_len, hidden]

# final_embedding[image_mask].shape = [N, hidden]
# N = 选出的 image token 总数,可能因 batch 不同而不等于 256

# ❌ 不能写:
final_embedding[image_mask] = scaled_image_embeddings[image_mask]
# shape 可能不匹配,且丢失 batch 配对信息

# ✅ 必须 for 循环:
for i in range(bsz):
    image_indices = image_mask[i].nonzero(as_tuple=True)[0]
    final_embedding[i, image_indices] = scaled_image_features[i, :len(image_indices)]
```

---

## 四、`yuepi0.py` 复现时的 4 个 bug

### Bug 1: `pad_token_id` 定义顺序错 (已修)

```python
# ❌ 错误:在 embed_tokens 使用后才定义
self.embed_tokens = nn.Embedding(..., self.pad_token_id)  # 引用了还不存在的 self.
self.pad_token_id = cfg.pad_token_id  # 太晚了

# ✅ 正确:先定义,后使用
self.pad_token_id = cfg.pad_token_id
self.image_text_hidden_size = cfg.hidden_size
self.embed_tokens = nn.Embedding(..., self.pad_token_id, self.image_text_hidden_size)
```

### Bug 2: `text_mask` 用 Python `and` 而非 PyTorch `&` (已修)

```python
# ❌ 错误:Python and 用于标量布尔,不支持张量逐元素操作
text_mask = (input_ids != a) and (input_ids != b)

# ✅ 正确:PyTorch 逐元素按位与
text_mask = (input_ids != a) & (input_ids != b)
```

### Bug 3: 图片 embedding 赋值不用 for 循环 (已修)

```python
# ❌ 错误:布尔索引丢失 batch 配对信息
final_embedding[image_mask] = scaled_image_embeddings[image_mask]

# ✅ 正确:for 循环按 batch 逐个配对
for i in range(bsz):
    image_indices = image_mask[i].nonzero(as_tuple=True)[0]
    final_embedding[i, image_indices] = scaled_image_embeddings[i, :len(image_indices)]
```

### Bug 4: `multi_modal_projector` 命名不一致 (已修)

加载原版权重时 key 是 `multi_modal_projector.linear.weight`,所以必须叫 `self.multi_modal_projector`(不能叫 `vision_projector`)。

---

## 五、下一步

下一步要看 `joint_model.py` 的 `forward_mixture_attn`——理解 PiZero 的 MoT(Mixture-of-Tokens)架构。

---

## 六、关键文件路径

| 文件 | 作用 |
|---|---|
| `open-pi-zero/src/model/vla/pizero.py:408-468` | `_forward_siglip_and_text_embedding` 原文 |
| `open-pi-zero/src/model/paligemma/processing.py:11-24` | `add_image_tokens_to_prompt` |
| `open-pi-zero/src/model/paligemma/siglip.py:20-42` | `PaliGemmaMultiModalProjector` |
| `YuePi0/src/model/vla/yuepi0.py` | 我的复现 `PaliGemmaEmbedder` |
| `YuePi0/docs/paligemma_embedder_notes.md` | 本笔记 |

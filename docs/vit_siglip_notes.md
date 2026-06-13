# Day N - ViT / SigLIP 复现笔记

> 复现对象：`open-pi-zero/src/model/paligemma/siglip.py`
> 我的复现：`YuePi0/src/model/paligemma/vit.py`
> 测试：`YuePi0/tests/model/paligemma/test_vit.py`

---

## 一、知识点：ViT 与 SigLIP

### 1.1 ViT 是什么

**ViT = Vision Transformer**（Dosovitskiy et al., ICLR 2021）。

一句话：**把图片切成若干小块（patch），每个 patch 当成一个 token，扔进标准 Transformer Encoder。**

ViT 的核心 insight：图像不是非要用卷积处理，把它转成 token 序列后，纯 attention 也能工作得很好（前提是预训练数据足够）。

### 1.2 ViT 处理图片的标准流程

```
输入 [B, 3, H, W]
    ↓ ① patch + 嵌入：用 Conv2d(kernel=patch_size, stride=patch_size)
[B, embed_dim, H/p, W/p]
    ↓ flatten + transpose
[B, num_patches, embed_dim]   ← 现在每个 patch 是一个 token
    ↓ ② 加位置编码（可学习的 nn.Embedding）
[B, num_patches, embed_dim]
    ↓ ③ 多层 Pre-LN Transformer Block（self-attn + MLP）
[B, num_patches, embed_dim]
    ↓ ④ post_layernorm
[B, num_patches, embed_dim]   ← 输出，每个 patch 都是一个特征向量
```

### 1.3 SigLIP 是什么

**SigLIP = Sigmoid Loss for Language-Image Pre-training**（Google, ICCV 2023）。

是 CLIP 的改进版，**模型架构本质就是标准 ViT**，区别只在预训练 loss：

| | CLIP | SigLIP |
|---|---|---|
| 损失 | softmax 对比损失（batch 内归一化） | sigmoid 损失（每对图文独立判二分类） |
| Batch 依赖 | 强 | 弱 |
| 训练效率 | 一般 | 更快、更省显存 |

**关键认知：从架构和推理角度，SigLIP ≈ ViT，"Sig"只承载"用 sigmoid loss 训出来"这个历史事实。**

PaliGemma 使用 SigLIP-So400m/14 作为视觉塔，所以本项目里的 `siglip.py` 就是 ViT。

### 1.4 PaliGemma 整体结构

```
图片 [B, 3, 224, 224]
    ↓ SigLIP（vision tower）
[B, 256, 1152]
    ↓ Projector（Linear 1152→2048，对齐 LLM 维度）
[B, 256, 2048]                ← image tokens
                                          ┐
文本 input_ids [B, seq_len]               │
    ↓ embed_tokens (lookup)              │ 拼起来
[B, seq_len, 2048]            ← text tokens
                                          ┘
    ↓ Gemma（LLM）
output
```

---

## 二、ViT 的 8 个核心模块

| 编号 | 类名（我的复现） | 对应原版 | 作用 |
|---|---|---|---|
| 1 | `ViTVisionConfig` | (无) | 集中放超参的 dataclass |
| 2 | `ViTVisionEmbedding` | `SiglipVisionEmbeddings` | 图片 → patch token + 位置编码 |
| 3 | `ViTVisionAttention` | `SiglipAttention` | 标准 Multi-Head Attention（无 mask、无 RoPE） |
| 4 | `ViTMLP` | `SiglipMLP` | 两层 Linear + GELU(tanh approx) |
| 5 | `ViTEncoderLayer` | `SiglipEncoderLayer` | Pre-LN block：x = x + Attn(LN(x))，再 x = x + MLP(LN(x)) |
| 6 | `ViTEncoder` | `SiglipEncoder` | N 层 EncoderLayer 串起来 |
| 7 | `ViTVisionTransformer` | `SiglipVisionTransformer` | embed + encoder + post_layernorm，顶层 ViT |
| 8 | `ViTVisionModel` | `SiglipVisionModel` | 最外层薄壳，加 `self.vision_model` 命名空间，方便加载 HF 权重 |
| (额外) | `ImageProjector` | `PaliGemmaMultiModalProjector` | 1152 → 2048，对齐 LLM；严格不属于 ViT |

### 真实超参（来自 `bridge.yaml`）

| 参数 | 值 |
|---|---|
| `image_size` | 224 |
| `patch_size` | 14 |
| `num_patches` | (224/14)² = **256** |
| `hidden_size`（embed_dim） | 1152 |
| `num_hidden_layers` | 27 |
| `num_attention_heads` | 16 |
| `head_dim` | 1152/16 = 72 |
| `intermediate_size` | 4304 |
| `projection_dim`（输出对齐 LLM） | 2048 |

---

## 三、关键知识点深入

### 3.1 为什么 patch 用 Conv2d，不用 reshape？

把 `Conv2d(in=3, out=embed_dim, kernel=p, stride=p)` 看成 **「切 patch」+「线性投影」一步到位**：

- 卷积核张量形状 `[embed_dim, 3, p, p]`：
  - 一个核 `[3, p, p]` 覆盖输入图上 `p×p` 一块区域（3 是 RGB 通道）
  - 共有 `embed_dim` 个这样的核 → 每个 patch 输出 `embed_dim` 维特征
- `stride=p` 保证 patch 之间不重叠

数学上等价于：`flatten(每个 patch) → Linear(3·p², embed_dim)`，但写成 Conv2d 代码极简且 GPU 友好。

### 3.2 SigLIP 的 attention 没有 mask、没有 RoPE

和你之前学的 GQA / Gemma attention 不同：

| 对比项 | LLM Attention（Gemma） | ViT Attention（SigLIP） |
|---|---|---|
| Mask | 因果 mask（下三角） | 无（全双向） |
| 位置编码 | RoPE（旋转 Q/K） | 可学习 nn.Embedding（加在 token 上） |
| KV 头数 | GQA：Q 头多、KV 头少 | MHA：Q/K/V 头数相同 |
| 长度 | 变长（生成时 KV 增长） | 固定（256 个 patch） |

> 直观理解：图像中每个 patch 都该看到所有其他 patch（树叶应该看到天空、地面），不存在"未来"的概念，所以不需要 mask。

### 3.3 Pre-LN vs Post-LN

ViT 用的是 **Pre-LN**：

```python
x = x + Attn(LN(x))   # LN 在 sub-layer 前面
x = x + MLP(LN(x))
```

不是 Post-LN：

```python
x = LN(x + Attn(x))   # LN 在 sub-layer 后面（原版 Transformer）
x = LN(x + MLP(x))
```

**Pre-LN 优势**：训练更稳定，梯度传播更顺畅，深层模型不容易崩。现代 Transformer（GPT-2 之后）几乎全用 Pre-LN。

### 3.4 Attention scale = `1/√head_dim`，不是 `1/√embed_dim`

```python
self.scale = self.head_dim ** -0.5     # ✅
# self.scale = self.embed_dim ** -0.5  # ❌
```

**为什么**：Scaled Dot-Product Attention 是在每个 head **内部独立**做 `Q·Kᵀ`，每个 head 的 Q、K 维度是 `head_dim`。点积方差 ~ `head_dim`，softmax 前要除 `√head_dim` 把数值压回正常范围。

如果错用 `1/√embed_dim=1/√1152≈0.029`，scale 太小，softmax 输出过于平均（接近均匀分布），attention 几乎"看不出哪个重要"，模型学不出来。

**这是隐蔽性高的 bug**：代码不会报错，但模型训不动。

### 3.5 GELU(approximate="tanh") 的来由

GELU = Gaussian Error Linear Unit：`GELU(x) = x · Φ(x)`，其中 Φ 是标准正态 CDF。

精确版要算 `erf`，慢；论文给了 tanh 近似公式：

$$
\text{GELU}_{\text{tanh}}(x) \approx 0.5x \cdot \left[1 + \tanh\left(\sqrt{\tfrac{2}{\pi}}(x + 0.044715 x^3)\right)\right]
$$

**为什么 SigLIP 必须用 tanh 版**：Google 用 JAX 训练时默认用 tanh 近似，HF 移植权重时必须保持一致，否则 27 层叠加后输出会偏。

> 教训：复现的"逐字"细节背后通常都有训练时的具体决定，不能想当然换。

### 3.6 softmax 强转 fp32 的数值稳定性

```python
attn_weight = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
```

bf16 / fp16 训练时，softmax 的指数运算容易溢出或精度丢失。**统一在 fp32 算 softmax，算完转回原 dtype** 是 LLM 训练里的标准技巧，不是可选项。

### 3.7 `register_buffer` 的作用

```python
self.register_buffer("position_ids", torch.arange(...).unsqueeze(0), persistent=False)
```

| 性质 | `nn.Parameter` | `register_buffer` |
|---|---|---|
| 可学习（被 optimizer 更新） | ✅ | ❌ |
| 跟模型一起 `.to(device)` | ✅ | ✅ |
| 出现在 `state_dict()` | ✅ | 取决于 `persistent` |

`position_ids` 是固定的 `[0,1,...,255]`，**不能学**，但要跟模型一起搬到 GPU。所以用 buffer，不用 parameter。`persistent=False` 表示不进 state_dict（因为 arange 可重建，存盘没意义）。

### 3.8 「逐元素」vs「跨元素」操作

| 类型 | 例子 | 是否需要 `dim=` |
|---|---|---|
| 逐元素 | gelu, relu, sigmoid, +, * | ❌ |
| 跨元素聚合 | softmax, sum, mean, max, layernorm | ✅ |

判别口诀：**如果一个数的输出依赖多个其他数 → 跨元素 → 需要 dim**。

---

## 四、复现时的硬错（自己曾经踩过的）

按"会让代码崩"→"逻辑错"→"小瑕疵"排序：

### 🔴 类一：标识符 / 拼写错（最常见，最便宜）

写代码时手抖、没对照原版、靠脑子记字段名 —— 都会落到这一类。
共同特征：**Python 报 AttributeError / NameError，或者悄悄定义了一个新变量**。
解药：写完一段立刻跑一次（哪怕只是实例化），让解释器告诉你哪个名字不存在。

具体形态有 5 种：

| # | 形态 | 例子 | 后果 |
|---|---|---|---|
| 1 | 字段名记错 | `self.embed_dim = config.embed_dim`（实际叫 `hidden_size`） | 实例化时 AttributeError |
| 2 | forward 写了一半 | `def forward(self, )` | 语法错误，文件都 import 不进来 |
| 3 | 作用域用错 | `range(self.num_hidden_layers)` 写在 `__init__` 里，那时 `self` 上还没这个属性 | 实例化时 AttributeError |
| 4 | 单复数不一致 | `register_buffer("position_id", ...)` 但 forward 用 `self.position_ids` | forward 时 AttributeError |
| 5 | typo 引入新变量 | `embedddings = embeddings + pe`（三个 d）→ return 的还是没加 PE 的旧变量 | **shape 测试看不出**，allclose 必挂——最阴 |

> 教训：单复数 / 多一个字母这类错，**Python 不会帮你 catch**，只能靠
> （a）写完立刻跑、（b）把同一个名字在文件里 grep 一遍确认只出现你预期的次数。

### 🔴 类二：数值 / 算法错（最致命，最隐蔽）

代码能跑、shape 也对，**但数学错了**。这一类只能靠"跟原版 allclose"才能 catch。

#### 4.1 attention scale 错用 embed_dim

```python
self.scale = self.head_dim ** -0.5     # ✅
# self.scale = self.embed_dim ** -0.5  # ❌
```

详见 3.4。错用后 attention 分布过于平均，模型收敛慢、最终性能显著下降。

#### 4.2 softmax 后没转回原 dtype

```python
attn = F.softmax(scores, dim=-1, dtype=torch.float32)            # ❌ 留在 fp32
attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) # ✅
```

fp32 测试时看不出，**bf16 训练时**后续 matmul 的 dtype 不一致会引入微小偏差，27 层叠加后 allclose 挂。

#### 4.3 LayerNorm 漏传 eps

```python
nn.LayerNorm(d)                          # ❌ 默认 eps=1e-5
nn.LayerNorm(d, eps=config.layer_norm_eps) # ✅ 1e-6
```

差 10 倍，atol=1e-6 的 allclose 直接挂。

### 🟡 类三：风格瑕疵（不挂，但读着别扭）

- 重复赋值同一个属性：`self.dropout = config.attention_dropout` 紧接着 `self.dropout = nn.Dropout(...)`，后者覆盖前者。
- 类名 / 变量名拼写不规范（其实属于类一第 5 种 typo，但当文件里只出现一次时不会崩，只是读着丑）。

---

## 五、测试设计：三层结构

| 层 | 测什么 | 例子 |
|---|---|---|
| Shape 测试 | 输入 → 输出 shape 对不对 | `assert out.shape == (B, 256, 1152)` |
| 不变量测试 | 不依赖具体数值的数学性质 | softmax 行和 = 1；Pre-LN 残差不爆炸；位置编码确实加上去了 |
| allclose 对齐 | 跟原版逐位对比 | 把原版的 `state_dict()` 灌进 my 实现，再 forward 比 `torch.allclose` —— 这才是"我的算法 == 原版算法"的硬证据，是复现学习阶段的真正验收（不是非得用 HF 预训练权重才有意义） |

### pytest 关键技巧

- **fixture**：被 `@pytest.fixture` 装饰的函数，测试函数把它名字写进参数列表，pytest 自动注入返回值。
- **parametrize**：同一个测试用多组参数跑，自动展开成多个 case。
- **不需要 `unittest.TestCase`**：函数名 `test_` 开头 + `assert` 语句即可。

---

## 六、自我反思

### 6.1 收获

1. **ViT 远比想象简单**：理解了 patch + 位置编码 + 标准 Transformer 之后，ViT 几乎是把 LLM 的 attention 砍掉因果 mask、砍掉 RoPE 之后的"简化版"。
2. **复现练的是细节敏感度**：scale 用 `head_dim` 还是 `embed_dim`、GELU 用不用 tanh approx、softmax 要不要转 fp32 —— 这些细节单看都"无所谓"，叠在一起决定模型能不能训。
3. **测试比代码更值得花时间**：写 shape 测试 + 不变量测试的过程，逼着我重新审视每个模块的"约定"，比单纯写完代码有用得多。

### 6.2 还差什么（留到下次）

- [ ] 补 `ImageProjector`（1152 → 2048）
- [ ] 端到端 27 层 smoke test
- [ ] 让 `ViTVisionAttention.forward` 返回 `attn_weights`，写名副其实的"softmax 行和=1"测试
- [ ] （进阶）加载 HF 预训练权重，跟原版逐位对拍

### 6.3 下一步学习路线

按 `LEARNING_PLAN.md`：
- 已完成：第 4 层 SigLIP 视觉编码器
- 下一步：第 5 层 PaliGemma = SigLIP + Gemma（理解 image token / text token 拼接逻辑，对应 `pizero.py` 的 `_forward_siglip_and_text_embedding`）

---

## 七、关键文件路径

| 文件 | 作用 |
|---|---|
| `open-pi-zero/src/model/paligemma/siglip.py` | 原版 SigLIP（参照对象） |
| `YuePi0/src/model/paligemma/vit.py` | 我的复现 |
| `YuePi0/tests/model/paligemma/test_vit.py` | shape + 不变量测试 |
| `open-pi-zero/config/train/bridge.yaml` | 真实超参（vision 段） |
| `open-pi-zero/src/model/vla/pizero.py:_forward_siglip_and_text_embedding` | 下一步要看的图文拼接逻辑 |

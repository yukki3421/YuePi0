# `processing.py` 复现总结

> 你已经从 0 复现了 pi-zero 的 **VLA 输入预处理器**。
> 下面是这一阶段的完整知识地图，按 **概念 → 工程 → Python 细节** 三层组织。

## 目录

- [一、核心概念](#一核心概念)
- [二、图像预处理](#二图像预处理)
- [三、文本预处理](#三文本预处理)
- [四、Tokenizer 输出](#四tokenizer-输出)
- [五、Python / 工程经验](#五python--工程经验)
- [六、最终数据流回顾](#六最终数据流回顾)
- [七、自检 checklist](#七自检-checklist)
- [八、下一步](#八下一步pizeropy)

## 一、核心概念

### 1. 什么是 VLA Processor

机器人 VLA (Vision-Language-Action) 模型的 **入口翻译官**：把人类世界的 `(图像, 文本指令)` 翻译成模型能吃的 `(pixel_values, input_ids, attention_mask)` 张量字典。

```text
(uint8 图像, "pick up the cube")
        │
        ▼  VLAProcessor.__call__
        │
{pixel_values:   float [-1, 1],
 input_ids:      long tensor,
 attention_mask: 0/1 tensor}
        │
        ▼
   VLM 模型 forward
```

### 2. PaliGemma 的"占位符 + 后期替换"技巧

>  PaliGemma 是 Google 开发的一个 VLM（Vision-Language Model）模型，核心特点：                                      
                                                                                                                 
  架构：                                                                                                           
  - 视觉编码器：SigLIP（纯视觉 Transformer）
  - 语言解码器：Gemma（Google 的 LLM）                                                                             
  - 连接方式：视觉特征直接拼接为 token 序列，输入 LLM
                                                                                                                   
  名字含义：      
  - Pali = Pathway for Language and Image（语言与图像的路径）
  - Gemma = Google 的轻量级 LLM 系列名称                     
                                        
  主要能力：
  - 图像描述 / 问答                                                                                                
  - 目标检测（输出 <loc0042> 这样的坐标 token）
  - 图像分割（输出 <seg042> 这样的分割 token）                                                                     
  - 文档理解      

  和你当前代码的关系：
  VLAProcessor 就是参考 PaliGemma 的数据预处理流程设计的：
  - <image> token × 256 拼接图像序列
  - <loc*> / <seg*> 坐标和分割 token
  - 同样的归一化策略（ImageNet Standard）
                                                                                                                   
  PaliGemma 本身是开源的（Apache 2.0），可以在 HuggingFace 下载：google/paligemma-3b-mix
                                                                                                                   
  简单说：就是一个 Google 出品的"看图说话"模型，当前代码在复用它的预处理流程。  

> **PaliGemma 是纯 decoder-only 语言模型，不认识图像。**

解决方案：在 prompt 前面塞 N 个 `<image>` 占位符 token，模型 forward 时再把这些位置的 embedding **替换**成 vision encoder 输出的图像 patch embedding。

```text
prompt: "<image>×256 + <bos> + 用户指令 + \n"
            ↑
   占位符，forward 时被图像 embedding 覆盖
```

**为什么 N = 256？** SigLIP 把 `224×224` 图像切成 `14×14 = 256` 个 patch。

### 3. PaliGemma 的 prompt 格式约定

**顺序固定，不能错：**

```text
<image> × 256  +  <bos>  +  prompt  +  \n
```

- `<bos>` 在图像 token **之后**（标记语言部分的开始）
- 末尾 `\n` 是训练时的硬约定，漏了模型行为会漂移

## 二、图像预处理

### 1. 数据流

```text
uint8 [0, 255]  ──×(1/255)──▶  float [0, 1]  ──(x-0.5)/0.5──▶  float [-1, 1]
```

### 2. `mean = std = 0.5` 的真相

**不是 ImageNet 统计量，而是线性映射参数：**

$$
\frac{x - 0.5}{0.5} = 2x - 1
$$

效果：把 `[0, 1]` 映射到 `[-1, 1]`。这是 PaliGemma 训练时用的归一化方式（**SigLIP 风格**），不要和 ResNet 的 `[0.485, 0.456, 0.406]` 搞混。

### 3. 按通道标准化的广播技巧

图像 shape `(B, 3, H, W)`，mean/std 原始 shape `(3,)`：

```python
mean = mean[None, :, None, None]   # (1, 3, 1, 1)
```

让 mean **在 B / H / W 三个维度上广播**，只在 C 维一一对应 —— 这就是"按通道标准化"。

### 4. dtype 防御性检查

```python
assert images.dtype == torch.uint8
```

为什么必须 uint8？因为我们默认 `scale = 1/255`。如果传进来已经被别人除过 255 的 float，再除一次就 **错得离谱**。dtype 检查是最便宜的防线。

## 三、文本预处理

### 1. Tokenizer 词表扩展的 3 类 token


| Token 类别                | 注册方式             | 数量 | 用途                                      |
| ------------------------- | -------------------- | ---- | ----------------------------------------- |
| `<image>`                 | `add_special_tokens` | 1    | 占位符，decode 时会被 skip                |
| `<loc0000>` ~ `<loc1023>` | `add_tokens`         | 1024 | 把 bbox 坐标量化成 token（每个 bin 一个） |
| `<seg000>` ~ `<seg127>`   | `add_tokens`         | 128  | 把分割 mask 用 VQ-VAE 编码成 token        |

### 2. `add_special_tokens` vs `add_tokens` 的关键区别


|                                       | `add_tokens`             | `add_special_tokens`       |
| ------------------------------------- | ------------------------ | -------------------------- |
| `skip_special_tokens=True` 时自动跳过 | ❌                       | ✅                         |
| 进入`tokenizer.special_tokens_map`    | ❌                       | ✅                         |
| 适用场景                              | 模型要生成给用户看的内容 | 结构性占位符，用户不该看到 |

- **`<image>` 用 special** —— 它是给代码用的，绝不能漏到用户输出
- **`<loc> / <seg>` 用普通** —— 它们就是模型要输出的合法内容

### 3. 为什么 pi-zero 用不到 `<loc> / <seg>` 还要注册？

3 个工程理由：

1. **词表对齐**：PaliGemma 预训练权重的 embedding shape 是 `(vocab + 1152, hidden)`，少了对不上
2. **不破坏预训练**：这些 token 已有学到的语义，删掉会让其它 token id 错位
3. **留口子**：未来可能用得上

### 4. 关闭自动 BOS / EOS

```python
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False
```

因为我们 **手动** 在 prompt 里拼了 `<bos>`，避免重复。

## 四、Tokenizer 输出

### 1. 返回类型 `BatchEncoding`

- **继承自 dict**，所以可以 `["key"]` 取值、`**` 解包、`.keys()` 遍历
- 也支持属性访问 `inputs.input_ids`
- 标准字段：`input_ids` + `attention_mask`，shape 都是 `(B, max_seq_len)`，dtype `int64`

### 2. tokenize 关键参数


| 参数                   | 作用                                                    |
| ---------------------- | ------------------------------------------------------- |
| `return_tensors="pt"`  | 返回 PyTorch 张量                                       |
| `padding="max_length"` | 所有样本填充到`max_seq_len`（shape 稳定，利于编译优化） |
| `max_length=...`       | 上限长度                                                |
| `truncation=True`      | 太长就从右边截断                                        |

### 3. `attention_mask` 含义

```text
1 = 真 token（包括 <image>、<bos>、正文） → 参与 attention
0 = padding                              → 屏蔽
```

> **特别注意：** `<image>` 占位符位置 `mask = 1`，**不是 0**。
> 因为它要参与 attention（虽然 embedding 会被覆盖）。

### 4. `padding_side`


| 取值      | 用途                                       |
| --------- | ------------------------------------------ |
| `"right"` | 训练时用，padding 在末尾                   |
| `"left"`  | 生成时用，padding 在开头（不干扰右端续写） |

pi-zero 用 `right`。

## 五、Python / 工程经验

### 1. dict 解包语法

```python
output = {"pixel_values": pixel_values, **inputs}
# 等价于：
output = {
    "pixel_values":   pixel_values,
    "input_ids":      inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
}
```

**好处：** 少写代码 + 未来兼容（tokenizer 多返回字段时自动带上）。

### 2. f-string 数字格式化

```python
f"<loc{i:04d}>"     # 0 填充，宽度 4，十进制 → <loc0042>
f"<seg{i:03d}>"     # 0 填充，宽度 3，十进制 → <seg007>
```

**补零的目的：** 字符串排序 = 数字排序，调试方便。

### 3. 参数 ≥ 3 个时永远用关键字传参

```python
# ❌ 顺序错了不一定立刻报错，bug 难查
add_image_token_to_prompts(prompt, bos, max_len)

# ✅ 永远清晰
add_image_token_to_prompts(
    num_image_token=256,
    image_token="<image>",
    prompt=prompt,
    bos_token=bos,
)
```

### 4. Python 报错读法

```text
NameError: name 'truncation' is not defined. Did you mean: 'trunction'?
```

→ **反向看**：先怀疑定义那边拼错了。

### 5. assert 是最便宜的契约

```python
assert images.ndim == 4
assert images.dtype == torch.uint8
assert len(prompts) == len(images)
```

写在函数开头 = **fail fast**。错误信息越早越好查。

## 六、最终数据流回顾

> 背一遍，确保每个箭头都能讲清楚。

```text
INPUT
  text   = ["pick up the cube", "open the drawer"]    # List[str], len=2
  images = uint8 tensor (2, 3, 224, 224)

STEP 1  校验
  ✓ len(text) == len(images)
  ✓ images.dtype == uint8

STEP 2  图像预处理
  imagePreprocess: rescale(/255) → normalize((x-0.5)/0.5)
  → pixel_values: float32 (2, 3, 224, 224), 值域 [-1, 1]

STEP 3  prompt 拼接
  对每个 prompt → "<image>×256<bos>pick up the cube\n"
  → input_strings: List[str], len=2

STEP 4  tokenize
  tokenizer(input_strings,
            return_tensors="pt",
            padding="max_length",
            max_length=300,
            truncation=True)
  → BatchEncoding {
      input_ids:      int64 (2, 300),
      attention_mask: int64 (2, 300),
    }

STEP 5  组装
  return {"pixel_values": ..., **inputs}

OUTPUT
  {
    "pixel_values":   float32 (2, 3, 224, 224),
    "input_ids":      int64   (2, 300),
    "attention_mask": int64   (2, 300),
  }
  → 喂给 VLM 模型: model(**output)
```

## 七、自检 checklist

> 如果你能 **不看代码** 回答这些问题，说明 `processing.py` 真的吃透了。

**Q1. 为什么图像归一化用 `(0.5, 0.5, 0.5)` 而不是 ImageNet 的 `(0.485, 0.456, 0.406)`？**

> 这不是统计量，而是**线性映射参数**：`(x - 0.5) / 0.5 = 2x - 1`，把 `[0, 1]` 映射到 `[-1, 1]`。这是 SigLIP / PaliGemma 训练时用的归一化方式。ImageNet 的 `(0.485, 0.456, 0.406)` 是 ResNet/VGG 时代统计出来的真·均值，目的不一样，不能混用。

**Q2. mean reshape 成 `(1, 3, 1, 1)` 是为了什么？**

> 为了 **广播对齐**。图像是 `(B, 3, H, W)`，mean 原本是 `(3,)`，直接相减会广播失败。reshape 成 `(1, 3, 1, 1)` 后，mean 在 B/H/W 三个维度上自动广播，只在 C 维度一一对应 —— 这正是"按通道标准化"的语义：每个通道用自己的 mean/std。

**Q3. PaliGemma 的 prompt 格式是什么？`<bos>` 在哪里？**

> 格式固定为 `<image>×256 + <bos> + prompt + \n`。**`<bos>` 在图像 token 之后**，标记"语言部分的开始"。图像占位符是前缀，`<bos>` 不能放最前面 —— 这是训练时的硬约定，顺序错了模型行为会崩。

**Q4. 为什么 prompt 末尾必须有 `\n`？**

> 因为 PaliGemma 训练时每条 prompt 末尾都带 `\n`，模型已经把它学成 **"prompt 结束，该开始回答"** 的信号 token（id=108）。漏掉 `\n` 模型不知道 prompt 结束了，会去续写 prompt 而不是回答。预训练模型对输入格式极度敏感，差一个 token 都不行。

**Q5. `<image>` 占位符为什么用 `add_special_tokens` 而不是 `add_tokens`？**

> 两点关键区别：(1) `add_special_tokens` 注册的 token 在 `decode(skip_special_tokens=True)` 时**会被自动跳过**，不会污染用户看到的输出；(2) 会进入 `tokenizer.special_tokens_map`，方便代码语义化访问。`<image>` 是给代码用的内部占位符，必须用 special。

**Q6. `<loc>` 和 `<seg>` token 是干什么用的？pi-zero 用得上吗？为什么还要注册？**

> `<loc0000-1023>` 把 bbox 坐标量化成 token（检测任务），`<seg000-127>` 把 mask 经 VQ-VAE 编码成 token（分割任务）。pi-zero 不用这些，但**必须注册**：PaliGemma 预训练权重的 embedding shape 已包含这 1152 个 token，少了维度对不上，加载权重直接崩。

**Q7. `attention_mask` 在 `<image>` 位置是 0 还是 1？为什么？**

> **是 1**。`attention_mask` 区分的是"真 token vs padding"，不是"文本 vs 图像"。`<image>` 是真 token（虽然它的 embedding 在 forward 时会被图像 patch embedding 覆盖），文本部分必须能 attend 到这些位置才能"看到"图像。只有末尾 `<pad>` 才是 0。

**Q8. `output = {..., **inputs}` 里的 `**` 是什么意思？**

> dict 字面量里的 **解包语法**，把 `inputs` 的所有键值对"摊平"合并到 `output` 里，等价于手抄 `input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]`。好处是少写代码 + 自动兼容（tokenizer 多返回字段时自动带上），且扁平结构方便 `model(**output)` 直接喂给 HF 模型。

**Q9. 为什么要关掉 tokenizer 的 `add_bos_token`？**

> 因为我们在 `add_image_token_to_prompts` 里**手动**拼了 `<bos>`。如果不关，tokenizer 编码时会在序列开头**再加一个 `<bos>`**，结果变成 `<bos><image>×256<bos>prompt\n`，多了一个 token 导致 prompt 格式和训练时不一致，模型行为会异常。

**Q10. `padding_side="right"` 和 `"left"` 分别用在什么场景？**

> **`right`** 用于训练 / 一次性 forward：真 token 在左，padding 在右。**`left`** 用于自回归生成：padding 在左，真 token 紧贴右端，这样生成时新 token 直接拼在末尾，不会被 padding 隔断 attention。pi-zero 训练为主，用 `right`。

## 八、下一步：`pizero.py`

你已经掌握了 **输入侧** 的所有概念。`pizero.py` 是 **模型主体**，会涉及：


| 模块                            | 内容                                                |
| ------------------------------- | --------------------------------------------------- |
| **VLM (PaliGemma) backbone**    | 处理 image + text token，输出隐状态                 |
| **Action Expert**               | 从隐状态预测连续 action（不是文本）                 |
| **Flow Matching / Diffusion**   | action 是连续的，怎么生成？                         |
| **混合 KV cache**               | VLM 和 Action Expert 共享 attention 但参数独立      |
| **Block-wise causal attention** | 图像块 / prompt 块 / action 块的特殊 attention 模式 |

准备好就开始 `pizero.py` 第一步 🚀
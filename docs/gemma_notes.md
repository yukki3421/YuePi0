# Day N+2 - Gemma 语言模型复现笔记

> 复现对象：`open-pi-zero/src/model/paligemma/gemma.py`
> 我的复现：`YuePi0/src/model/paligemma/gemma.py`

---

## 一、Gemma 模块是做什么的

Gemma 是一个 **18 层 Decoder-only Transformer**，接收已查好表的 token embedding 序列，输出每个位置的 hidden state，再通过 `lm_head` 转成词表 logits（用于预测下一个 token）。

它是 PiZero VLM Expert 的核心，负责图文融合后的**文本理解与生成**。

---

## 二、5 个组件

| 类 | 职责 |
|---|---|
| `GemmaAttention` | 单层 GQA Attention + RoPE |
| `GemmaMLP` | SwiGLU 结构的两层 MLP |
| `GemmaDecoderLayer` | Pre-LN 单层：LN → Attn → 残差 → LN → MLP → 残差 |
| `GemmaModel` | 堆 18 层 DecoderLayer + 最终 LN |
| `GemmaForCausalLM` | GemmaModel + lm_head（输出 logits） |

---

## 三、数据流向

```
输入：[B, seq_len]  # raw token ids（仅 GemmaModel 自己用）
        ↓ embed_tokens
    [B, seq_len, hidden=2048]  # token embedding
        ↓ × sqrt(hidden)  # 补偿初始化方差
    [B, seq_len, hidden]
        ↓ 18 × GemmaDecoderLayer
    [B, seq_len, hidden]
        ↓ final LN
    [B, seq_len, hidden]
        ↓ lm_head (Linear 2048 → vocab_size)
输出：[B, seq_len, vocab_size]  # logits
```

**注意**：PiZero 里的 Gemma 不走 `embed_tokens`（图文 embedding 在外面已查好表），直接接收 `inputs_embeds`。

---

## 四、Pre-LN 结构（GemmaDecoderLayer）

公式：`output = x + sublayer(LN(x))`

```python
# Attention block
residual = hidden_state
x = self.input_layernorm(hidden_state)       # LN 先
x, _ = self.self_attn(x, ...)                # Attention
hidden_state = residual + x                  # 残差后加

# MLP block
residual = hidden_state
x = self.post_attention_layernorm(hidden_state)  # LN 先
x = self.mlp(x)
hidden_state = residual + x                  # 残差后加
```

**不是** `x = x + sublayer(x)`，而是 `x = x + sublayer(LN(x))`。

---

## 五、SwiGLU（GemmaMLP）

普通 MLP：`y = W_down @ gelu(W_up @ x)`

SwiGLU 多了一个门：

```python
gate = W_gate @ x        # 学一个 0~1 的数（门卫）
value = W_up @ x         # 主信息
y = W_down @ (gelu(gate) * value)
```

门控让模型自适应决定每个 token 应该让多少信息通过，比普通 MLP 表达能力更强。

---

## 六、RoPE + GQA

- **RoPE**：在 Q/K 上做旋转，引入相对位置信息（你已有 `RoPE` / `GemmaRoPE`）
- **GQA**：Q 头多（8个），K/V 头少（4个），K/V 要 `repeat_kv` 扩展到 Q 的头数

GemmaAttention 的 forward 流程：

```python
1. Q/K/V 投影 → reshape + transpose → [B, H, T, head_dim]
2. RoPE 旋转 Q 和 K
3. KV Cache（推理时用）
4. repeat_kv：K/V 从 4 头 → 8 头
5. attn_weights = Q @ K^T / sqrt(head_dim)
6. attn_weights + attention_mask
7. softmax → matmul with V
8. output projection
```

---

## 七、attention_mask 是什么

`attention_mask` 是 **[B, 1, T, T]** 的下三角矩阵：

- 左下角（包含对角线）= 0 → 能看到
- 右上角 = -inf → 看不到（因果mask）

```python
# 造法：
causal_mask = torch.full((B, T, T), 0)
causal_mask = causal_mask.triu(1).masked_fill_(torch.ones_like(causal_mask).bool(), -1e9)
causal_mask = causal_mask.unsqueeze(1)  # → [B, 1, T, T]
```
也可以用
```python
mask = torch.triu(torch.ones(T, T), diagonal=1)*(-1e9)
```
**为什么是 4 维**：要 broadcast 到 `[B, num_heads, T, T]` 去加在 attention score 上。

---

## 八、embed_tokens 为什么要乘 sqrt(hidden)

`nn.Embedding` 的权重初始化标准差是 `1/sqrt(hidden_size)`，所以输出 embedding 的值在 `~0.022` 量级，很小。

乘以 `sqrt(hidden_size)` 把方差拉回到 O(1)，这样后续 18 层才能正常训练——这是初始化补偿机制。

---

## 九、今天疑惑的点

### Q1：Decoder 为什么不叫 Encoder？

Decoder 不是看信息流方向（都是输入→输出），而是**使用场景**决定的：
- Encoder（BERT、ViT）：双向 attention，理解任务
- Decoder（LLaMA、Gemma）：单向/因果 attention，自回归生成

Decoder 内部的 Layer 本身不绑定单向或双向——mask 由外部传入。

### Q2：GemmaDecoderLayer 的 Pre-LN 为什么是 x + sublayer(LN(x))？

因为 Pre-LN 的公式就是 `output = x + sublayer(LN(x))`：
- 先对 x 做 LayerNorm
- 再过 sublayer（Attention 或 MLP）
- 最后残差相加

**不是** `LN(x + sublayer(x))`（那是 Post-LN）。

### Q3：SwiGLU 是什么？

这是 LLaMA/Gemma 的标配 MLP，有 3 个 Linear 层：gate_proj、up_proj、down_proj。

**普通 MLP**（比如 ViT 里的）：
```python
x → up_proj → gelu → down_proj → output
```

**SwiGLU**：
```python
x → gate_proj → gelu  ──┐
                        ├── multiply → down_proj → output
x → up_proj     ────────┘
```

| | 普通 MLP | SwiGLU |
|---|---|---|
| 层数 | 2 层 | 3 层 |
| 公式 | `W2 @ gelu(W1x)` | `W3 @ (gelu(W1x) * W2x)` |
| 特点 | 线性激活 | 门控：两部分相乘 |

**为什么叫"门"**：gelu 的输出在 0 附近，接近一个软阈值——决定哪些信息通过、哪些被压掉。不是非 0 即 1 的硬门，而是连续可微的软门。

---

## 十、下一步

JointModel（MoT 架构）——三个 expert 怎么通过 attention 互相交流。

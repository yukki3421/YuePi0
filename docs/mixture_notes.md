# Mixture 复现笔记

## 这一部分解决什么问题

Mixture 是一个单独的 expert。

在 π0 / MoT 里，VLM、proprio、action 三段 token 不是共用同一套 Transformer 权重，而是各自有一套 expert 权重。

但是 attention 的 softmax 不能在每个 expert 内部独立做。因为 MoT 的关键是：

```text
三个 expert 各自算 Q/K/V
然后把 Q/K/V 沿 token 维拼起来
一起做一次联合 attention
再切回各自 expert
```

所以 Mixture 本身不是一个完整的 Transformer forward。它更像是给 JointModel 使用的一组“原子操作”。

## 文件结构

```text
Mixture
├── layers: N x MixtureDecoderLayer
│   ├── self_attn: MixtureAttention
│   ├── mlp: GemmaMLP
│   ├── input_layernorm: GemmaRMSNorm
│   └── post_attention_layernorm: GemmaRMSNorm
└── norm: final GemmaRMSNorm
```

## MixtureAttention 做什么

MixtureAttention 只负责本 expert 自己的投影和局部操作：

```text
forward_q_proj:   (B, T, hidden_size) -> (B, num_heads, T, head_dim)
forward_k_proj:   (B, T, hidden_size) -> (B, num_kv_heads, T, head_dim)
forward_v_proj:   (B, T, hidden_size) -> (B, num_kv_heads, T, head_dim)
forward_rotary_emb
forward_apply_rotary_emb
repeat_kv
forward_o_proj:   (B, T, num_heads * head_dim) -> (B, T, hidden_size)
```

注意：这里没有完整 attention forward。
完整的 softmax(QK^T)V 在 JointModel 里做。

## GQA 的关键 shape

Q 的头数是 `num_heads`：

```text
q: (B, num_heads, T, head_dim)
```

K/V 的头数是 `num_kv_heads`：

```text
k: (B, num_kv_heads, T, head_dim)
v: (B, num_kv_heads, T, head_dim)
```

在真正做 attention 之前，需要 repeat：

```text
k/v: (B, num_kv_heads, T, head_dim)
  -> (B, num_heads, T, head_dim)
```

顺序是：

```text
q/k/v projection
-> RoPE 作用在 q 和 k 上
-> repeat_kv 作用在 k 和 v 上
-> 交给 JointModel cat
```

先 RoPE 再 repeat_kv。
原因是 RoPE 不依赖 head index，先在较少的 KV heads 上算更省，也更符合 GQA 语义。

## o_proj 的方向

这是最容易写反的一层。

正确方向：

```python
nn.Linear(num_heads * head_dim, hidden_size)
```

原因：attention 输出先是共享 attention space：

```text
(B, T, num_heads * head_dim)
```

然后每个 expert 的 o_proj 把它投回自己的 hidden_size，才能和 residual 相加。

如果写成：

```python
nn.Linear(hidden_size, num_heads * head_dim)
```

在 hidden_size 恰好等于 num_heads * head_dim 的测试里可能不会暴露，所以测试里最好让这两个值不相等。

## 派发器 layer_func / attn_func

Mixture 里有两个派发器：

```python
layer_func(method_name, layer_idx, *args)
attn_func(method_name, layer_idx, *args)
```

它们本质是 `getattr` 反射调用。

例如：

```python
mixture.attn_func("forward_q_proj", 0, x)
```

等价于：

```python
mixture.layers[0].self_attn.forward_q_proj(x)
```

这样 JointModel 可以在循环里统一调度多个 expert 的同名方法。

缺点是字符串写错不会被 IDE 提前发现，只会运行时报 AttributeError。

## 当前保留和砍掉的内容

保留：

```text
q/k/v/o projection
RoPE
repeat_kv
input_layernorm
post_attention_layernorm
MLP
final norm
layer_func / attn_func 派发器
```

暂时砍掉：

```text
LoRA
quantize
adaLN / time_cond
KV cache
attention softclamp
复杂 config merge
```

这些不影响 Day 8/9 的主线：先跑通 MoT joint attention。

## 当前验证

Mixture 的功能被 JointModel 测试间接覆盖：

```bash
uv run pytest /home/cxy/projects/YuePi0/tests/model/vla/joint_model.py -v
```

当前结果：

```text
4 passed
```

其中包括：

```text
forward_mixture_attn shape
forward_mixture_layers shape
JointModel.forward shape
不同 hidden_size 的 JointModel.forward shape
```

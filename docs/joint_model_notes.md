# JointModel 复现笔记

## 这一部分解决什么问题

JointModel 是 MoT 的核心调度器。

Mixture 只是单个 expert，不能独立完成完整 attention。JointModel 要做的是：

```text
VLM expert      处理 image/text tokens
proprio expert  处理 robot state token
action expert   处理 action chunk tokens

三者各自算 Q/K/V
然后在同一个 attention 里互相通信
最后再切回各自 expert
```

一句话：

```text
Mixture 负责“各自投影”
JointModel 负责“联合 attention”
```

## forward_mixture_attn 的数据流

输入：

```text
embeds_all = {
  "vlm":     (B, T_vlm, H_vlm),
  "proprio": (B, T_prop, H_prop),
  "action":  (B, T_act, H_action),
}
```

第一步：每个 expert 各自算 q/k/v。

```text
q: (B, num_heads, T, head_dim)
k: (B, num_kv_heads, T, head_dim)
v: (B, num_kv_heads, T, head_dim)
```

第二步：RoPE。

```text
q = apply_rope(q)
k = apply_rope(k)
v 不做 RoPE
```

RoPE 只作用在 q/k 上，因为 attention score 是 q 和 k 的内积，v 只是被加权求和的内容。

第三步：repeat_kv。

```text
k/v: (B, num_kv_heads, T, head_dim)
  -> (B, num_heads, T, head_dim)
```

repeat_kv 必须在 cat 之前做。因为不同 expert 的 num_kv_heads 可以不同，但 cat 之前必须都变成相同 num_heads。

第四步：沿 token 维 cat。

```text
Q = cat([q_vlm, q_prop, q_action], dim=-2)
K = cat([k_vlm, k_prop, k_action], dim=-2)
V = cat([v_vlm, v_prop, v_action], dim=-2)
```

shape：

```text
Q/K/V: (B, num_heads, T_total, head_dim)
T_total = T_vlm + T_prop + T_action
```

第五步：标准 scaled dot-product attention。

```text
attn_scores = Q @ K.transpose(-1, -2) / sqrt(head_dim)
attn_scores = attn_scores + attention_mask
attn_weights = softmax(attn_scores)
attn_output = attn_weights @ V
```

shape：

```text
attn_scores: (B, num_heads, T_total, T_total)
attn_output: (B, num_heads, T_total, head_dim)
```

第六步：reshape + split。

```text
(B, num_heads, T_total, head_dim)
-> transpose
(B, T_total, num_heads, head_dim)
-> view
(B, T_total, num_heads * head_dim)
```

然后按原来的 token 长度切回三段：

```text
vlm:     (B, T_vlm, num_heads * head_dim)
proprio: (B, T_prop, num_heads * head_dim)
action:  (B, T_act, num_heads * head_dim)
```

第七步：各自 o_proj。

```text
vlm.o_proj:     num_heads * head_dim -> H_vlm
proprio.o_proj: num_heads * head_dim -> H_prop
action.o_proj:  num_heads * head_dim -> H_action
```

forward_mixture_attn 的返回值已经回到各自 hidden_size：

```text
vlm:     (B, T_vlm, H_vlm)
proprio: (B, T_prop, H_prop)
action:  (B, T_act, H_action)
```

这样才能和各自 residual 相加。

## hidden_size 不一样为什么还能 joint attention

关键点：Joint attention 不发生在原始 hidden space，而发生在 q/k/v 投影后的共享 attention space。

原始 hidden size 可以不同：

```text
vlm:     (B, T_vlm, 256)
proprio: (B, T_prop, 128)
action:  (B, T_act, 128)
```

但是 q_proj 可以把它们投到同一个 attention space：

```text
vlm.q_proj:     256 -> num_heads * head_dim
proprio.q_proj: 128 -> num_heads * head_dim
action.q_proj:  128 -> num_heads * head_dim
```

只要投影后满足：

```text
q/k/v after repeat: (B, same_num_heads, T, same_head_dim)
```

就可以沿 T 维 cat。

真正的硬约束不是 hidden_size 相同，而是：

```text
num_heads 相同
head_dim 相同
```

hidden_size 不同由各自的 q/k/v_proj 和 o_proj 负责适配。

## attention_mask 的形状

forward_mixture_attn 里：

```text
attn_scores: (B, num_heads, T_total, T_total)
```

所以 mask 需要能 broadcast 到这个 shape。

当前最简单使用：

```text
attention_mask: (B, 1, T_total, T_total)
```

head 维是 1，会自动 broadcast 到所有 heads。

mask 是加法 mask：

```text
0 = 允许 attention
很小的负数 = 屏蔽 attention
```

当前测试里用 full causal mask。后续 Day 10 再实现 block-wise causal mask。

## forward_mixture_layers 的数据流

一层完整 decoder block 是 Pre-LN 结构：

```text
x
-> input_layernorm
-> forward_mixture_attn
-> residual add
-> post_attention_layernorm
-> MLP
-> residual add
```

对应逻辑：

```text
residual_pre_attn = x
x_norm = input_layernorm(x)
attn_out = forward_mixture_attn(x_norm)
x = residual_pre_attn + attn_out

residual_pre_mlp = x
x_norm = post_attention_layernorm(x)
mlp_out = mlp(x_norm)
x = residual_pre_mlp + mlp_out
```

输入输出都是 dict，shape 不变。

## JointModel.forward 的职责

JointModel.forward 只做外壳调度：

```text
1. 每段 embedding 乘 sqrt(hidden_size)
2. 循环跑 num_hidden_layers 次 forward_mixture_layers
3. 每个 expert 做 final norm
4. 返回三段 hidden states
```

注意不要原地修改输入的 embeds_all。应该创建新的 hidden_states dict。

正确的循环状态更新应该是：

```python
hidden_states = scaled_embeds
for layer_idx in range(self.num_hidden_layers):
    hidden_states = forward_mixture_layers(
        self.mixtures,
        attention_mask,
        position_ids_all,
        hidden_states,
        layer_idx,
    )
```

如果写成每一层都传最初的 scaled_embeds，就不是 N 层堆叠，而是重复用初始输入跑不同 layer。

## 当前测试覆盖

测试文件：

```text
/home/cxy/projects/YuePi0/tests/model/vla/joint_model.py
```

当前覆盖：

```text
1. forward_mixture_attn shape + finite
2. forward_mixture_layers shape + finite
3. JointModel.forward shape + finite
4. hidden_size 不同情况下的 JointModel.forward shape + finite
```

运行命令：

```bash
uv run pytest /home/cxy/projects/YuePi0/tests/model/vla/joint_model.py -v
```

当前结果：

```text
4 passed
```

## 仍然没有覆盖的内容

当前测试还不能证明：

```text
block-wise causal mask 正确
attention 数值和原版 allclose
梯度跨 expert 正确传播
KV cache 正确
adaLN/time_cond 正确
Flow Matching 训练/推理正确
```

这些是后续 Day 10+ 的内容。

## 易错点总结

1. q/k/v cat 的维度是 token 维：`dim=-2`，不是 hidden 维。
2. RoPE 先于 repeat_kv。
3. repeat_kv 必须在 cat 之前做。
4. v 不做 RoPE。
5. mask 是加法 mask，不是 0/1 乘法 mask。
6. softmax 建议用 fp32 算，再转回 query dtype。
7. split 后必须先过各自 o_proj，再 residual add。
8. hidden_size 可以不同，但 num_heads/head_dim 必须一致。
9. JointModel.forward 不能原地改 embeds_all。
10. 多层循环必须更新 hidden_states。

## 下一步

Day 10 建议做 block-wise causal mask。

目标是从当前简单 full causal mask，换成 π0 风格的分块 mask：

```text
VLM tokens      可以看 VLM
proprio token   可以看 VLM + proprio
action tokens   可以看 VLM + proprio + action
```

先写 mask 构造函数和可视化/断言测试，再接入 JointModel。

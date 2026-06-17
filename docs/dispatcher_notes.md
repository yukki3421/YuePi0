# Python 派发器 / 反射调用笔记

## 这篇笔记解决什么问题

在 `Mixture` 和 `JointModel` 里，有两段一开始很绕的代码：

```python
def layer_func(self, method_name: str, layer_idx: int, *args):
    args = [arg for arg in args if arg is not None]
    return getattr(self.layers[layer_idx], method_name)(*args)


def attn_func(self, method_name: str, layer_idx: int, *args):
    args = [arg for arg in args if arg is not None]
    return getattr(self.layers[layer_idx].self_attn, method_name)(*args)
```

它们叫“派发器”，本质是 Python 的反射调用：

```text
把“我要调用哪个方法”从写死的代码，变成运行时传入的字符串。
```

## 先不用派发器时怎么写

如果不用派发器，JointModel 里调用第 0 层 attention 的 q_proj 要这么写：

```python
q = mixtures["vlm"].layers[0].self_attn.forward_q_proj(x)
```

调用第 0 层 k_proj：

```python
k = mixtures["vlm"].layers[0].self_attn.forward_k_proj(x)
```

调用 action expert 第 3 层的 o_proj：

```python
out = mixtures["action"].layers[3].self_attn.forward_o_proj(x)
```

这些代码的问题是：

```text
1. 链很长：mixtures[name].layers[layer_idx].self_attn.xxx
2. name 和 layer_idx 都在循环里变化
3. 要调用的方法也经常变化：q_proj/k_proj/v_proj/o_proj/rope/repeat_kv
4. JointModel 里会充满重复的四级访问
```

所以用派发器把“访问路径”封装起来。

## getattr 是什么

这两行等价：

```python
obj.forward_q_proj(x)
```

```python
getattr(obj, "forward_q_proj")(x)
```

`getattr(obj, "forward_q_proj")` 的意思是：

```text
从 obj 这个对象身上，取出名字叫 forward_q_proj 的属性/方法。
```

取出来以后再加 `(x)`，就是调用它。

所以：

```python
method = getattr(obj, "forward_q_proj")
result = method(x)
```

等价于：

```python
result = obj.forward_q_proj(x)
```

区别是：

```text
obj.forward_q_proj(x)          方法名写死在代码里
getattr(obj, method_name)(x)   方法名来自变量 method_name
```

这就是“动态调用”。

## *args 是什么

函数定义里：

```python
def attn_func(self, method_name, layer_idx, *args):
    ...
```

`*args` 的意思是：

```text
把前面没有被 method_name 和 layer_idx 接住的所有位置参数，打包成一个 tuple。
```

例如调用：

```python
mixture.attn_func("forward_apply_rotary_emb", 0, q, cos, sin)
```

Python 会这样分配参数：

```text
method_name = "forward_apply_rotary_emb"
layer_idx   = 0
args        = (q, cos, sin)
```

函数体里再写：

```python
getattr(..., method_name)(*args)
```

这里的 `*args` 是“解包”：

```python
method(*args)
```

等价于：

```python
method(q, cos, sin)
```

所以 `*args` 有两个方向：

```text
函数定义处：*args = 打包多余参数
函数调用处：*args = 解包参数 tuple
```

## attn_func 的完整展开

这句：

```python
q = mixtures[name].attn_func("forward_q_proj", layer_idx, x)
```

等价于：

```python
q = mixtures[name].layers[layer_idx].self_attn.forward_q_proj(x)
```

展开过程：

```text
1. mixtures[name]
   取出某个 expert，例如 vlm/proprio/action

2. .attn_func("forward_q_proj", layer_idx, x)
   调用这个 expert 的 attention 派发器

3. getattr(self.layers[layer_idx].self_attn, "forward_q_proj")
   从第 layer_idx 层的 self_attn 里取出 forward_q_proj 方法

4. (*args)
   把 x 解包传进去

5. 得到 q
```

所以：

```python
mixtures[name].attn_func("forward_q_proj", layer_idx, x)
```

可以理解成：

```text
在 name 这个 expert 的第 layer_idx 层 attention 里，调用 forward_q_proj(x)。
```

## layer_func 的完整展开

这句：

```python
x_norm = mixtures[name].layer_func(
    "forward_norm",
    layer_idx,
    "input_layernorm",
    x,
)
```

等价于：

```python
x_norm = mixtures[name].layers[layer_idx].forward_norm("input_layernorm", x)
```

而你的 `MixtureDecoderLayer.forward_norm` 是：

```python
def forward_norm(self, norm_name, x):
    return getattr(self, norm_name)(x)
```

所以继续展开：

```python
x_norm = mixtures[name].layers[layer_idx].input_layernorm(x)
```

完整链路：

```text
layer_func("forward_norm", L, "input_layernorm", x)
-> layers[L].forward_norm("input_layernorm", x)
-> getattr(layer, "input_layernorm")(x)
-> layer.input_layernorm(x)
```

post attention norm 也一样：

```python
mixtures[name].layer_func(
    "forward_norm",
    layer_idx,
    "post_attention_layernorm",
    x,
)
```

等价于：

```python
mixtures[name].layers[layer_idx].post_attention_layernorm(x)
```

## 为什么这里适合用派发器

JointModel 的核心特点是：

```text
同一层里，要对多个 expert 做同样的一批操作。
```

比如：

```text
vlm      第 L 层 forward_q_proj
proprio  第 L 层 forward_q_proj
action   第 L 层 forward_q_proj

vlm      第 L 层 forward_k_proj
proprio  第 L 层 forward_k_proj
action   第 L 层 forward_k_proj
```

如果不用派发器，代码会反复出现：

```python
mixtures[name].layers[layer_idx].self_attn.forward_q_proj(x)
mixtures[name].layers[layer_idx].self_attn.forward_k_proj(x)
mixtures[name].layers[layer_idx].self_attn.forward_v_proj(x)
```

派发器把共同路径藏起来：

```python
mixtures[name].attn_func("forward_q_proj", layer_idx, x)
mixtures[name].attn_func("forward_k_proj", layer_idx, x)
mixtures[name].attn_func("forward_v_proj", layer_idx, x)
```

好处是 JointModel 更像在描述算法流程：

```text
每个 expert：
1. 算 q
2. 算 k
3. 算 v
4. 做 rope
5. 做 repeat_kv
```

而不是被很长的对象访问链淹没。

## 派发器的代价

派发器不是无脑好，它有代价。

### 1. 字符串写错，IDE 不一定能发现

例如：

```python
mixtures[name].attn_func("forward_q_porj", layer_idx, x)
```

`forward_q_porj` 拼错了。

IDE 很可能不会报错，因为它只是一个字符串。只有运行到这里时，`getattr` 才会报：

```text
AttributeError: object has no attribute 'forward_q_porj'
```

### 2. 代码跳转变差

直接写：

```python
self_attn.forward_q_proj(x)
```

IDE 可以跳到定义。

写成：

```python
attn_func("forward_q_proj", ...)
```

IDE 不一定知道这个字符串对应哪个方法。

### 3. 类型检查变弱

静态类型工具更擅长检查直接调用，不擅长检查字符串动态调用。

所以派发器适合用在“确实需要统一调度”的地方，不适合到处乱用。

## 什么时候不用派发器更好

如果只调用一次，或者代码不在循环里，直接调用更清楚：

```python
mixtures[name].layers[layer_idx].input_layernorm(x)
```

如果要在多个 expert、多层、多方法之间重复调度，派发器更合适：

```python
mixtures[name].attn_func("forward_q_proj", layer_idx, x)
```

判断标准：

```text
如果 method_name 本身就是算法的一部分，会随着流程变化，用派发器。
如果方法固定不变，直接调用也可以。
```

## 在当前代码里的两个派发器

### attn_func

用于调用 attention 内部原子方法：

```text
forward_q_proj
forward_k_proj
forward_v_proj
forward_rotary_emb
forward_apply_rotary_emb
repeat_kv
forward_o_proj
```

入口：

```python
mixtures[name].attn_func(method_name, layer_idx, *args)
```

实际访问：

```python
mixtures[name].layers[layer_idx].self_attn.<method_name>(*args)
```

### layer_func

用于调用 decoder layer 级别的方法/模块：

```text
forward_norm
mlp
```

入口：

```python
mixtures[name].layer_func(method_name, layer_idx, *args)
```

实际访问：

```python
mixtures[name].layers[layer_idx].<method_name>(*args)
```

## 一个具体例子：forward_mixture_attn

在 `forward_mixture_attn` 里：

```python
q = mixtures[name].attn_func("forward_q_proj", layer_idx, x)
k = mixtures[name].attn_func("forward_k_proj", layer_idx, x)
v = mixtures[name].attn_func("forward_v_proj", layer_idx, x)
```

等价于：

```python
attn = mixtures[name].layers[layer_idx].self_attn
q = attn.forward_q_proj(x)
k = attn.forward_k_proj(x)
v = attn.forward_v_proj(x)
```

后者更直观，前者更统一。

如果你刚开始读不懂，可以先在脑子里把派发器还原成直接调用。

## 一个具体例子：forward_mixture_layers

在 `forward_mixture_layers` 里：

```python
hidden_states_input_norm[name] = mixtures[name].layer_func(
    "forward_norm",
    layer_idx,
    "input_layernorm",
    embeds_all[name],
)
```

一步步展开：

```text
mixtures[name]
-> 第 name 个 expert

.layer_func("forward_norm", layer_idx, "input_layernorm", embeds)
-> 去第 layer_idx 层找 forward_norm 方法

forward_norm("input_layernorm", embeds)
-> 去当前 layer 里找 input_layernorm

input_layernorm(embeds)
-> 真正执行 RMSNorm
```

所以它最终就是：

```python
mixtures[name].layers[layer_idx].input_layernorm(embeds_all[name])
```

## 如何 debug 派发器报错

如果看到：

```text
AttributeError: 'Xxx' object has no attribute 'yyy'
```

按这三步排查：

### 1. 看 method_name 字符串

例如：

```python
attn_func("forward_apply_rotary_emb", ...)
```

确认 `MixtureAttention` 里真的有：

```python
def forward_apply_rotary_emb(...):
    ...
```

### 2. 看你用的是 layer_func 还是 attn_func

如果方法在 `self_attn` 里，应该用：

```python
attn_func(...)
```

如果方法在 `MixtureDecoderLayer` 里，应该用：

```python
layer_func(...)
```

例如：

```text
forward_q_proj 在 MixtureAttention 里 -> 用 attn_func
forward_norm 在 MixtureDecoderLayer 里 -> 用 layer_func
mlp 在 MixtureDecoderLayer 里 -> 用 layer_func
```

### 3. 手动展开调用链

把：

```python
mixtures[name].attn_func("xxx", layer_idx, a, b)
```

展开成：

```python
mixtures[name].layers[layer_idx].self_attn.xxx(a, b)
```

看这句是否真的存在。

## 最容易犯的错

1. 字符串拼错：`forward_apply_rotary_emb` / `forward_apply_rotary_pos_emb` 不一致。
2. 用错派发器：attention 方法却用了 `layer_func`。
3. 忘记 `*args` 会把后面所有参数打包。
4. 调用处参数顺序错了，派发器不会帮你检查语义。
5. IDE 自动补全把变量名改坏，例如 `Mixture(cfg)` 被补成奇怪的属性访问。

## 一句话总结

派发器就是：

```text
用字符串 method_name + getattr，把“调用哪个方法”推迟到运行时决定。
```

在 MoT 里它的价值是：

```text
JointModel 可以用同一套循环，统一调度 vlm/proprio/action 三个 expert 的同名原子操作。
```

读不懂时，不要硬想抽象。把它展开成直接调用：

```python
mixtures[name].attn_func("forward_q_proj", L, x)
```

等价于：

```python
mixtures[name].layers[L].self_attn.forward_q_proj(x)
```

先这样理解就够了。

# YuePi0 数值对拍（parity test）学习笔记

记录把 open-pi-zero 训练好的权重加载进 YuePi0、并验证两边前向"逐数值等价"的完整过程。
重点是搞懂对拍的逻辑：为什么要对拍、怎么对拍、卡住时怎么定位 bug。

---

## 1. 为什么要对拍

加载别人训练好的权重，分两层验证，缺一不可：

1. 权重装得进吗？—— `load_state_dict(strict=True)` 通过，`missing=0 unexpected=0`。
   这只证明：key 名字对得上、张量形状对得上。

2. 装进去算得对吗？—— strict 加载**根本不检查你的 forward 逻辑**。
   只要名字和形状对，哪怕你的 attention 写错了、norm 公式差一项，strict 照样通过。
   但推理出来的动作会是错的。

所以光有第 1 层是危险的假象。对拍就是补上第 2 层：
**喂同一组输入给原版和 YuePi0，比较两边输出是否一致。**

一致（差异到 1e-6 浮点噪声级）才说明：你复现的前向计算和原版逐数值等价。

> 一句话：strict 加载证明"形状对"，对拍证明"算得对"。

---

## 2. 对拍的三个核心设计

### 2.1 为什么必须拆成两个进程

原版 open-pi-zero 和 YuePi0：
- 类名都叫 `PiZero`（同名，同进程 import 会冲突）；
- 依赖不同（原版要 hydra/simpler 那套），各自在不同的 venv 里。

硬塞进一个进程会 import 打架。所以拆成两步，用磁盘文件中转：

- **进程 A**（原版 venv）：造输入 → 跑原版 → 把"输入 + 输出"存到 `/tmp/parity_io.pt`
- **进程 B**（YuePi0 venv）：读同一份输入 → 跑 YuePi0 → 比对输出

```
进程A (原版venv)                    进程B (YuePi0 venv)
  造随机输入                            读 /tmp/parity_io.pt
  跑 infer_action_naive      ──磁盘──>  跑 YuePi0.infer_action
  存 输入+噪声+输出 action_ref          比 action_yue vs action_ref
```

### 2.2 为什么噪声要"存盘传过去"，不能靠随机种子

Flow Matching 推理从**纯高斯噪声**出发，逐步去噪。两边内部都会 `torch.randn` 采初始噪声。
要让结果可比，两边必须用**同一份**噪声。

靠种子（`torch.manual_seed`）对齐很脆弱：
- 种子要求"在 randn 那一刻，两边消耗随机数的历史完全一致"。两边代码路径不同，
  上游任何一次多/少调用 randn，后面的噪声就整体错位。
- CPU 和 CUDA 是两套独立的随机发生器，种子一样数也不同。
- dtype（float32/bf16）不同，生成路径也可能不同。

稳的做法：**进程 A 把它实际用的噪声 tensor 直接存盘，进程 B 读这份 tensor 喂进去。**
两边用的是同一块内存里的同一组数，100% 对齐，永不漂移。

代价：给两边的 `infer_action` 各加一个可选参数 `noise=None`：

```python
if noise is None:
    x = torch.randn(B, self.num_action_tokens, self.action_dim, device=device, dtype=dtype)
else:
    x = noise.to(device=device, dtype=dtype).clone()   # .clone() 很重要，见下
```

> `.clone()` 的原因：去噪循环里有 `x = x + dt * v`（或原版 `action += ...`）这类原地/复用操作。
> 如果直接用传进来的 noise，复用同一份 noise 跑两次时会被改掉。clone 切断共享。

### 2.3 其它必须对齐的点

- **对 naive 版，不对 cached 版**：YuePi0 的 `infer_action` 是无 KV cache 的朴素版，
  所以要对原版的 `infer_action_naive`（同样无 cache），不能对带 cache 的 `infer_action`。
- **float32 + CPU**：对拍要精度，关掉 bf16 和 GPU 的非确定性。
- **配置项对齐**：像 `final_action_clip_value` 这种会改变输出的开关，两边要一致
  （都 None 或都同值），否则一边 clamp 一边不 clamp，没法比。

判定阈值：`max_abs_diff < 1e-3` 算 PASS（真正等价时通常看到 ~1e-8）。

---

## 3. 脚本结构

两个脚本在 `scripts/` 下：

### `parity_A_reference.py`（原版 venv 跑）

```bash
cd /home/cxy/projects/open-pi-zero && \
VLA_LOG_DIR=/tmp/vla_log VLA_DATA_DIR=/tmp/vla_data TRANSFORMERS_CACHE=$HOME/.cache/huggingface/hub \
.venv/bin/python /home/cxy/projects/YuePi0/scripts/parity_A_reference.py
```

干的事：
1. 用 `config/eval/bridge.yaml` build 原版模型，`load_state_dict(strict=True)`（原始命名）。
2. 造一组随机但合法的输入：`input_ids`（前 256 个填图像占位符、末尾留 padding）、
   `pixel_values`、`attention_mask`、`proprios`，以及初始 `noise`。
3. `build_causal_mask_and_position_ids` 造 mask 和 position_ids。
4. 调 `infer_action_naive(..., noise=noise)`，得到 `action_ref`。
5. 把"原始输入 + noise + action_ref"存到 `/tmp/parity_io.pt`。

> 注意：存的是**原始输入**（不存 mask），因为两边各自 build mask，这样顺便验证两边 build_mask 是否等价。

### `parity_B_compare.py`（YuePi0 venv 跑）

```bash
cd /home/cxy/projects/YuePi0 && PYTHONPATH=src .venv/bin/python scripts/parity_B_compare.py
```

干的事：
1. `load_pretrained_pizero(cfg, ckpt)` 拿到装好原版权重的 YuePi0。
2. 读 `/tmp/parity_io.pt`，按 YuePi0 的 batch 字典格式组装输入
   （注意 YuePi0 用单数 key `proprio`，原版是 `proprios`）。
3. 调 `YuePi0.infer_action(batch, noise=noise)`，得到 `action_yue`。
4. 算 `max abs diff` / `mean abs diff`，< 1e-3 判 PASS。

---

## 4. 关键打法：组件隔离对拍（卡住时怎么定位 bug）

这是整个过程最值得学的一招。

**症状**：端到端 diff 卡在某个常数不动（比如一直 1.03），而且输出全饱和在 clamp 值（±1）。
**陷阱**：最终输出被 `clamp(-1, 1)` 截断，速度场已经爆炸到饱和，所以你每改一处的效果
都被 clamp 吃掉，diff 看起来纹丝不动——你**无法判断某个修复到底有没有用**。

**正确做法：别再"改一处跑一次"地猜，改成二分法按组件隔离。**

1. **先切最大的块**。pi0 里最复杂的是 `joint`（18 层 transformer）。
   给它喂**完全相同**的随机 `embeds` + `mask` + `position_ids`（从磁盘读，两边一致），
   只跑一次 forward，比 joint 输出。
   - 这一步很快（不用跑 10 步去噪循环）。
   - 两边 joint 权重来自同一 ckpt（Phase A 已验证一致），所以输出**必须**一致。
2. **一刀定位**：
   - joint 输出 0 diff → bug 在 joint **外面**（embedder / encoder / 迭代循环 / mask 构造）。
   - joint 输出有差异 → bug 在 joint **内部**，再递归切它的子块。
3. 缩小到几个小模块后，逐行读原版源码和自己的对比。

这次的实际结果：joint 隔离对拍 **diff = 0.000（逐位相同）**，一刀把范围从"整个模型"
切到"joint 外的几个小组件"，省掉在 18 层里盲找。

隔离脚本：`scripts/parity_joint_A.py` / `parity_joint_B.py`，逻辑同上，只是把对拍对象
从整个 `infer_action` 换成单个 `joint.forward`。

---

## 5. 本次实际找到并修掉的 bug 清单

按"主犯程度"排，全部在 joint 外面（所以 joint 隔离才 0 diff）：

1. **ActionEncoder 里 time / action 的 cat 顺序反了**（主犯）
   - 原版：`cat([time_emb, action_emb], dim=-1)`（time 在前）
   - 错误写法：`cat([action_emb, time_emb], dim=-1)`（action 在前）
   - 后果：`linear_2` 权重是 (1024, 2048)，前 1024 列对应 time、后 1024 列对应 action，
     是按这个顺序训练出来的。cat 反了 = 把 time 的权重拿去乘 action，从第一步就错，
     10 步迭代后彻底偏掉、饱和到 ±1。

2. **TimeEncoder 正弦频率公式分母差一**
   - 原版：`log(max_period) / (half_dim - 1)`
   - 错误：`log(max_period) / half_dim`
   - 后果：每个频率档位整体偏移，time embedding 数值不对。

3. **缺 Gemma attention soft-capping**
   - 原版在加 mask + softmax 之前：`attn = tanh(attn / 50.0) * 50.0`（softclamp=50）。
   - 作用：把注意力 logits 用 tanh 软压到 (-50, 50)，防止个别打分冲到极大导致 softmax 独占。
     小值近似线性几乎不变，只削极端大值。这是 Gemma2 固定组件，ckpt 带着它训练。
   - 顺序坑：softclamp 必须在**加 mask 之前**。否则 mask 的 -inf 会被 tanh 压成 -1，屏蔽失效。

4. **action expert 的 RoPE theta 配错**
   - 原版 `action_expert_rope_theta = 10000.0`，错配成 100.0。
   - 后果：base 差 100 倍，RoPE 旋转频率全错，action/proprio 段位置编码错位，attention 彻底乱。

5. **哑 config 字段**（不是 bug，是雷）
   - `time_max_period: 100.0` 写在 yaml 里，但 TimeEncoder 构造时没传这个参数，
     用的是代码里的默认值 10000（歪打正着对上）。"看着是 100，实际是 10000"。
   - 修法：把字段值改对（10000.0）并真正传进 TimeEncoder，消除"配置和实际不一致"的雷。

> 教训：2、3、4 三处的修复效果一开始全被 clamp 饱和掩盖（diff 不动），
> 直到 joint 隔离对拍才确认 joint 内部其实已经对了，逼出真正的主犯（第 1 条在 joint 外）。

---

## 6. 复现这次实验的环境坑（open-pi-zero 侧）

- 用 `.venv/bin/python` 直接跑，**不要用 `uv run`**（uv 会按 pyproject 重新同步，
  卸掉手动装的包，比如 sapien 的 pkg_resources 需要的 setuptools<81）。
- 裸 build 原版 PiZero（不碰 SimplerEnv）需要：
  - 环境变量 `VLA_LOG_DIR` / `VLA_DATA_DIR` / `TRANSFORMERS_CACHE`
  - 注册一个假的 hydra `now` resolver，否则 OmegaConf 解析 log_dir 里的 `${now:...}` 会报错：
    ```python
    OmegaConf.register_new_resolver("now", lambda fmt="": "na", replace=True)
    OmegaConf.resolve(cfg)   # 再构造模型
    ```
- 原版 checkpoint：`open-pi-zero/checkpoints/bridge_beta_step19296_*.pt`
- 原版配置：`config/eval/bridge.yaml`

---

## 7. 最终结果

```
[B] max abs diff = 5.960e-08
[B] mean abs diff = 2.348e-08
[B] PASS ✅  两边前向数值等价
```

5.96e-08 是 float32 浮点噪声级别，等于逐位相同。

**结论**：YuePi0 与 open-pi-zero 的完整前向逐数值等价。不只是"权重装得进"，
而是"装进去算出来的动作和原版一模一样"。模型这条线闭环，可以放心进 SimplerEnv 部署——
后面若出问题，必定是环境对接/反归一化，不是模型。

# Bridge V2 数据集笔记

> 日期: 2026-06-25
> 目标: 搞清楚 Bridge V2 数据集里有什么、是什么格式、每个字段是什么意思，以及后续 YuePi0 dataloader 应该怎么取数据。

---

## 一、核心概念

### 1. episode / trajectory / demo 是什么

在 Bridge 这类机器人数据集中：

- **episode**：一次完整任务的录制。
- **trajectory**：轨迹，基本等同于 episode。
- **demo / demonstration**：人类遥操作演示，也基本等同于 episode。

例子：

> 人遥控机械臂完成任务 "put small spoon from basket to tray"。
> 从开始录制到任务完成停止，这整段就是一条 episode。

一条 episode 不是一张图，而是一串 step：

```text
episode = [step0, step1, step2, ..., step(T-1)]
```

每个 step 是一个时间点的快照，里面有：

```text
step = {
    image,      # 当前看到的图像
    state,      # 当前机器人状态
    action,     # 当前/下一控制周期的专家动作
    language,   # 任务语言指令
    ...
}
```

Bridge 数据是 5Hz 左右采样，可以粗略理解为每 0.2 秒一个 step。

---

### 2. 什么叫 episode 不等长

不同任务、不同执行速度、不同操作员导致每条 episode 的 step 数不同。

例如：

```text
episode 0: put small spoon from basket to tray   -> 28 steps
episode 1: open the drawer                       -> 53 steps
episode 2: pick up the banana                    -> 17 steps
```

这就是 **episode 不等长**。

不等长是机器人数据的自然属性，因为真实任务完成时间不可能都一样。

这也是为什么数据不用简单的 numpy array 存成 `(N, T, ...)`：

- 如果所有 episode 都强行 pad 到最长长度，会浪费大量空间；
- 如果不 pad，普通 numpy array 无法直接表达变长序列；
- 所以 Bridge 使用 TFDS / RLDS / TFRecord 这种能表达变长 episode 的格式。

---

### 3. step、action、action chunk 的关系

一条 episode 里有一串 action：

```text
actions = [a0, a1, a2, ..., a(T-1)]
```

单个 action 是一个 7 维向量：

```text
a_t.shape = (7,)
```

**action chunk** 是从 episode 里截出来的一小段连续动作。

如果 `action_horizon = 4`，在时间 t 取样时：

```text
输入: image[t], state[t], language
标签: [action[t], action[t+1], action[t+2], action[t+3]]
```

这个 `[action[t], ..., action[t+3]]` 就是 action chunk，形状：

```text
(4, 7)
```

如果 batch size 是 64，那么 action batch 的形状是：

```text
(64, 4, 7)
```

含义：64 个样本；每个样本预测 4 个连续动作；每个动作 7 维。

---

## 二、文件格式

Bridge V2 以 TFDS / RLDS 格式存储。

本地目录：

```text
/home/cxy/datasets/bridge_dataset/1.0.0/
├── dataset_info.json
├── features.json
├── action_proprio_stats_*.json
└── bridge_dataset-train.tfrecord-00000-of-01024
```

### 1. TFRecord shard

`bridge_dataset-train.tfrecord-00000-of-01024` 是 train split 的第 0 个 shard。

- `00000`：当前 shard 编号；
- `01024`：train split 总共有 1024 个 shard；
- 每个 shard 是一个二进制 TFRecord 文件；
- 每个 shard 里大约有几十条 episode；
- episode 不是单独的文件，而是作为二进制 record 打包在 tfrecord 里。

逻辑结构：

```text
bridge_dataset-train.tfrecord-00000-of-01024
├── record 0 -> episode 0 -> steps[0..27]
├── record 1 -> episode 1 -> steps[0..52]
├── record 2 -> episode 2 -> steps[0..16]
└── ...
```

TFRecord 文件不能直接用文本方式查看，需要通过 TFDS 根据 `features.json` 的 schema 解码。

---

### 2. features.json 的作用

`features.json` 描述数据 schema，即每个 record 里面有哪些字段、字段类型和 shape。

顶层结构只有两个字段：

```text
{
    "episode_metadata": {...},
    "steps": Dataset(...)
}
```

其中：

- `episode_metadata`：一条 episode 只有一份；
- `steps`：变长 step 序列。

---

## 三、字段含义

### 1. episode_metadata

每条 episode 的元信息。

| 字段 | 类型 | 含义 |
|---|---|---|
| `episode_id` | int32 | episode 在原始文件中的编号 |
| `file_path` | string | 原始数据路径，能看出场景和任务 |
| `has_language` | bool | 是否有语言指令 |
| `has_image_0/1/2/3` | bool | 对应相机视角是否存在 |

注意：实际检查时发现 `has_image_X` flag 可能不可靠。例如 metadata 里可能写 `has_image_0=False`，但 `image_0` 实际不是 dummy。因此后续判断相机是否有效不能完全依赖这个 flag，可以检查图像像素均值是否接近 0。

---

### 2. steps.action

形状：

```text
(7,) float32
```

它是专家控制指令，即机器人应该执行的动作。

实际语义可理解为：

| 维度 | 含义 |
|---|---|
| 0 | Δx，末端 x 方向增量 |
| 1 | Δy，末端 y 方向增量 |
| 2 | Δz，末端 z 方向增量 |
| 3 | Δroll，末端姿态 roll 增量 |
| 4 | Δpitch，末端姿态 pitch 增量 |
| 5 | Δyaw，末端姿态 yaw 增量 |
| 6 | gripper 指令，通常接近 0/1 |

`features.json` 里的 description 写着类似 "joint velocities"，但结合 Bridge 常见格式和统计量看，实际更应该按末端 delta + gripper 理解。

从 stats 看，action 第 6 维：

```text
min = 0.0
max = 1.0
mean ≈ 0.59
std ≈ 0.49
```

这明显是二值/近二值的 gripper 指令。

---

### 3. steps.observation.state

形状：

```text
(7,) float32
```

它是机器人当前状态，也叫 proprio / proprioception / 本体感知。

可理解为：

| 维度 | 含义 |
|---|---|
| 0 | 末端 x 位置 |
| 1 | 末端 y 位置 |
| 2 | 末端 z 位置 |
| 3 | 末端 roll |
| 4 | 末端 pitch |
| 5 | 末端 yaw |
| 6 | 当前 gripper 开合程度 |

state 是当前实际观测到的机器人状态；action 是控制器收到的专家动作指令。

---

### 4. action 和 state 差值的关系

因为 action 的前几维是增量，直觉上可能认为：

```text
action[t] = state[t+1] - state[t]
```

但在真实机器人数据里不能简单等号。

更准确的说法：

```text
action[t] 是控制指令，表示想让机器人怎么动；
state[t+1] - state[t] 是实际运动结果，表示机器人最后真的动了多少。
```

它们相关，但不完全相等，原因包括：

1. action 是命令，state 差值是执行结果；
2. 真实机器人有控制延迟、摩擦、限幅、碰撞和传感器噪声；
3. 姿态角不能简单相减，存在 wrap-around 问题；
4. gripper action 是指令，state gripper 是实际开合程度，不是差分关系；
5. action 与 observation 的时间对齐可能差一个 timestep。

训练时应该使用数据集提供的 `action` 作为监督标签，而不是从 state 差分重新构造 action。

---

### 5. steps.observation.image_0/1/2/3

每个图像字段形状：

```text
(256, 256, 3) uint8
```

含义：不同相机视角。

通常：

- `image_0`：主要第三人称视角，后续优先使用；
- `image_1`：另一个固定相机视角；
- `image_2`：有些 episode 中可能有效，有些是 dummy；
- `image_3`：很多时候是 dummy，全黑或全 0。

虽然 TFRecord 里物理上以 JPEG 压缩形式存储，但通过 TFDS 解码后拿到的是 `(256,256,3)` 的 uint8 数组。

---

### 6. language_instruction 和 language_embedding

`language_instruction`：自然语言任务指令，例如：

```text
"put small spoon from basket to tray"
```

同一条 episode 的每个 step 里通常重复存同一句 language instruction。

`language_embedding`：512 维 Universal Sentence Encoder embedding。

YuePi0 / PaliGemma 训练中更应该使用 `language_instruction` 文本本身，让 PaliGemma tokenizer 和 language model 处理，而不是直接使用这个 USE embedding。

---

### 7. is_first / is_last / is_terminal / reward / discount

这些是 RLDS 标准字段。

| 字段 | 含义 |
|---|---|
| `is_first` | 是否 episode 第一步 |
| `is_last` | 是否 episode 最后一步 |
| `is_terminal` | 是否终止状态 |
| `reward` | 奖励，演示数据里通常只在最后一步为 1 |
| `discount` | RL 折扣因子，通常为 1 |

模仿学习中主要使用 image/state/language/action，这些 RL 字段暂时不是重点。

---

## 四、归一化统计

目录中有 `action_proprio_stats_*.json`，里面保存全量数据上的统计量：

```text
action: mean / std / min / max
proprio: mean / std / min / max
```

其中 proprio 对应 observation.state。

示例统计：

```text
action std ≈ [0.0097, 0.0133, 0.0124, 0.0282, 0.0302, 0.0959, 0.4916]
```

不同维度量级差别很大，例如：

- xyz delta 量级约 0.01；
- yaw delta 量级约 0.1；
- gripper 量级约 0.5。

如果不归一化，loss 会被大量级维度主导，小量级维度难学。

后续训练应对 action 和 proprio 做 z-score：

```text
x_norm = (x - mean) / std
```

训练时：

- state/proprio 作为输入，使用归一化值；
- action 作为监督标签，也使用归一化值；
- 推理或真机执行前需要反归一化：

```text
x = x_norm * std + mean
```

---

## 五、训练采样方式

训练时不是一次喂完整 episode，而是从 episode 中采样时间点。

一个训练样本大致是：

```text
sample = {
    image: image_0[t],
    proprio: state[t],
    language: language_instruction,
    actions: action[t : t + action_horizon]
}
```

如果 `action_horizon = 4`：

```text
actions.shape = (4, 7)
```

batch size 为 64 时：

```text
batch["actions"].shape = (64, 4, 7)
```

### batch 里的样本来源

一个 batch 里的 64 个样本不应该都来自同一条 episode。

正确做法是从很多 episode、很多时间点随机采样，使 batch 尽量近似 IID：

```text
sample 0: episode 5,  step 10
sample 1: episode 47, step 2
sample 2: episode 12, step 31
sample 3: episode 89, step 7
...
```

如果一个 batch 全来自同一条 episode 的连续帧，样本高度相关，梯度会被单一场景主导，不利于泛化。

---

## 六、action_horizon 边界问题

假设一条 episode 只有 5 步：

```text
step index: 0, 1, 2, 3, 4
```

如果 `action_horizon = 4`，从 `t = 3` 开始取：

```text
想取: action[3], action[4], action[5], action[6]
实际: action[3], action[4] 存在，action[5], action[6] 越界
```

解决方式常见有两种：

1. 跳过接近末尾的样本；
2. padding，用最后一个 action 重复填充。

例如：

```text
[action[3], action[4], action[4], action[4]]
```

同时可以配合 mask，告诉 loss 哪些位置是真实动作、哪些位置是 padding。

类似地，如果使用过去若干步作为条件，开头越界时可以用第一帧重复填充。

---

## 七、Bridge 数据到 YuePi0 batch 的概念映射

后续 dataloader 的目标不是直接把 episode 整条喂给模型，而是把 Bridge 的 RLDS episode 转成 YuePi0 训练需要的 batch。

概念上：

```text
Bridge RLDS:
  episode -> steps -> image_0/state/action/language

YuePi0 batch:
  image       -> PaliGemma vision input
  input_ids   -> language_instruction tokenize 后的文本 token
  proprio     -> state 归一化后
  actions     -> action chunk 归一化后
```

预期 batch 形状：

```text
images:    (B, 1, 3, 224, 224)    # image_0 resize 256 -> 224
input_ids: (B, L)
proprio:   (B, 1, 7)
actions:   (B, 4, 7)              # action_horizon = 4
```

---

## 八、目前的理解结论

1. Bridge 的基本单位是 episode，一条 episode 是一次完整任务演示。
2. episode 内部是变长 step 序列，不同 episode 的 step 数不同。
3. 每个 step 包含图像、机器人状态、动作、语言等字段。
4. action 是专家控制指令，不应简单从相邻 state 差分生成。
5. action chunk 是连续多个 action 的片段，例如 horizon=4 时形状是 `(4,7)`。
6. 一个 batch 应从多个 episode/多个时间点随机采样，而不是来自同一条 episode。
7. action 和 proprio 需要使用官方统计做 z-score 归一化。
8. 后续写 dataloader 前，应先继续理解 open-pi-zero 的 RLDS 处理和 resize/transform 逻辑。

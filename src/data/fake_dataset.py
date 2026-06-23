"""
FakeBridgeDataset —— 模拟 BridgeData V2 真实数据集的【RAW】格式
=================================================================

真实 Bridge 一条 transition 长这样 (RLDS pipeline 读出):
    image_primary:        uint8   (1, H, W, 3)            ← 主摄像头, HWC, 未归一化
    proprio:              float32 (1, 7)                  ← 当前关节状态
    action:               float32 (1, horizon, 7)         ← 未来 4 步动作
    language_instruction: bytes   b"put the spoon..."     ← 原始文本

数据层【不做任何模型相关处理】:
    - 不 tokenize 文本     -> processor 干的
    - 不归一化图像         -> processor 干的
    - 不加 image 占位符    -> processor 干的
    - 不算 attention_mask  -> tokenizer 干的

DataLoader 用 PyTorch 默认 collate 即可: 张量自动 stack, str 自动变成 List[str].
processor 在训练循环里手动调 (对齐原版 train.py 的 preprocess_batch).
"""

import torch
from torch.utils.data import Dataset


class FakeBridgeDataset(Dataset):
    """模拟 Bridge raw 数据集. 输出与真实 Bridge 字段、dtype、shape 一致"""

    # 有代表性的 Bridge 语言指令样本
    SAMPLE_INSTRUCTIONS = [
        "put the spoon in the pot",
        "pick up the red cube",
        "open the microwave",
        "move the towel to the sink",
        "stack the blue block on the red block",
    ]

    def __init__(self, cfg, num_samples: int = 100, seed: int = 42):
        """
        Args:
            cfg:         OmegaConf 配置
            num_samples: 数据集中样本数. overfit 测试设 1
            seed:        固定随机种子, 保证每次跑 dataset[i] 拿到一样的样本
        """
        super().__init__()
        self.num_samples = num_samples

        # 真实 Bridge 的几何尺寸
        self.image_size = cfg.vision_config.image_size  # 224
        self.cond_steps = cfg.cond_steps                # 1   window/history
        self.horizon_steps = cfg.horizon_steps          # 4
        self.action_dim = cfg.action_dim                # 7
        self.proprio_dim = cfg.proprio_dim              # 7

        # 预生成所有样本, 保证 dataset[i] 每次返回相同数据 (overfit 必需)
        gen = torch.Generator().manual_seed(seed)
        self._samples = [self._generate_one(gen, i) for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._samples[idx]

    # -----------------------------------------------------------------
    def _generate_one(self, gen: torch.Generator, idx: int) -> dict:
        """造一条 raw 样本"""

        # ====== 1) image: HWC uint8, 0~255 ======
        # 真实 Bridge 是 PIL 读出的图. 用随机像素模拟
        image = torch.randint(
            0, 256,
            (self.cond_steps, self.image_size, self.image_size, 3),
            dtype=torch.uint8, generator=gen,
        )

        # ====== 2) proprio: 关节状态, Bridge 归一化到 [-1, 1] ======
        proprio = torch.rand(
            self.cond_steps, self.proprio_dim, generator=gen,
        ) * 2 - 1

        # ====== 3) action: 未来 horizon 步真实动作, [-1, 1] ======
        action = torch.rand(
            self.horizon_steps, self.action_dim, generator=gen,
        ) * 2 - 1

        # ====== 4) text: 字符串 (真实 Bridge 是 bytes, 我们直接给 str) ======
        text = self.SAMPLE_INSTRUCTIONS[idx % len(self.SAMPLE_INSTRUCTIONS)]

        return {
            "image":   image,    # (1, 224, 224, 3) uint8
            "proprio": proprio,  # (1, 7) float32
            "action":  action,   # (4, 7) float32
            "text":    text,     # str
        }

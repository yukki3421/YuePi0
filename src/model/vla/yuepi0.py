from typing import Optional, Tuple

import torch
from torch import nn
from omegaconf import OmegaConf

from model.paligemma.vit import ViTVisionModel,ImageProjector
from model.vla.joint_model import JointModel
from model.vla.modules import TimeEncoder, ActionEncoder, ActionDecoder, ProprioEncoder
class PaliGemmaEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_token_index = cfg.image_token_index # 图像占位token的ID
        self.pad_token_id = cfg.pad_token_id # padding token的ID, 用于补齐到最长

        # Pizero的一次Forward里不是只有一段token,
        # 而是分成三段 [ 图像+文本 tokens ][ 机器人状态 tokens ][ 动作 tokens ]
        self.image_text_hidden_size = cfg.hidden_size

        #  Gemma 语言模型的词嵌入层
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )

        self.vision_tower = ViTVisionModel(cfg.vision_config)
        self.multi_modal_projector = ImageProjector(cfg.vision_config)


    """
        把 [图像, 文本] 两路输入编码成统一的 embedding 张量，作为 VLM expert 的输入。

        关键步骤：
            1) input_ids 里早已被 processor 预留了 num_image_tokens 个 <image> 占位符
            2) 文本走 self.embed_tokens（普通 lookup）
            3) 图片走 ViT -> projector，输出和文本同维度（image_text_hidden_size，例如 2048）
            4) 把图片 embedding 按位置塞回 final_embedding 的 <image> 占位符位置

        输入：
            input_ids:    [B, seq_len]  其中含有 image_token_index 占位符 + 实际文本 + padding
            pixel_values: [B, 3, 224, 224]
        输出：
            final_embedding: [B, seq_len, hidden]   一段同时含图、文、padding 的 embedding 序列
        """
    def forward(
            self,
            input_ids: torch.LongTensor, # 有三类token: 图片占位符*256 + bos + 文本prompt产生的token + padding token
            pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # 1) 文本 embedding lookup（图片占位符这里也被lookup了)
        image_text_embeddings = self.embed_tokens(input_ids)

        # 2) 图片embedding, 先经过VisionTransformer 再投影到2048维
        image_embeddings = self.vision_tower(pixel_values) # (B, num_patches=256, hidden=1152)
        image_embeddings = self.multi_modal_projector(image_embeddings)

        # 3) 按PaliGemma习惯, 将图片特征做一次缩放scale(除以sqrt(hidden)), 让其量级和text embed接近
        scaled_image_embeddings = image_embeddings / (self.image_text_hidden_size ** 0.5)

        # 4) 准备最终输出张量
        # trick : 用pad_token_id来填初始值
        _, _, embed_dim = image_embeddings.shape
        bsz, seq_len = input_ids.shape
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )
        # text_mask：[B, seq_len] True 真实文本token
        # image_mask: [B, seq_len] True 图片占位符token的位置
        text_mask = (input_ids != self.image_token_index) & (input_ids != self.pad_token_id) # 这里用& 而不是and
        image_mask = input_ids == self.image_token_index
        # 把文本位置的embedding替换进去
        final_embedding[text_mask] = image_text_embeddings[text_mask]
        # 把256个patch图 对应的embedding替换进去
        # 文本不用 for 循环——是因为 embed_tokens 的 seq 维跟你要填进去的那个位置的 seq 维是同一个维,布尔索引天然保留
        #   batch 对应关系。
        #   图像必须 for 循环——是因为图像特征的 seq 维(固定 256)跟 mask 选出来的位置数(可变)根本不是一回事,必须手动配对。
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_token = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_embeddings[i, :num_image_token]
        # final_embedding[image_mask] = scaled_image_embeddings
        return final_embedding

class PiZero(nn.Module):
    def __init__(self, config):
        super().__init__()
        OmegaConf.resolve(config) # 把${...}提前换成具体值
        self.vocab_size = config.vocab_size

        self.max_image_text_tokens = config.max_image_text_tokens
        self.num_proprio_tokens = config.cond_steps
        self.num_action_tokens = config.horizon_steps
        self.total_num_tokens = self.max_image_text_tokens + self.num_proprio_tokens + self.num_action_tokens

        self.action_dim = config.action_dim
        self.flow_sig_min = config.get("flow_sig_min", 0.001)
        self.adaptive_mode = config.action_expert_adaptive_mode

        self.final_action_clip_value = config.get("final_action_clip_value", None)
        self.num_inference_steps = config.num_inference_steps # Flow Matching 推理时去噪的步数。10

        self.embedder = PaliGemmaEmbedder(config)
        self.joint = JointModel(config.joint)
        if self.adaptive_mode:
            # adaLN / adaLN-Zero:time 用 256 维,不 cat 进 action
            self.time_encoder = TimeEncoder(config.time_hidden_size, config.time_max_period)
            self.action_encoder = ActionEncoder(config.action_dim, config.action_hidden_size, time_cond=False)
        else:
            # # 朴素 time_cond:time 必须 1024 维,跟 action cat
            self.time_encoder = TimeEncoder(config.action_hidden_size, config.time_max_period) # action和time必须同维度才能cat
            self.action_encoder = ActionEncoder(config.action_dim, config.action_hidden_size, time_cond=True)
        self.proprio_encoder = ProprioEncoder(config.proprio_dim, config.proprio_hidden_size)
        self.action_decoder = ActionDecoder(config.action_hidden_size, config.action_dim)

        self.tie_action_proprio_weights()

    def forward(self, batch):
        '''
        batch:
            input_ids:    (B, L_text)
            attention_mask: (B, max_seq_len) 有效token标记
            pixel_values: (B, 3, 224, 224)
            proprio:      (B, cond_step, proprio_dim)
            action:      (B, T_action, action_dim)   ← x_1，真实动作
        '''
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']
        attention_mask = batch['attention_mask']
        proprio = batch['proprio']
        action = batch['action']

        # 步骤1： 三段embed
        vlm_emb = self.embedder(input_ids, pixel_values)
        proprio_emb = self.proprio_encoder(proprio)

        # 步骤2： FM采样
        B, T_a, A = action.shape
        t = torch.rand(B, device=action.device, dtype=action.dtype) # 生成均匀分布(0, 1)之间的随机数, 正好是Flow Matching所需要的
        noise = torch.randn_like(action)
        t_b = t[:, None, None]
        sig = self.flow_sig_min
        x_t = (1 - (1-sig) * t_b) * noise + t_b * action # 1024

        # 步骤3：t embeddig + atciont(x_t) embedding
        time_emb = self.time_encoder(t) # 256
        if self.adaptive_mode:
            action_emb = self.action_encoder(x_t) # # adaLN: time 走 norm，不进 encoder
        else:
            action_emb = self.action_encoder(x_t, time_emb) # 朴素: time 和 action cat

        # 步骤4：position_ids + block-wise causal mask
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = \
            self.build_mask_and_position_ids(attention_mask, action_emb.dtype)

        # 步骤5: joint forward
        embeds_all = {"vlm": vlm_emb, "proprio": proprio_emb, "action": action_emb}
        positions_all = {"vlm": vlm_position_ids, "proprio": proprio_position_ids, "action": action_position_ids}
        out = self.joint(causal_mask, positions_all, embeds_all, time_cond=time_emb if self.adaptive_mode else None)

        # 步骤6：action_expert -> 预测速度
        v_pred = self.action_decoder(out['action'])

        # 步骤7： FM Loss
        v_target = action - (1-sig) * noise # 真实速度场, 先用最简单的x_t = (1-t)*x_0 + t*x_1
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss


    @torch.no_grad()
    def infer_action_naive(self, batch, num_inference_steps: int= 10, noise: torch.FloatTensor = None):
        '''朴素推理（不缓存 KV）：每步重跑 vlm+proprio+action 全套'''
        input_ids      = batch['input_ids']
        pixel_values   = batch['pixel_values']
        attention_mask = batch['attention_mask']
        proprio        = batch['proprio']

        dtype = pixel_values.dtype
        device = pixel_values.device
        B = pixel_values.shape[0]
        # 步骤1： 准备vlm_emb, mask, position_ids
        vlm_emb = self.embedder(input_ids, pixel_values)
        proprio_emb = self.proprio_encoder(proprio)
        total_mask, vlm_pos, proprio_pos, action_pos = self.build_mask_and_position_ids(attention_mask, dtype)

        # 步骤2：从纯噪声出发
        if noise is None:
            x = torch.randn(B, self.num_action_tokens, self.action_dim, device=device, dtype=dtype)
        else:
            x = noise.to(device=device, dtype=dtype).clone()
        # 步骤3：欧拉积分
        dt = 1.0 / num_inference_steps
        t = torch.zeros(B, device=device, dtype=dtype)
        for _ in range(num_inference_steps):
            # 编码当前x 和 t
            time_emb = self.time_encoder(t)
            if self.adaptive_mode:
                action_emb = self.action_encoder(x)
            else:
                action_emb = self.action_encoder(x, time_emb)

            # 拿到当前位置的速度场
            out = self.joint(
                attention_mask = total_mask,
                position_ids_all = {'vlm': vlm_pos, 'proprio': proprio_pos, 'action': action_pos},
                embeds_all = {'vlm': vlm_emb, 'proprio': proprio_emb, 'action': action_emb},
                time_cond=time_emb if self.adaptive_mode else None,
                kv_caches = None,
                cache_mode = "no_append",
                )

            v = self.action_decoder(out['action'])
            # Euler
            x = x + dt * v
            t = t + dt
        if self.final_action_clip_value is not None:
            x = torch.clamp(
                x,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return x

    @torch.no_grad()
    def infer_action(self, batch, num_inference_steps: int = 10, noise: torch.FloatTensor = None):
        '''batch:
          input_ids:      (B, max_image_text_tokens)   ← VLM 文本+图像占位
          pixel_values:   (B, 3, 224, 224)             ← 图像
          attention_mask: (B, max_image_text_tokens)   ← padding mask
          proprio:        (B, cond_steps, proprio_dim) ← 机器人当前状态
      返回:
          action_pred:    (B, horizon_steps, action_dim)'''
        input_ids      = batch['input_ids']
        pixel_values   = batch['pixel_values']
        attention_mask = batch['attention_mask']
        proprio        = batch['proprio']

        dtype = pixel_values.dtype
        device = pixel_values.device
        B = pixel_values.shape[0]
        # 步骤1： 准备vlm_emb, mask, position_ids
        vlm_emb = self.embedder(input_ids, pixel_values)
        proprio_emb = self.proprio_encoder(proprio)
        total_mask, vlm_pos, proprio_pos, action_pos = self.build_mask_and_position_ids(attention_mask, dtype)
        # 拆分mask
        vlm_proprio_mask, action_mask = self.split_full_mask_into_submask(causal_mask=total_mask)
        kv_caches = self.joint.build_mixture_caches() # 创建KV Cache

        # 阶段1： prefill
        # vlm expert 和 proprio expert的输出不需要, 用_来接收
        if self.adaptive_mode:
            time_emb_prefill = self.time_encoder(torch.zeros(B, device=device, dtype=dtype))
        else:
            time_emb_prefill = None

        _, kv_caches = self.joint(
            attention_mask = vlm_proprio_mask,
            position_ids_all = {'vlm': vlm_pos, 'proprio': proprio_pos},
            embeds_all = {'vlm': vlm_emb.clone(), 'proprio': proprio_emb.clone()},
            time_cond = time_emb_prefill,
            kv_caches = kv_caches,
            # cache_mode = "no_append",
            return_caches = True,
        )

        # 阶段2： denoise 循环
        position_ids_all = {"action": action_pos}
        # 步骤2：从纯噪声出发
        if noise is None:
            x = torch.randn(B, self.num_action_tokens, self.action_dim, device=device, dtype=dtype)
        else:
            x = noise.to(device=device, dtype=dtype).clone()
        # 步骤3：欧拉积分
        dt = 1.0 / num_inference_steps
        t = torch.zeros(B, device=device, dtype=dtype)
        for _ in range(num_inference_steps):
            # 编码当前x 和 t
            time_emb = self.time_encoder(t)
            if self.adaptive_mode:
                action_emb = self.action_encoder(x)
            else:
                action_emb = self.action_encoder(x, time_emb)

            # 拿到当前位置的速度场
            out = self.joint(
                attention_mask = action_mask,
                position_ids_all = {'action': action_pos},
                embeds_all = {'action': action_emb.clone()},
                time_cond=time_emb if self.adaptive_mode else None,
                kv_caches = kv_caches,
                cache_mode = "append_non_active",
                )

            v = self.action_decoder(out['action'])
            # Euler
            x = x + dt * v
            t = t + dt
        if self.final_action_clip_value is not None:
            x = torch.clamp(
                x,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return x

    def build_mask_and_position_ids(self, attention_mask, dtype:torch.dtype):
        bsz = attention_mask.shape[0]
        device = attention_mask.device
        proprio_start = self.max_image_text_tokens
        action_start = self.max_image_text_tokens + self.num_proprio_tokens
        # 每个batch实际有效的image/text token数量
        valid_image_text_token = torch.sum(attention_mask, dim=-1)

        mask_pre = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens), torch.finfo(dtype).min,
            dtype=dtype, device=device)
        for idx, cnt in enumerate(valid_image_text_token):
            # 有效image/text token内部相互可见
            mask_pre[idx, :cnt, :cnt] = 0
            # proprio/action 可以看到image/text 分两步写，跳过image/text到proprio中间填充的padding
            mask_pre[idx, proprio_start:, :cnt] = 0

        # proprio内部相互可见
        mask_pre[:, proprio_start:action_start, proprio_start:action_start ] = 0
        # action可以看proprio
        mask_pre[:, action_start:, proprio_start:] = 0
        # 加head 维, [B, T_total, T_total] -> [B, 1, T_total, T_total]
        causal_mask = mask_pre.unsqueeze(1)

        # 位置编码id: 每段都从1开始, 方便和RoPE配合
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens+1, device=device).expand(bsz, -1)
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens+1, device=device).expand(bsz, -1)
        # action_postion_ids = torch.arange(1, self.num_action_tokens).expand(bsz, -1)
         # action_position_ids 接在 proprio 后面继续编号：例如 proprio=1 step 时 action=[2,3,4,5]
        # 因为 proprio 和 action 共享 mixture 权重，用连续编号更合理
        action_position_ids = torch.arange(self.num_proprio_tokens+1, self.num_proprio_tokens+self.num_action_tokens+1,
                                           device=device ).expand(bsz, -1)
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submask(
            self,
            causal_mask: torch.FloatTensor
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        - prefill 时 active 只有 vlm+proprio，Q 和 K/V 都只到 vlm+proprio 那段 →
        需要左上的 [vlm+proprio, vlm+proprio] 子方块
        - denoise 时 Q 是 action 那 num_action_tokens 行，
        但 K/V 要全长（vlm+proprio 从 cache 来 + action 当步算）→ 需要最后

        把完整 mask 切成两块，用于 infer_action 的「先跑 VLM+proprio 缓存 KV，再循环跑 action」流程。

        输入  causal_mask:                 [B, 1, total_tokens, total_tokens]
        输出:
            image_text_proprio_mask:       [B, 1, vlm+proprio, vlm+proprio]    —— 推理第一阶段，prefill 用
            action_mask:                   [B, 1, num_action_tokens, total]    —— 推理第二阶段，每个去噪步用
                                                                                 (Q 只取 action 那几行，K/V 是全长)
        '''
        iamge_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return iamge_text_proprio_mask, action_mask

    def tie_action_proprio_weights(self):
        """proprio 和 action 共享同一个动作专家的权重"""
        self.joint.mixtures["proprio"] = self.joint.mixtures["action"]


if __name__ == "__main__":
    import time
    from PIL import Image
    import numpy as np
    from transformers import AutoTokenizer

    from model.vla.processing import VLAPreProcessor
    from model.load_pretrained import load_pretrained_pizero

    config = OmegaConf.load("config/yuepi0.yaml")
    deploy_cfg = OmegaConf.load("config/deploy_simpler.yaml")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    ckpt = deploy_cfg.checkpoint_path
    model = load_pretrained_pizero(config, ckpt, strict=True)
    model.to(dtype).to(device).eval()

    print(f"Using {device} and {dtype}...")

    # dummy image --- replace the first image with a real one
    bsz = 1
    # 随机噪声图
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 224, 224), dtype=torch.uint8
    )  # not used if text_only
    real_image_path = "/home/cxy/projects/open-pi-zero/media/maniskill_pp.png"
    real_image = Image.open(real_image_path).convert("RGB")
    real_image_t = torch.as_tensor(
        np.array(real_image.resize((224, 224))).transpose(2, 0, 1)
    )
    dummy_images[0] = real_image_t

    # 输入指令文本
    # text and proprio,
    dummy_texts = [
        "this image shows ",
        "this is a nice portrait of London because ",
    ][:bsz]
    dummy_proprio = torch.rand(bsz, config.cond_steps, config.action_dim)


    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")

    processor = VLAPreProcessor(tokenizer=tokenizer,
                    num_image_token=config.vision_config.num_image_tokens,
                    max_seq_len=config.max_seq_len)

    # process image and text
    model_inputs = processor(prompts=dummy_texts, images=dummy_images)
    input_ids = model_inputs['input_ids']
    image_text_attention_mask = model_inputs['attention_mask']
    pixel_values = model_inputs['pixel_values'].to(dtype)

    # 推理
    start_time = time.time()

    # 把输入移到 device + dtype
    batch = {
        "input_ids":      input_ids.to(device),
        "pixel_values":   pixel_values.to(device),
        "attention_mask": image_text_attention_mask.to(device),
        "proprio":        dummy_proprio.to(dtype).to(device),
    }

    # 固定 noise（对拍时两版必须用同一个 noise 才能比）
    torch.manual_seed(42)
    noise = torch.randn(
        bsz, config.horizon_steps, config.action_dim,
        device=device, dtype=dtype,
    )

    # 跑 cache 版推理
    actions = model.infer_action(batch, num_inference_steps=10, noise=noise)

    print("\n=========================")
    print("action shape:", actions.shape)
    print("action values:", actions)
    print("Time taken:", time.time() - start_time)
    print("=========================\n")


    # 朴素版对拍（同一个 noise）
    actions_naive = model.infer_action_naive(batch, num_inference_steps=10, noise=noise)
    print("naive action shape:", actions_naive.shape)
    print("max diff (cache vs naive):", (actions - actions_naive).abs().max())
import torch
from torch import nn
from model.paligemma.vit import ViTVisionModel,ImageProjector

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


if __name__ == "__main__":
    
    from transformers import AutoTokenizer
    from model.vla.processing import VLAPreProcessor
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('config/yuepi0.yaml')
    model = PaliGemmaEmbedder(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    
    assert tokenizer.padding_side == "right"

    bsz = 1
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 224, 224), dtype=torch.uint8
    )
    prompts = ["this images contains ", "this is a nice portrait of "][:bsz]

    num_image_tokens = 256
    processor = VLAPreProcessor(tokenizer, 
                                num_image_token=num_image_tokens, 
                                max_seq_len=config.max_seq_len)
    
    # 生成图片占位token
    model_inputs = processor(prompts=prompts, images=dummy_images)
    input_ids = model_inputs['input_ids']
    pixel_values = model_inputs['pixel_values']

    with torch.no_grad():
        out = model(input_ids, pixel_values)
    print("out.shape: ", out.shape)
    print("nan/inf count:", torch.isfinite(out).sum().item())
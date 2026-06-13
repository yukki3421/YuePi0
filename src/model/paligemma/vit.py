import torch
from torch import nn
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ViTVisionConfig:
    hidden_size: int = 1152 # 每个patch的维度
    intermediate_size: int = 4304 # MLP中间层的维度
    patch_size: int = 14
    image_size: int = 224
    num_attention_heads: int = 16 # 注意力头数
    num_channels: int = 3 # RGB
    num_hidden_layers: int = 27 # encoder层数
    layer_norm_eps: float = 1e-6 
    attention_dropout: float = 0.0

# 实现Conv2d + position_embedding
class ViTVisionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.embed_dim = config.hidden_size

        # 先对图片做patch, 得到embed_dim
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, out_channels=self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        self.num_positions = self.num_patches
        # position_embeddings
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # register_buffer 是什么：把一个张量注册到模块里，但不算可学习参数（不会被 optimizer 更新）。
        # 这里 position_ids 就是 [0, 1, 2, ..., 255]，固定不变，但要跟着模型一起搬到 GPU、保存进 state_dict。
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).unsqueeze(0), persistent=False,
        )
    
    # 接收图像 pixel_values: [B, 3, 224, 224]
    def forward(self, pixel_values):
        # (B, embed_dim, 224/patch_size, 224/patch_size) -> (B, num_patches, embed_dim)
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

# 标准MHA
class ViTVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 注意！这里是self.head_dim而不是embed_dim!
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    # x: (B, num_patches, embed_dim)
    def forward(self, x):
        B = x.size(0)
        T = x.size(1)
        # (B, num_patches, head, head_dim) -> (B, head, num_patches, head_dim)
        q = self.q_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weight = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weight = self.dropout(attn_weight)

        assert attn_weight.size() == (B, self.num_heads, T, T)
        # (B, num_heads, T, head_dim)
        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(1,2).contiguous().view(B, T, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output


# 两层 Linear +  GELU(tanh approx) 
class ViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    # input_state: [batch_size, num_patches, embed_dim]
    def forward(self, input_state):
        hidden_state = self.fc1(input_state)
        hidden_state = nn.functional.gelu(hidden_state, approximate="tanh")
        hidden_state = self.fc2(hidden_state)
        return hidden_state


# Pre-LN block：x = x + Attn(LN(x))，再 x = x + MLP(LN(x)) 
class ViTEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = ViTVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = ViTMLP(config)
    
    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

# 将ViTEncoderLayer串起来
class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        # self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList([
            ViTEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
    # Inputs : [Batch_size, num_patches, embed_dim]
    def forward(self, input_state):
        hidden_state = input_state
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

# embeddings + encoder + post_layernorm
class ViTVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.embedding = ViTVisionEmbedding(config)
        self.encoder = ViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, input_images):
        input_states = self.embedding(input_images) # (B, 3, 224, 224) -> (B, num_patches, embed_dim)
        hidden_state = self.encoder(input_states)
        hidden_state = self.post_layernorm(hidden_state)
        return hidden_state

'''最外层的薄壳, 加一层self.vision_model名字, (为了对齐HuggingFace PaliGemma的state_dict key命名, 方便加载预训练权重)'''
class ViTVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = ViTVisionTransformer(config)

    def forward(self, images):
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(images)

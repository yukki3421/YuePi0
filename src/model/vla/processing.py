from typing import List
import torch


'''
    经典的 ImageNet 数据集均值/标准差（在 [0,1] 范围内逐通道统计得到）是：
    
        MEAN = [0.485, 0.456, 0.406]   # R, G, B
        STD  = [0.229, 0.224, 0.225]
    
    这是 ResNet / VGG 这一代 CNN 用的，目的是把数据 z-score 标准化到 N(0,1) 附近。
    
'''

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])
# 注意：
# [0.5, 0.5, 0.5] 不是统计量，而是一个线性映射的参数：    
#         x_norm = (x - 0.5) / 0.5  =  2*x - 1
#     效果：把像素值从 [0, 1] 线性映射到 [-1, 1]。

'''
输入：images [batch, 3, hidth, width]
对images的像素 放缩到 [0, 1]
'''
def imageScale(images: torch.Tensor, scale: float) -> torch.Tensor:
    
    return images * scale

'''
输入：images [batch, 3, hidth, width]
mean: [ m, m, m] 三个通道的均值
var: [v, v, v] 三个通道的方差
输出：对每个通道做标准化的结果
'''
def imageNormalize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    assert images.ndim == 4 and images.shape[1] == 3, "images的形状应为(batch_size, 3, width, hidth)"
    assert mean.ndim == 1 and std.ndim == 1
    assert len(mean) == len(std) == 3
    mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return (images - mean) / std

'''
将图片先 scale归一化 再 z-score标准化到[-1, 1]
'''
def imagePreprocess(images, scale, mean, std):
    return imageNormalize(imageScale(images, scale), mean, std)

'''
PaliGemma（pi-zero 的 VLM backbone）是个纯 decoder-only 的语言模型。
它只会读 token 序列，根本不知道"图像"
那图像怎么进来？答案是 "占位符 + 后期替换" 的技巧：
 文本 prompt:    "pick up the red cube"
                            ↓ 预处理
  真正送入模型:   <image><image>...<image><bos>pick up the red cube\n
                  └─── 256 个占位符 ───┘
                            ↓ tokenizer 编码
  input_ids:      [257152, 257152, ..., 257152, 2, 1495, 738, ...]
                            ↓ 模型 forward 时
  embedding 层:   每个 <image> 的 embedding 被「图像 patch embedding」替换掉   
'''
def add_image_token_to_prompts(
        num_image_token : int, # 图像token的个数, 通常为256
        image_token : str,  # 图像token占位符 "<image>"
        prompt : str, # 用户指令, 比如"show the pictures content", 
        bos_token: str, # tokenizer.bos_token, 通常是"<bos>"
) -> str:
    return f"{image_token * num_image_token}{bos_token}{prompt}\n"


'''

'''
class VLAPreProcessor:
    IMAGE_TOKEN = "<image>"
    def __init__(
            self, 
            tokenizer, 
            num_image_token :int, 
            max_seq_len: int, 
            tokenizer_padding: str = "max_length") -> None:
        # TODO 1： 保存参数
        self.num_image_token = num_image_token
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # TODO 2: 给tokenizer注册特殊token
        add_special_tokens = {'additional_special_tokens': [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(add_special_tokens)

        # TODO 3: 生成并注册 1024 个 <locXXXX> + 128 个 <segXXX>
        # <loc:i:04d> 四位补零, <seg:o3d> 三位补零
        EXTRA_TOKENS = [F"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS) # 这些是用于目标检测和图像分割
        '''
        这里在做什么？为什么要塞这些token? ----  把"几何信息"也变成"语言"
         1. <loc> 系列 —— 把坐标变成 token

        普通的目标检测模型输出 bounding box 是 4 个浮点数：

        bbox = (x1, y1, x2, y2) = (0.12, 0.34, 0.56, 0.78)

        但 PaliGemma 是纯语言模型，它只会生成 token，不会输出浮点数。怎么办？

        做法：把 [0, 1] 这个连续区间离散化成 1024 个 bin，每个 bin 给一个 token：

        坐标 0.000  →  <loc0000>
        坐标 0.001  →  <loc0001>     (其实是 floor(0.001 × 1024) = <loc0001>)
        坐标 0.500  →  <loc0512>
        坐标 0.999  →  <loc1023>

        于是一个 bbox (0.12, 0.34, 0.56, 0.78) 在 PaliGemma 眼里变成 4 个 token：

        <loc0122><loc0348><loc0573><loc0798>

        模型输出这串 token，外面解码回坐标即可。检测任务 = 文本生成任务。

        举个完整例子，PaliGemma 训练时见过这种数据：
        输入: <image>×256 <bos> detect cat \n
        输出: <loc0122><loc0348><loc0573><loc0798> cat

        2. <seg> 系列 —— 把分割 mask 变成 token

        分割 mask 比 bbox 信息量大得多（是个 H×W 的二值矩阵），1024 个 bin 装不下。

        PaliGemma 的做法：先用一个 VQ-VAE 把 mask 压缩成 16 个离散 codebook id，每个 id 的取值范围是 [0, 128)。所以需要 128 个 <seg>
        token：

        一个 mask  →  VQ-VAE 编码  →  16 个 id  →  <seg042><seg017>...<seg089>

        模型输出这 16 个 token，外面用 VQ-VAE 解码器还原回 mask。
        '''
        
        # TODO 4: 拿到<image>的token_id保存
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # TODO 5: 关掉自动 BOS/ EOS
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        prompts: List[str], # 一批用户指令
        images: torch.Tensor, # 未处理的图片[B, 3, H, W]
        truncation: bool = True) -> dict:
        # TODO 1: 校验参数
        assert len(prompts) == len(images), f"接收{len(images)}张图片, {len(prompts)} 条文本输入"
        assert images.dtype == torch.uint8
        # torch.uint8的范围是[0, 255], 后面除以255, 正好变成[0, 1]之间的浮点数. 确保传入的是原始像素值

        # TODO 2: 图像预处理, 将每个像素点的取值调整到[-1, 1]区间
        pixel_image = imagePreprocess(images, 1/255.0, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD)

        # TODO 3: 对prompts中的每一个prompt加入image token占位
        input_strings = [
            add_image_token_to_prompts(
                num_image_token = self.num_image_token,
                image_token = self.IMAGE_TOKEN,
                prompt=prompt,
                bos_token = self.tokenizer.bos_token
            ) for prompt in prompts
        ]

        # TODO 4: 利用self.tokenizer(...) 编码input_strings
        inputs = self.tokenizer(input_strings, 
                       return_tensors="pt", 
                       max_length=self.max_seq_len, 
                       padding=self.tokenizer_padding, 
                       truncation=truncation)
        '''
        对于一个普通的 tokenizer（比如 PaliGemma / Gemma 用的 SentencePiece），返回 BatchEncoding 包含 2 个键：
        {               
            "input_ids":      torch.LongTensor of shape (B, max_seq_len),                                                        
            "attention_mask": torch.LongTensor of shape (B, max_seq_len),                                                        
        }
        具体到我们的参数：                                                                                                                         
        ┌─────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐              
        │            参数设置             │                              影响的字段                               │
        ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤              
        │ return_tensors="pt"             │ 让值变成 torch.Tensor 而不是 List[List[int]]                          │
        ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤              
        │ padding="max_length"            │ 让所有样本被填充到 max_seq_len 长度，决定 attention_mask 里有多少个 0 │              
        ├─────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤              
        │ max_length=300、truncation=True │ 决定最长不超过 300，超出从右边截断                                    │              
        └─────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘                                                                             
        attention_mask输出：
            1 = 这个位置是真 token，模型要看它
            0 = 这个位置是 padding，模型要忽略  
        '''
        # TODO 5: 组装返回 dict: {"pixel_values": ..., **inputs}    
        output = {'pixel_values': pixel_image, **inputs}
        return output
        
        


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/home/cxy/.cache/huggingface/hub/paligemma-3b-pt-224', padding_side="right"
    )
    valprocessor = VLAPreProcessor(tokenizer=tokenizer, num_image_token=10, max_seq_len=20, tokenizer_padding="max_length")
    fake_images = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    fake_texts = ['pick up the cube', 'open the door']
    out = valprocessor(fake_texts, fake_images,)

    for k, v in out.items():
        print(k, v.shape, v.dtype)
    print(out['attention_mask'])


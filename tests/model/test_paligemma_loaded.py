"""验证加载 PaliGemma 后 PiZero 能正常前向, loss 数值合理。"""                                                                                       
from pathlib import Path                                                                                                                             
import sys                                                                                                                                           
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))                                                                                 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] ))                                                                                 
                                                                                                                                                    
import torch
from omegaconf import OmegaConf                                                                                                                      
from torch.utils.data import DataLoader                                                                                                              
from transformers import AutoTokenizer
                                                                                                                                                    
from data.fake_dataset import FakeBridgeDataset                                                                                                      
from model.vla.processing import VLAPreProcessor                                                                                                     
from model.vla.yuepi0 import PiZero                                                                                                                  
from model.utils import load_paligemma_weights  # 复用你刚写的                                                                            
                                                                                                                                                    
                                                                                                                                                    
def main():                                                                                                                                          
    # 用 yuepi0.yaml (全尺寸架构), 但只在 CPU 上做 sanity 检查
    # 全尺寸 3B 模型放 GPU 会 OOM, CPU 上做 forward 验证就够                                                                                         
    config = OmegaConf.load("config/yuepi0.yaml")                                                                                                    
    OmegaConf.resolve(config)                                                                                                                        
                                                                                                                                                    
    # 1) 建模型 + 加载权重 (CPU, fp32)                                                                                                               
    print("Building model...")
    model = PiZero(config)                                                                                                                           
    print("Loading PaliGemma weights...")
    load_paligemma_weights(model, Path(config.pretrained_model_path))                                                                                
    model.eval()  # 我们只测前向, 不训练                                                                                                             
                                                                                                                                                    
    # 2) 准备 fake 数据 (跟 train.py 完全一样的流程)                                                                                                 
    # 注意:fakedataTrain.yaml 里 num_samples/seed 等没在 yuepi0.yaml 里                                                                              
    config.num_samples = 4                                                                                                                           
    config.seed = 42
    config.batch_size = 2                                                                                                                            
    dataset = FakeBridgeDataset(config, num_samples=4, seed=42)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)                                                                                        
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")                                                    
    processor = VLAPreProcessor(                                                                                                                     
        tokenizer=tokenizer,                                                                                                                         
        num_image_token=config.vision_config.num_image_tokens,                                                                                       
        max_seq_len=config.max_seq_len,                                                                                                              
    )                                                                                                                                                
                                                                                                                                                    
    raw_batch = next(iter(loader))                                                                                                                   
    images = raw_batch['image'].squeeze(1).permute(0, 3, 1, 2)
    out = processor(prompts=raw_batch['text'], images=images, truncation=True)                                                                       
    inputs = {                                                                                                                                       
        "input_ids":      out['input_ids'],                                                                                                          
        "attention_mask": out['attention_mask'],                                                                                                     
        "pixel_values":   out['pixel_values'],                                                                                                       
        "proprio":        raw_batch['proprio'],                                                                                                      
        "action":         raw_batch['action'],                                                                                                       
    }                                                                                                                                                
                                                                                                                                                    
    # 3) 前向: 跟训练完全一样的代码路径                                                                                                              
    print("Running forward...")
    with torch.no_grad():                                                                                                                            
        loss = model(inputs)
    print(f"\nloss = {loss.item():.4f}")                                                                                                             
    print(f"isfinite: {torch.isfinite(loss).item()}")                                                                                                
                                                                                                                                                    
    # 4) 健康检查: 各个模块输出有没有 NaN                                                                                                            
    print("\n=== quick layer health check ===")                                                                                                      
    with torch.no_grad():                                                                                                                            
        vlm_emb = model.embedder(inputs['input_ids'], inputs['pixel_values'])                                                                        
        print(f"vlm_emb: shape={tuple(vlm_emb.shape)} "                                                                                              
            f"finite={torch.isfinite(vlm_emb).all().item()} "                                                                                      
            f"mean={vlm_emb.mean().item():.4f} "                                                                                                   
            f"std={vlm_emb.std().item():.4f}")                                                                                                     
                                                                                                                                                    
                                                                                                                                                    
if __name__ == "__main__":                                                                                                                           
    main()
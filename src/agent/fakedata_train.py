import sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader  
from omegaconf import OmegaConf  
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                                                                                 

from data.fake_dataset import FakeBridgeDataset                                                                                                      
from model.vla.processing import VLAPreProcessor                                                                                                     
from model.vla.yuepi0 import PiZero 
from model.utils import load_paligemma_weights, to_device_bf16
from agent.train import preprocess_batch

def fakedatatrain():
    # 1.加载配置
    config = OmegaConf.load("config/fakedataTrain.yaml")
    OmegaConf.resolve(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.创建dataset + dataloader
    dataset = FakeBridgeDataset(config, num_samples=config.num_samples, seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # 3.创建tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")
    processor = VLAPreProcessor(tokenizer=tokenizer, num_image_token=config.vision_config.num_image_tokens, max_seq_len=config.max_seq_len)

    # 4.创建 model + optimizer
    model = PiZero(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    step = 0
    loader_iter = iter(loader)
    start_time = time.time()

    step = 0
    while step < config.num_steps:
        # 1. 取一个batch
        try:
            raw_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader) # 数据用完了, 重新开始
            raw_batch = next(loader_iter)
        
        # 2. raw -> model输入
        inputs = preprocess_batch(raw_batch=raw_batch, processor=processor)

        # 3. 搬到device
        inputs = {k:v.to(device) for k, v in inputs.items()}

        # 4. forward
        loss = model(inputs)

        # 5. backward + step 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 5. log
        if step % config.log_every == 0:
            print(f"step {step:4d}/{config.num_steps}  loss = {loss.item():.6f}  grad = {grad_norm.item():.3f}")

        step += 1
    end_time = time.time()
    print("train time: ", end_time-start_time)
    # =============================评估overfit验证=========================
    model.eval()
    #  1) 用 shuffle=False 的 loader 重新取数据 
    eval_loader = DataLoader(dataset, batch_size=config.num_samples, shuffle=False)
    raw_batch = next(iter(eval_loader))

    # 2) 同样的预处理流程
    inputs = preprocess_batch(raw_batch=raw_batch, processor=processor)                                                                                                            
    inputs = {k: v.to(device) for k, v in inputs.items()}  

    # 3) 留着 GT，下一步要用                                                                                                                         
    action_gt = inputs['action']            # (B=4, horizon=4, action_dim=7)
    print("eval batch ready:")                                                                                                                       
    print(f"  action_gt.shape = {action_gt.shape}")                                                                                                  
    print(f"  pixel_values.shape = {inputs['pixel_values'].shape}")

    # 4）跑推理
    with torch.no_grad():
        action_pred = model.infer_action(batch=inputs, num_inference_steps=config.num_steps)
    print(f" action_pred.shape = {action_pred.shape}")

    # 5) 对比 GT vs Pred ——————————————————————                                                                                                      
    diff = action_pred - action_gt                              # (B, horizon, action_dim)                                                           
                                                                                                                                                    
    mae  = diff.abs().mean()                                     # 标量                                                                               
    mse  = (diff ** 2).mean()                                   # 标量                                                                               
    per_sample_mae = diff.abs().mean(dim=(1, 2))                 # (B,) 每个样本一个                                                                  
                                                                                                                                                    
    # baseline：用随机噪声当预测，看 MAE 多少                                                                                                        
    baseline_pred = torch.rand_like(action_gt) * 2 - 1          # ← 为啥这样写？回想 fake_dataset                                                    
    baseline_mae  = (baseline_pred - action_gt).abs().mean()                                                                                         
                                                                                                                                                    
    print("\n========= overfit eval =========")                                                                                                      
    print(f"  MAE             = {mae.item():.4f}")                                                                                                   
    print(f"  MSE             = {mse.item():.4f}")                                                                                                   
    print(f"  per-sample MAE  = {per_sample_mae.tolist()}")                                                                                        
    print(f"  baseline MAE    = {baseline_mae.item():.4f}  (随机预测)")                                                                              
    print(f"  ratio (MAE/baseline) = {mae.item()/baseline_mae.item():.3f}  (越小越好)")
import sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader  
from omegaconf import OmegaConf  
from transformers import AutoTokenizer

from data.fake_dataset import FakeBridgeDataset                                                                                                      
from model.vla.processing import VLAPreProcessor                                                                                                     
from model.vla.yuepi0 import PiZero 

def preprocess_batch(raw_batch, processor):
    # 1) 取出raw_batch里的字段
    images = raw_batch['image'].squeeze(1).permute(0, 3, 1, 2)
    proprio = raw_batch['proprio']
    action = raw_batch['action']
    texts = raw_batch['text']

    output = processor(prompts=texts, images=images, truncation=True)
    input_ids = output['input_ids']
    pixel_values = output['pixel_values']
    attention_mask = output['attention_mask']

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "proprio": proprio,
        "action": action,
    }


def main():
    # 1.加载配置
    config = OmegaConf.load("config/fakedataTrain.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.创建dataset + dataloader
    dataset = FakeBridgeDataset(config, num_samples=config.num_samples, seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 3.创建tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")
    processor = VLAPreProcessor(tokenizer=tokenizer, num_image_token=config.vision_config.num_image_tokens, max_seq_len=config.max_seq_len)

    # 4.创建 model + optimizer
    model = PiZero(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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
        optimizer.step()

        # 5. log
        if step % config.log_every == 0:
          print(f"step {step:4d}/{config.num_steps}  loss = {loss.item():.6f}")

        step += 1


if __name__ == "__main__":
    main()
import sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader  
from omegaconf import OmegaConf  
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                                                                                 

from data.bridge_dataset import BridgeDataset                                                                                                      
from model.vla.processing import VLAPreProcessor                                                                                                     
from model.vla.yuepi0 import PiZero 
from model.utils import load_paligemma_weights, to_device_bf16

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


def freeze_vlm(model):
    """
    冻结从 PaliGemma 加载的所有模块, 只训 action/proprio expert + encoders/decoders。

    返回:
        trainable_params: 可训练参数列表, 用来传给 optimizer
        n_total, n_trainable: 参数数量统计
    """
    # 1) 列出要冻结的 模块
    modules_to_freeze = [
        model.embedder.embed_tokens,
        model.embedder.vision_tower,
        model.embedder.multi_modal_projector,
        model.joint.mixtures['vlm']
    ]
    # 2) 把这些模块的所有参数 requires_grad = False
    for m in modules_to_freeze:
        for p in m.parameters():
            p.requires_grad = False
    # 3）收集可训练参数 + 统计
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)

    print(f"frozen VLM. trainable: {n_trainable/1e6:.1f}M / total: {n_total/1e6:.1f}M "
            f"({100*n_trainable/n_total:.1f}%)")

    return trainable_params, n_total, n_trainable


def main():
    # 1.加载配置
    config = OmegaConf.load("config/realdataTrain.yaml")
    OmegaConf.resolve(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.创建dataset + dataloader
    dataset = BridgeDataset(config, config.data_dir, config.max_episodes)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 3.创建tokenizer + processor
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path, padding_side="right")
    processor = VLAPreProcessor(tokenizer=tokenizer, num_image_token=config.vision_config.num_image_tokens, max_seq_len=config.max_seq_len)

    # 4.创建 model + optimizer, 加载预训练权重
    model = PiZero(config)
    load_paligemma_weights(model, Path(config.pretrained_model_path))
    model = model.to(torch.bfloat16).to(device)
    trainable_params, n_total, n_trainable = freeze_vlm(model)

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)   
    # === 显存自检 ===                                                                                                                               
    torch.cuda.synchronize()
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model+optimizer setup: {mem_gb:.2f} GB")  
    
    model.train()
    loader_iter = iter(loader)
    start_time = time.time()

    loss_history = []
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
        inputs = to_device_bf16(inputs=inputs, device=device)
        # 4. forward
        loss = model(inputs)
        # 5. backward + step 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # 5. log
        loss_history.append(loss.item())
        if step % config.log_every == 0:
            recent = loss_history[-50:]
            avg = sum(recent) / len(recent)
            print(f"step {step:4d}/{config.num_steps}  loss = {loss.item():.6f}  avg50 = {avg:.6f}  grad = {grad_norm.item():.3f}")
        step += 1
    end_time = time.time()
    print("Spend time: ", end_time - start_time)

    # 6. 保存 checkpoint (只存可训练的部分够推理用, 这里简单起见整模型都存)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "yuepi0_bridge.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

   

if __name__ == "__main__":

    main()
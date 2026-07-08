import sys, time
from pathlib import Path
from collections import deque 

import torch
from torch.utils.data import DataLoader  
from omegaconf import OmegaConf  
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                                                                                 

from data.bridge_dataset import BridgeDataset                                                                                                      
from model.vla.processing import VLAPreProcessor                                                                                                     
from model.vla.yuepi0 import PiZero 
from model.utils import load_paligemma_weights, to_device_bf16
from utils.optim import WarmupCosineScheduler

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

class TrainAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.flow_sampling = cfg.flow_sampling
        if self.flow_sampling == "beta":
            flow_alpha = cfg.get("flow_alpha", 1.5)
            flow_beta = cfg.get("flow_beta", 1)
            self.flow_t_max = 1 - cfg.get("flow_sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        # 训练超参
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm # 梯度裁剪阈值
        self.use_amp = cfg.get("use_amp", True) # 是否启用autocast 自动混合精度
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        self.use_torch_compile = cfg.get("use_torch_compile", True) 

        # 梯度累积
        world_size = 1 # 单卡
        # config 里global_batch_size=1024、batch_size=2, 每攒 512 个小 batch 才更新一次
        self.grad_accumulation_steps = max(cfg.global_batch_size // cfg.batch_size // world_size, 1)
        actual_global_batch_size = cfg.batch_size * self.grad_accumulation_steps * world_size
        print(f"grad_accumulation_steps = {self.grad_accumulation_steps}"                                      
                f"(per_device={cfg.batch_size}, global={actual_global_batch_size})")
        
        #  self.model = PiZero + load_paligemma + to(bf16) + freeze_vlm
        self.model = PiZero(cfg)
        if cfg.load_pretrained_weights:
            load_paligemma_weights(self.model, Path(cfg.pretrained_model_path))
        elif cfg.resume_checkpoint_path: # 从断点恢复
            self.load_checkpoint(cfg.resume_checkpoint_path)
        self.model = self.model.to(self.dtype).to(self.device)
        trainable_params, n_total, n_trainable = freeze_vlm(self.model) # 冻结vlm, 只训 action expert

        if self.use_torch_compile:
            self.model = torch.compile(
                self.model, mode="default"
            )
        # data loader
        # 2.创建dataset + dataloader
        dataset = BridgeDataset(cfg, cfg.data_dir, cfg.max_episodes)
        self.data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        # 3.创建tokenizer + processor
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path, padding_side="right")
        self.processor = VLAPreProcessor(tokenizer=self.tokenizer, num_image_token=cfg.vision_config.num_image_tokens, max_seq_len=cfg.max_seq_len)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            warmup_steps=cfg.lr_scheduler.warmup_steps,
            total_steps=self.n_updates,
            max_lr = cfg.lr,
            min_lr=cfg.lr_scheduler.min_lr)

        self.eval_freq = cfg.eval_freq
        # 留一个固定 batch 做 eval: 每次用同一包, L1 变化只反映模型权重变化
        # 注意: 真正复现要用独立 val split, 这里简化成复用训练数据 (偏乐观)         
        self.eval_batch = next(iter(self.data_loader)) 

    def run(self):
        self.model.train()
        loader_iter = iter(self.data_loader)

        # 滑动窗口: 存最近 grad_accumulation_steps 个 batch 的 loss, 用于平滑显示                                                             
        loss_deque = deque(maxlen=self.grad_accumulation_steps)
        cnt_batch = 0 # 数据 batch 计数: 每取一个 batch +1    
        cnt_update = 0 # # 优化器更新计数: 每攒满 grad_accumulation_steps 个 batch 才 +1            

        start_time = time.time()
        while cnt_update < self.n_updates:
            # 1. 取一个batch
            try:
                raw_batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.data_loader) # 数据用完了, 重新开始
                raw_batch = next(loader_iter)   
            # 2. raw -> model输入
            inputs = preprocess_batch(raw_batch=raw_batch, processor=self.processor)
            # 3. 搬到device
            inputs = to_device_bf16(inputs=inputs, device=self.device)
            # 4. forward
            bsz = inputs['input_ids'].shape[0]
            t = self.sample_flow_matching_time(bsz).to(self.device).to(self.dtype)
            loss = self.model(inputs, t)
            # 5. loss归一化 backward + step 
            normalized_loss = loss / self.grad_accumulation_steps
            normalized_loss.backward()
            loss_deque.append(loss.item())

            # 6.只在累积窗口末尾做clip + step + zero
            if (cnt_batch + 1) % self.grad_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                cnt_update += 1
                if cnt_update % self.eval_freq == 0:                                
                    eval_l1 = self.evaluate()
                    print(f"  [eval] update {cnt_update}  L1 = {eval_l1:.4f}")  
                    
                if cnt_update % self.cfg.log_every == 0:
                    avg = sum(loss_deque) / len(loss_deque)
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"update {cnt_update:4d}/{self.n_updates}  batch {cnt_batch}  "
                            f"loss = {avg:.6f}  grad = {grad_norm.item():.3f}  lr = {lr:.2e}")
            cnt_batch += 1

        end_time = time.time()
        print("Spend time: ", end_time - start_time)
        
        # 7. 保存 checkpoint (只存可训练的部分够推理用, 这里简单起见整模型都存)
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / "yuepi0_bridge.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"saved checkpoint to {ckpt_path}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            inputs = preprocess_batch(raw_batch=self.eval_batch, processor=self.processor)
            gt = inputs['action']
            pred = self.model.infer_action(inputs, num_inference_steps=self.cfg.num_inference_steps)
            l1 = (pred - gt).abs().mean()
            self.model.train()                                                          
        return l1.item() 
    
    def sample_flow_matching_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "beta":
            z = self.flow_beta_dist.sample((bsz, ))
            t = self.flow_t_max * (1 -z) 
        elif self.flow_sampling == "uniform":
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % ( 1 - eps)
        return t

    def load_checkpoint(self, path: str):
        pass

def main():
    config = OmegaConf.load("config/realdataTrain.yaml")
    OmegaConf.resolve(config)
    agent = TrainAgent(config)
    agent.run()
   

if __name__ == "__main__":

    main()
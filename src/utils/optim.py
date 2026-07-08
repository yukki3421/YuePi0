import math

class WarmupCosineScheduler:
    '''
        LR调度器: 先线性warmup 再余弦退火
        - warmup 段 (step < warmup_steps): lr 从 min_lr 线性升到 max_lr
        - cosine 段 (step >= warmup_steps): lr 从 max_lr 沿余弦降到 min_lr
    '''

    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        # 一上来就把 lr 设成 warmup 起点(≈min_lr), 否则第一次更新会直接用 max_lr
        self._set_lr(self._lr_at(self.current_step))
        
    def _lr_at(self, step):
        if step < self.warmup_steps:
            # 线性 warmup: envelope = step / warmup_steps
            return self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
        # 余弦退火
        denom = max(1, self.total_steps - self.warmup_steps)  # 防 total==warmup时除零
        progress = (step - self.warmup_steps) / denom
        return self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    
    def step(self):
        self.current_step += 1
        self._set_lr(self._lr_at(self.current_step))
    
    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]
        self._set_lr(self._lr_at(self.current_step))
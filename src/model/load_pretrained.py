import torch
from omegaconf import OmegaConf
from model.vla.yuepi0 import PiZero

def remap_key(k: str) -> str:
    '''
    把open-pi-zero checkpoint 的一个key 改写成YuePi0的命名
    四类差异(已用诊断脚本验证可 100% 覆盖 938 个 key):
          1) 顶层三件套被你包进了 embedder.*
             embed_tokens / vision_tower / multi_modal_projector
          2) joint_model.* -> joint.*
          3) proprio_encoder.{w,b} -> proprio_encoder.proj.{w,b}  (你多包了一层 self.proj)
          4) action_decoder.{w,b}  -> action_decoder.proj.{w,b}   (同上)
          其余 (action_encoder.linear_* / 所有 mixtures.layers.*) 名字本就一致, 不动。
    '''

    if k.startswith(("embed_tokens", "vision_tower", "multi_modal_projector")):
        k = "embedder." + k
    
    elif k.startswith("joint_model"):
        k = k.replace("joint_model", "joint")

    elif k in ["proprio_encoder.weight", "proprio_encoder.bias", "action_decoder.weight", "action_decoder.bias"]:
        s = k.split(".")
        s.insert(1, "proj")
        k = ".".join(s)
    return k

def remap_state_dict(ckpt_sd: dict) -> dict:
    '''对整个state_dict字典逐key改名, 返回新的dict(value 即原样tensor搬过去)'''
    return {remap_key(k): v for k, v in ckpt_sd.items()}

def load_pretrained_pizero(config, ckpt_path: str, strict: bool=True):
    """构建 PiZero, 把 open-pi-zero 权重 remap 后加载进去, 返回 model。"""
    model = PiZero(config)

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_sd = raw['model']

    yue_sd = remap_state_dict(ckpt_sd=ckpt_sd)

    missing, unexpected = model.load_state_dict(yue_sd, strict=strict)
    print(f'[load] missing={len(missing)} unexpected={len(unexpected)}')
    return model

if __name__ == "__main__":
    cfg = OmegaConf.load('config/yuepi0.yaml')
    ckpt = '/home/cxy/projects/open-pi-zero/checkpoints/bridge_beta_step19296_2024-12-26_22-30_42.pt'
    model = load_pretrained_pizero(cfg, ckpt, strict=True)
    print('OK: 权重 strict 加载成功')

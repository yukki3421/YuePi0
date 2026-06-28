import json
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from model.vla.processing import VLAPreProcessor
from utils.geometry import euler2axangle, mat2euler, quat2mat
from agent.env_adapter.base import BaseEnvAdapter

'''通用骨架: 图像处理 + tokenize + 归一化框架'''
# SimplerAdapter 是“机器人/仿真器适配”，VLAProcessor 是“VLM 输入格式适配”
class SimplerAdapter(BaseEnvAdapter):
    def __init__(
        self,
        dataset_statistics_path: str, 
        pretrained_model_path: str,
        tokenizer_padding: str,
        num_image_tokens: int,
        image_size: Tuple[int, int],
        max_seq_len: int,
        action_normal_type: str = "bound",
        proprio_normal_type: str = "bound",
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.action_normal_type = action_normal_type
        self.proprio_normal_type = proprio_normal_type
        assert action_normal_type in ["bound", "gaussian"]
        assert proprio_normal_type in ["bound", "gaussian"]

        # for normalization,  dataset_statistics:训练集的统计量(p01/p99 分位数、mean/std)。
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, padding_side="right"
        )
        self.processor = VLAPreProcessor(
            self.tokenizer,
            num_image_token=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def reset(self):
        pass
    
    # 入模型前的翻译
    def preprocess(
        self,
        env,
        obs: dict,
        instruction: str,
    ) -> dict:
        """using sxyz convention for euler angles"""
        # 从仿真 obs 抠出图像 [H,W,3]  
        image = get_image_from_maniskill2_obs_dict(env, obs) # [H, W, 3]
        # 缩到 224×224
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        images = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)[None] # [1, 3, H, W]
        model_inputs = self.processor(prompts=[instruction], images=images)
        # process proprio depending on the robot
        raw_proprio = self.preprocess_proprio(obs) # ← 抽象方法,各机器人不同

        # normalize proprios - gripper opening is normalized
        if self.proprio_normal_type == "bound":
            proprio = self.normalize_bound(
                data=raw_proprio, 
                data_min=np.array(self.dataset_statistics['proprio']['p01']), 
                data_max=np.array(self.dataset_statistics['proprio']['p99']), 
                clip_min=-1, 
                clip_max=1,
            )
        else:
            proprio = self.normalize_gaussian(
                data=raw_proprio,
                mean=np.array(self.dataset_statistics['proprio']['mean']),
                std=np.array(self.dataset_statistics["proprio"]["std"])
            )
        
        return {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "attention_mask": model_inputs["attention_mask"],
            "proprio": torch.as_tensor(proprio, dtype=torch.float32)[None, None], # [B, T, dim]
        }
    
    # 模型后的翻译
    def postprocess(self, actions: np.array) -> List[dict]:
        if self.action_normal_type == "bound":
            #  前6维反归一化, 最后一维夹爪原样拼回
            raw_actions_except_gripper = self.denormalize_bound(
                data=actions[:, :-1],
                data_min=np.array(self.dataset_statistics["action"]["p01"])[:-1],
                data_max=np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normal_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                data=actions[:, :-1],
                mean=np.array(self.dataset_statistics["action"]["mean"])[:-1],
                std=np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [raw_actions_except_gripper, actions[:, -1:],], axis=1
        )

        # 做旋转表示的转换
        actions = np.zeros((len(raw_actions), 7)) # chunk
        #  模型输出的旋转是欧拉角(roll/pitch/yaw),但 SIMPLER 要的是轴角(axis-angle)
        for idx, raw_action in enumerate(raw_actions):
            roll, pitch, yaw = raw_action[3:6]
            action_rotation_ax, action_rotatation_angle = euler2axangle(roll, pitch, yaw)
            action_gripper = self.postprocess_gripper(raw_action[-1])

            actions[idx] = np.concatenate(
                [raw_action[:3], action_rotation_ax * action_rotatation_angle, [action_gripper],]
            )
        return actions
    
    def preprocess_proprio(self, obs: dict) -> np.array:
        raise NotImplementedError
    
    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError
    
    def get_video_frame(self, env, obs: dict) -> np.array:
        return get_image_from_maniskill2_obs_dict(env, obs)

class BridgeSimplerAdapter(SimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #  3×3 旋转矩阵
        #  Bridge 数据集采集时,机械臂末端(end-effector, EE)的旋转不是相对机器人基座记录的,
        # 而是相对一个"俯视"(top-down)基准姿态。
        # 仿真器(SIMPLER)给你的 obs["agent"]["eef_pos"]                       
        # 里的旋转是相对基座的。两个参考系差了一个固定旋转,就是这个
        # default_rot。如果不转换,模型看到的姿态分布跟训练时差了一整个旋转,直接乱掉。
        self.default_rot = np.array([[0, 0, 1.0,], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset(self):
        super().reset()
    
    def preprocess_proprio(self, obs) -> np.array:
        proprio = obs["agent"]["eef_pos"] # [x,y,z, qx,qy,qz,qw, gripper] 共8维
        rm_bridge = quat2mat(proprio[3:7]) # ① 四元数 → 旋转矩阵 
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T) # ② 换参考系 + 转欧拉角
        gripper_openness = proprio[7] # ③ 取夹爪开度
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def postprocess_gripper(self, action:float):
        # 训练时夹爪是 [0,1]:0=闭合,1=张开
        # SIMPLER 要 {-1, +1}: -1=闭合, +1=张开                                                  
        # - action > 0.5 → 布尔(True/False),即二值化:大于 0.5 算"开"
        # - 2.0 * True - 1.0 = 1.0(开),2.0 * False - 1.0 = -1.0(闭)
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper
    

    

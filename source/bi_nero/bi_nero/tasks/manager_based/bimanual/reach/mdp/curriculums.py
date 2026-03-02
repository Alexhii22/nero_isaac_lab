from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def modify_cuboid_offset(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    target_offset: float, 
    num_steps: int, 
    command_name: str
):
    """逐渐减小引导长方体的偏移量 (Curriculum)。
    
    该函数会根据当前的训练步数，从配置的初始偏移量线性插值到目标偏移量。
    
    Args:
        env: 环境对象。
        env_ids: 受影响的环境 ID。
        target_offset: 最终的目标偏移量 (m)。
        num_steps: 达到目标偏移量所需的总步数 (total world steps)。
        command_name: 命令管理器的名称 (例如 "left_sweep_command")。
    """
    # 获取当前的训练步数
    current_step = env.common_step_counter
    
    # 获取命令处理器
    command_term = env.command_manager.get_term(command_name)
    
    # 初始偏移量从配置中获取
    start_offset = command_term.cfg.cuboid_offset
    
    # 计算插值比例
    alpha = min(current_step / num_steps, 1.0)
    
    # 计算当前步数对应的偏移量
    current_offset = start_offset + alpha * (target_offset - start_offset)
    
    # 更新 command_term 的内部偏移量张量 (用于更新长方体物理位置)
    device = env.device
    command_term.cuboid_offsets = torch.tensor([
        [0.0,  current_offset, 0.0],  # Y+ (Cuboid 1)
        [0.0, -current_offset, 0.0],  # Y- (Cuboid 2)
        [current_offset, 0.0, 0.0],  # X+ (Cuboid 3)
        [-current_offset, 0.0, 0.0], # X- (Cuboid 4)
    ], device=device, dtype=torch.float32)

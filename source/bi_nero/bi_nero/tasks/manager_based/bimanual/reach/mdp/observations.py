# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""观测：世界坐标系下的末端关键点、末端/目标位置、关节绝对位置与速度，便于 sim2real。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_mul

from .rewards import get_keypoint_offsets_full_6d

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _keypoints_in_world(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    keypoint_scale: float,
    add_negative_axes: bool,
    device: torch.device,
) -> torch.Tensor:
    """将末端坐标系下 3 个正轴关键点变换到世界系。返回 (num_envs, 9)。"""
    offsets = get_keypoint_offsets_full_6d(
        add_cube_center_kp=False,
        add_negative_axes=add_negative_axes,
        device=device,
    ) * keypoint_scale
    num_envs = pos_w.shape[0]
    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0).expand(num_envs, 4)
    kp_world = []
    for i in range(offsets.shape[0]):
        offset = offsets[i].unsqueeze(0).expand(num_envs, 3)
        p, _ = combine_frame_transforms(pos_w, quat_w, offset, identity_quat)
        kp_world.append(p)
    return torch.cat(kp_world, dim=-1)


# ----- 世界坐标系观测（sim2real 友好）-----

def obs_ee_keypoints_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 0.25,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """末端三轴关键点在世界系下的坐标（仅正轴 3 点：X+/Y+/Z+）。

    shape (num_envs, 9)，即 3 点 × (x,y,z) 单位 m。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return _keypoints_in_world(
        curr_pos_w, curr_quat_w, keypoint_scale, add_negative_axes, curr_pos_w.device
    )


def obs_target_keypoints_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 0.25,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """目标三轴关键点在世界系下的坐标（仅正轴 3 点）。

    shape (num_envs, 9)，单位 m。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    return _keypoints_in_world(
        des_pos_w, des_quat_w, keypoint_scale, add_negative_axes, des_pos_w.device
    )


def obs_keypoints_error_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 0.25,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """关键点误差（世界系）：target_keypoints - ee_keypoints。

    让策略显式看到每个关键点的 (x,y,z) 误差，避免只靠标量距离导致“混在一起”的梯度弱问题。

    shape (num_envs, 9)，单位 m。
    """
    ee = obs_ee_keypoints_world(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        keypoint_scale=keypoint_scale,
        add_negative_axes=add_negative_axes,
    )
    tgt = obs_target_keypoints_world(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        keypoint_scale=keypoint_scale,
        add_negative_axes=add_negative_axes,
    )
    return tgt - ee


def obs_ee_position_world(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """末端位置在世界系下的坐标。shape (num_envs, 3)，单位 m。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore


def obs_target_position_world(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """目标位置在世界系下的坐标。shape (num_envs, 3)，单位 m。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    return des_pos_w


def _get_joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """从 asset 解析 joint_ids（框架可能已设置，否则用 find_joints）。"""
    asset = env.scene[asset_cfg.name]
    if hasattr(asset_cfg, "joint_ids") and getattr(asset_cfg, "joint_ids", None) is not None:
        return asset_cfg.joint_ids  # type: ignore
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names, preserve_order=True)  # type: ignore
    return joint_ids


def obs_joint_pos_absolute(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """关节当前位置（绝对角度），单位 rad。shape (num_envs, joint_dim)。"""
    asset = env.scene[asset_cfg.name]
    joint_ids = _get_joint_ids(env, asset_cfg)
    return asset.data.joint_pos[:, joint_ids]  # type: ignore


def obs_joint_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """关节当前速度，单位 rad/s。shape (num_envs, joint_dim)。"""
    asset = env.scene[asset_cfg.name]
    joint_ids = _get_joint_ids(env, asset_cfg)
    return asset.data.joint_vel[:, joint_ids]  # type: ignore


def obs_joint_prev_pos(
    env: ManagerBasedRLEnv,
    action_name: str,
    asset_cfg: SceneEntityCfg,
    default_joint_pos: list[float],
) -> torch.Tensor:
    """上一时刻关节位置（绝对）：用上一步指令目标近似，即 default + processed_action。

    shape (num_envs, joint_dim)，单位 rad。"""
    term = env.action_manager.get_term(action_name)
    processed = term.processed_actions
    device = processed.device
    default = torch.tensor(
        default_joint_pos, device=device, dtype=processed.dtype
    ).unsqueeze(0).expand(processed.shape[0], -1)
    return default + processed

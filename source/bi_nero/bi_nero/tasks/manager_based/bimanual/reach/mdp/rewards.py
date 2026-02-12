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

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_keypoint_offsets_full_6d(
    add_cube_center_kp: bool = True,
    add_negative_axes: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Keypoint offsets for pose alignment: axis-aligned points.

    - 3 points (X/Y/Z 正轴): add_cube_center_kp=False, add_negative_axes=False
      末端与目标在 x/y/z 三轴上的点对齐即可唯一确定位姿。
    - 6 points (正负轴): add_cube_center_kp=False, add_negative_axes=True
    - 7 points (中心+正负轴): add_cube_center_kp=True, add_negative_axes=True (默认)

    Returns:
        Keypoint offsets (num_keypoints, 3). 3 / 6 / 7 points.
    """
    # 正轴三点：X+ [1,0,0], Y+ [0,1,0], Z+ [0,0,1]
    corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if add_negative_axes:
        corners = corners + [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    keypoints = torch.tensor(corners, device=device, dtype=torch.float32)
    if add_cube_center_kp:
        center = torch.zeros((1, 3), device=device, dtype=torch.float32)
        keypoints = torch.cat((center, keypoints), dim=0)
    return keypoints

def compute_keypoint_distance(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """L2 distance between corresponding keypoints of current and target poses.

    Keypoints are axis-aligned offsets transformed to world by each pose.
    Returns shape (num_envs, num_keypoints). num_keypoints = 3 / 6 / 7.
    """
    device = current_pos.device
    num_envs = current_pos.shape[0]
    offsets = get_keypoint_offsets_full_6d(
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
        device=device,
    ) * keypoint_scale
    num_kp = offsets.shape[0]
    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0).expand(num_envs, 4)

    keypoints_curr = torch.zeros((num_envs, num_kp, 3), device=device, dtype=current_pos.dtype)
    keypoints_tgt = torch.zeros((num_envs, num_kp, 3), device=device, dtype=target_pos.dtype)

    for i in range(num_kp):
        offset = offsets[i].unsqueeze(0).expand(num_envs, 3)
        keypoints_curr[:, i], _ = combine_frame_transforms(
            current_pos, current_quat, offset, identity_quat
        )
        keypoints_tgt[:, i], _ = combine_frame_transforms(
            target_pos, target_quat, offset, identity_quat
        )

    return torch.norm(keypoints_tgt - keypoints_curr, p=2, dim=-1)
#计算各点之间的误差


def compute_keypoint_delta(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """Vector difference between corresponding keypoints of target and current poses.

    Returns shape (num_envs, num_keypoints, 3) for x/y/z axis-wise errors.
    """
    device = current_pos.device
    num_envs = current_pos.shape[0]
    offsets = get_keypoint_offsets_full_6d(
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
        device=device,
    ) * keypoint_scale
    num_kp = offsets.shape[0]
    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).unsqueeze(0).expand(num_envs, 4)

    keypoints_curr = torch.zeros((num_envs, num_kp, 3), device=device, dtype=current_pos.dtype)
    keypoints_tgt = torch.zeros((num_envs, num_kp, 3), device=device, dtype=target_pos.dtype)
    for i in range(num_kp):
        offset = offsets[i].unsqueeze(0).expand(num_envs, 3)
        keypoints_curr[:, i], _ = combine_frame_transforms(
            current_pos, current_quat, offset, identity_quat
        )
        keypoints_tgt[:, i], _ = combine_frame_transforms(
            target_pos, target_quat, offset, identity_quat
        )
    return keypoints_tgt - keypoints_curr


def keypoint_command_error_axis(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    axis: int | str,
    keypoint_scale: float = 0.25,
    add_cube_center_kp: bool = True,
    add_negative_axes: bool = True,
) -> torch.Tensor:
    """该轴关键点与目标的对齐误差（世界系）。

    关键点顺序：0=X+，1=Y+，2=Z+。只取该轴对应关键点的误差，不混入其它轴关键点，
    否则 Y/Z 关键点的 x 分量接近 0 会把“x 关键点很远”的真实误差摊薄成接近 0。

    Args:
        axis: 0/1/2 或 "x"/"y"/"z"

    Returns:
        (num_envs,) 该轴关键点与目标的 L2 距离 (m)。
    """
    if isinstance(axis, str):
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = axis_map[axis.lower()]
    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError(f"axis must be 0/1/2 or 'x'/'y'/'z', got {axis}")

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    delta = compute_keypoint_delta(
        curr_pos_w,
        curr_quat_w,
        des_pos_w,
        des_quat_w,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
    )
    # 只用该轴对应关键点（0=X+，1=Y+，2=Z+）的 L2 误差，不平均三个关键点
    return torch.norm(delta[:, axis, :], p=2, dim=-1)


def keypoint_command_error_axis_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    axis: int | str,
    keypoint_scale: float = 0.25,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """Axis-wise tanh-kernel reward: 1 - tanh(|axis_error| / std)."""
    d_axis = keypoint_command_error_axis(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        axis=axis,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
    )
    return 1.0 - torch.tanh(d_axis / std)


def reach_success_sparse_keypoints(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_thresh: float = 0.01,
    keypoint_scale: float = 0.25,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """稀疏奖励：三个关键点误差同时 < keypoint_thresh(m) 时给 1，否则 0。

    用于鼓励高精度对齐（如三关键点均 < 1cm）。
    Returns (num_envs,) 取值 0 或 1。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore

    delta = compute_keypoint_delta(
        curr_pos_w,
        curr_quat_w,
        des_pos_w,
        des_quat_w,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
    )
    dist_per_kp = torch.norm(delta, p=2, dim=-1)  # (num_envs, num_kp)
    all_under = (dist_per_kp < keypoint_thresh).all(dim=-1)
    return all_under.float()


# 关键点距离奖励
def keypoint_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """Mean keypoint distance between current and desired pose (L2 per keypoint, then mean).

    Pose aligned if all axis-aligned keypoints align. Returns (num_envs,) for use as penalty.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # typeignore末端当前世界坐标位置
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # 末端当前世界坐标位置

    dist_sep = compute_keypoint_distance(
        curr_pos_w, curr_quat_w, des_pos_w, des_quat_w,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
    )
    return dist_sep.mean(dim=-1)


def keypoint_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = False,
    add_negative_axes: bool = False,
) -> torch.Tensor:
    """关键点距离的 tanh kernel 奖励（参考 openarm 成熟设计）。

    先计算关键点平均距离 d（单位 m），再映射为 reward:
        r = 1 - tanh(d / std)
    - d 越小，r 越接近 1（细粒度）
    - d 越大，r 趋近 0（仍有区分度，且有界）

    Returns (num_envs,) in (0, 1].
    """
    d = keypoint_command_error(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        add_negative_axes=add_negative_axes,
    )
    return 1.0 - torch.tanh(d / std)

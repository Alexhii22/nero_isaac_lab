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

"""MAPPO 专用环境配置：去中心化 policy obs（30D×2）+ 集中化 critic 状态（60D），可选动作延迟（sim2real）。"""

from isaaclab.utils import configclass

from .joint_pos_env_cfg import BiNeroReachEnvCfg


@configclass
class BiNeroReachMAPPOEnvCfg(BiNeroReachEnvCfg):
    """MAPPO 专用配置。

    在 BiNeroReachEnvCfg 基础上：
    - 动作延迟 3ms，用于 sim2real 训练时模拟命令延迟
    - 覆盖三个新 ObsGroup 的 body_names / default_joint_pos
    """

    action_delay_seconds: float = 0.008  # 3ms，模拟真实机器人命令延迟

    def __post_init__(self):
        super().__post_init__()  # 完成 BiNeroReachEnvCfg 的基础覆盖（actions, commands, rewards 等）

        # ----------------------------------------------------------------
        # policy_left group (30D)：左臂局部观测
        # ----------------------------------------------------------------
        obs_pl = self.observations.policy_left
        obs_pl.left_keypoints_error_world.params["asset_cfg"].body_names = ["left_link7"]
        obs_pl.left_joint_prev_pos.params["default_joint_pos"] = [
            1.0, 1.0, -1.0, 1.5, 1.0, 0.0, 0.0,
        ]

        # ----------------------------------------------------------------
        # policy_right group (30D)：右臂局部观测
        # ----------------------------------------------------------------
        obs_pr = self.observations.policy_right
        obs_pr.right_keypoints_error_world.params["asset_cfg"].body_names = ["right_link7"]
        obs_pr.right_joint_prev_pos.params["default_joint_pos"] = [
            -1.0, 1.0, 1.0, 1.5, -1.0, 0.0, 0.0,
        ]

        # ----------------------------------------------------------------
        # critic group (60D)：全局状态，供集中化 critic 使用
        # ----------------------------------------------------------------
        obs_c = self.observations.critic
        obs_c.left_keypoints_error_world.params["asset_cfg"].body_names  = ["left_link7"]
        obs_c.right_keypoints_error_world.params["asset_cfg"].body_names = ["right_link7"]
        obs_c.left_joint_prev_pos.params["default_joint_pos"] = [
            1.0, 1.0, -1.0, 1.5, 1.0, 0.0, 0.0,
        ]
        obs_c.right_joint_prev_pos.params["default_joint_pos"] = [
            -1.0, 1.0, 1.0, 1.5, -1.0, 0.0, 0.0,
        ]


@configclass
class BiNeroReachMAPPOEnvCfg_PLAY(BiNeroReachMAPPOEnvCfg):
    """MAPPO play 专用：小场景、无延迟、无观测噪声，引导距离直接为 0.05。"""

    def __post_init__(self):
        super().__post_init__()
        self.action_delay_seconds = 0.0
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy_left.enable_corruption = False
        self.observations.policy_right.enable_corruption = False
        self.observations.critic.enable_corruption = False
        # 直接使用收窄后的引导距离 0.05，便于查看课程结束效果
        self.commands.left_ee_pose.cuboid_offset = 0.05
        self.commands.right_ee_pose.cuboid_offset = 0.05

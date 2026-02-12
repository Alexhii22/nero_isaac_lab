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

import math

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from .. import mdp
from ..reach_env_cfg import (
    ReachEnvCfg,
)

from bi_nero.assets.bi_nero import BI_NERO_HIGH_PD_CFG
from isaaclab.assets.articulation import ArticulationCfg

##
# Environment configuration
##


@configclass
class BiNeroReachEnvCfg(ReachEnvCfg):
    """Bi-Nero 双臂 reach 环境配置。

    action_delay_seconds: 动作延迟（秒），用于 sim2real 训练时模拟命令延迟。
        5ms 在 30Hz 下约 1 步延迟（ceil(0.005/0.033)≈1）。
    """

    action_delay_seconds: float = 0.005  # 5ms，模拟真实机器人命令延迟

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Bi-Nero bimanual
        self.scene.robot = BI_NERO_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "left_joint1": 1.6,
                    "left_joint2": 1.2,
                    "left_joint3": 0.52,
                    "left_joint4": 0.52,
                    "left_joint5": -0.6,
                    "left_joint6": 0.0,
                    "left_joint7": 0.0,
                    "right_joint1": -1.6,
                    "right_joint2": 1.2,
                    "right_joint3": -0.52,
                    "right_joint4": 0.52,
                    "right_joint5": 0.6,
                    "right_joint6": 0.0,
                    "right_joint7": 0.0,
                },  # Close the gripper
            ),
        )

        # override rewards：左臂 left_link7，右臂 right_link7
        for name in (
            "left_keypoint_error_x",
            "left_keypoint_error_y",
            "left_keypoint_error_z",
            "left_keypoint_tracking_tanh_x",
            "left_keypoint_tracking_tanh_y",
            "left_keypoint_tracking_tanh_z",
            "left_reach_success_sparse",
        ):
            getattr(self.rewards, name).params["asset_cfg"].body_names = ["left_link7"]
        for name in (
            "right_keypoint_error_x",
            "right_keypoint_error_y",
            "right_keypoint_error_z",
            "right_keypoint_tracking_tanh_x",
            "right_keypoint_tracking_tanh_y",
            "right_keypoint_tracking_tanh_z",
            "right_reach_success_sparse",
        ):
            getattr(self.rewards, name).params["asset_cfg"].body_names = ["right_link7"]

        # 观测：世界系关键点（body_names）
        getattr(self.observations.policy, "left_keypoints_error_world").params["asset_cfg"].body_names = ["left_link7"]
        getattr(self.observations.policy, "right_keypoints_error_world").params["asset_cfg"].body_names = ["right_link7"]

        # 关节上一时刻位置：默认关节角（与 init_state 一致）
        self.observations.policy.left_joint_prev_pos.params["default_joint_pos"] = [
            1.6, 1.2, 0.52, 0.52, -0.6, 0.0, 0.0,
        ]
        self.observations.policy.right_joint_prev_pos.params["default_joint_pos"] = [
            -1.6, 1.2, -0.52, 0.52, 0.6, 0.0, 0.0,
        ]

        # override actions
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                "left_joint5", "left_joint6", "left_joint7",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                "right_joint5", "right_joint6", "right_joint7",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body（末端连杆名与 USD 一致）
        self.commands.left_ee_pose.body_name = "left_link7"
        self.commands.right_ee_pose.body_name = "right_link7"

        # 关节限位内均匀随机化初始姿态，提升 sim2real 泛化能力
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        )


@configclass
class BiNeroReachEnvCfg_PLAY(BiNeroReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # play 时关闭动作延迟，便于直接观看 sim 理想行为
        self.action_delay_seconds = 0.0
        # play 时使用固定/缩放初始姿态，便于一致评估
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        )
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

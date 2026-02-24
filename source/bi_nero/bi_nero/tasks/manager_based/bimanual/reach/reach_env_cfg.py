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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.25, -0.15),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 9, math.pi / 9),#恒定朝向 （-30 30)
            pitch=(3 * math.pi / 2+math.pi/2, 3 * math.pi / 2+math.pi/2),#360
            yaw=(math.pi+9.5 * math.pi / 10, 2*math.pi),#351-360
        ),
    )
    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(0.15, 0.25),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 9, math.pi / 9),
            pitch=(math.pi+3 * math.pi / 2 + math.pi/2, math.pi + 3 * math.pi / 2 + math.pi/2),
            yaw=(9.6 * math.pi / 10, 10.4 * math.pi / 10),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """观测：双臂，世界坐标系 + 关节绝对位置，便于 sim2real。

        - 左/右：关键点误差(世界) 9D、关节当前位置、速度、上一时刻位置
        """

        # ----- 左臂：末端关键点(世界) 9D -----
        # left_ee_keypoints_world = ObsTerm(
        #     func=mdp.obs_ee_keypoints_world,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
        #         "command_name": "left_ee_pose",
        #         "keypoint_scale": 0.25,
        #         "add_negative_axes": False,
        #     },
        #     noise=Unoise(n_min=-0.001, n_max=0.001),
        # )
        # left_target_keypoints_world = ObsTerm(
        #     func=mdp.obs_target_keypoints_world,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
        #         "command_name": "left_ee_pose",
        #         "keypoint_scale": 0.25,
        #         "add_negative_axes": False,
        #     },
        #     noise=Unoise(n_min=-0.001, n_max=0.001),
        # )
        # 关键点误差(世界) 9D：让策略显式知道每个轴向差距
        left_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "left_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        # ----- 右臂：关键点误差 9D + 关节状态 -----
        right_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "right_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        left_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        left_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.008, n_max=0.008),
        )
        left_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "left_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )
        right_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        right_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.008, n_max=0.008),
        )
        right_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "right_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyLeftCfg(ObsGroup):
        """左臂局部观测，30D，仅供 left policy 使用（MAPPO 去中心化）。"""

        left_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "left_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        left_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        left_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.008, n_max=0.008),
        )
        left_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "left_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyRightCfg(ObsGroup):
        """右臂局部观测，30D，仅供 right policy 使用（MAPPO 去中心化）。"""

        right_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "right_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        right_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        right_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
            noise=Unoise(n_min=-0.008, n_max=0.008),
        )
        right_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "right_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """集中化 critic 全局状态，60D = left(30) + right(30)，无噪声。"""

        # --- left 部分 ---
        left_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "left_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
        )
        left_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
        )
        left_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
            },
        )
        left_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "left_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                    "left_joint5", "left_joint6", "left_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )
        # --- right 部分 ---
        right_keypoints_error_world = ObsTerm(
            func=mdp.obs_keypoints_error_world,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "command_name": "right_ee_pose",
                "keypoint_scale": 0.25,
                "add_negative_axes": False,
            },
        )
        right_joint_pos = ObsTerm(
            func=mdp.obs_joint_pos_absolute,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
        )
        right_joint_vel = ObsTerm(
            func=mdp.obs_joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
            },
        )
        right_joint_prev_pos = ObsTerm(
            func=mdp.obs_joint_prev_pos,
            params={
                "action_name": "right_arm_action",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                    "right_joint5", "right_joint6", "right_joint7",
                ]),
                "default_joint_pos": MISSING,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    policy_left:  PolicyLeftCfg  = PolicyLeftCfg()
    policy_right: PolicyRightCfg = PolicyRightCfg()
    critic:       CriticCfg      = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    
    """双臂：reward 基于关键点距离（左/右臂对称设计，参考 openarm 的 tanh kernel 思路）。"""

    # 轴向线性惩罚（密集）：权重加大，避免“到一定距离就不往近了走”
    # keypoint_scale：关键点沿轴长度(m)，即末端+scale*轴方向的点，非精度阈值
    left_keypoint_error_x = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.65,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "axis": "x",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    left_keypoint_error_y = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "axis": "y",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    left_keypoint_error_z = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.65,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "axis": "z",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )

    # 轴向 tanh：std 调小，近距离梯度才够大，否则会停在“恒定距离”不再靠近
    left_keypoint_tracking_tanh_x = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,  
            "command_name": "left_ee_pose",
            "axis": "x",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    left_keypoint_tracking_tanh_y = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "left_ee_pose",
            "axis": "y",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    left_keypoint_tracking_tanh_z = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "left_ee_pose",
            "axis": "z",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )

    # 稀疏奖励：三关键点误差同时 < 1cm 时给固定奖励，鼓励高精度
    left_reach_success_sparse = RewTerm(
        func=mdp.reach_success_sparse_keypoints,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
            "keypoint_thresh": 0.02,
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )

    # ----- 右臂：轴向惩罚 + tanh + 稀疏 + 关节速度 -----
    right_keypoint_error_x = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.65,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "axis": "x",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_keypoint_error_y = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "axis": "y",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_keypoint_error_z = RewTerm(
        func=mdp.keypoint_command_error_axis,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "axis": "z",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_keypoint_tracking_tanh_x = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "right_ee_pose",
            "axis": "x",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_keypoint_tracking_tanh_y = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "right_ee_pose",
            "axis": "y",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_keypoint_tracking_tanh_z = RewTerm(
        func=mdp.keypoint_command_error_axis_tanh,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.05,
            "command_name": "right_ee_pose",
            "axis": "z",
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )
    right_reach_success_sparse = RewTerm(
        func=mdp.reach_success_sparse_keypoints,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
            "keypoint_thresh": 0.02,
            "keypoint_scale": 0.25,
            "add_cube_center_kp": False,
            "add_negative_axes": False,
        },
    )

    # 平滑项（参考 openarm）：抑制抖动，课程中逐步加大权重
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_joint1", "left_joint2", "left_joint3", "left_joint4",
                "left_joint5", "left_joint6", "left_joint7",
            ]),
        },
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "right_joint1", "right_joint2", "right_joint3", "right_joint4",
                "right_joint5", "right_joint6", "right_joint7",
            ]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """平滑项课程（参考 openarm）：前期弱惩罚，后期逐步加大 action_rate / joint_vel 权重抑制抖动。"""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.1, "num_steps": 3000},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 4500},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 4500},
    )

##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 30Hz 控制，平衡精度与平滑（60Hz易抖动）
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

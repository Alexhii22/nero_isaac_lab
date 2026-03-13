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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

# The USD asset is located in fairino_description/urdf/fairino5_v6
# We assume the script is running from the project root or the path is reachable
# For robustness, we can try to find the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
NERO_USD_PATH = os.path.join(PROJECT_ROOT, "agx_arm/bi_nero_description/urdf/bi_nero_description/bi_nero_description.usd")

BI_NERO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=NERO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=3,
        ),
    ),
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
        },
    ),
    actuators={
        "bi_nero_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-7]", "right_joint[1-7]"],
            velocity_limit_sim=1.3,  # 最大关节角速度 (rad/s)，调小→整体更慢，原 2.175
            effort_limit_sim={
                "left_joint[1-7]": 40.0,
                "right_joint[1-7]": 40.0,
            },
            stiffness=100.0,#P值 刚度 越大 → 越“硬”，越想马上到目标 越小 → 越“软”，动作慢但稳定
            damping=80.0,#抑制速度，防止震荡
        ),
    },
    soft_joint_pos_limit_factor=1.0,#完全使用URDF限位
)

"""Configuration of NERO robot."""

BI_NERO_HIGH_PD_CFG = BI_NERO_CFG.copy()
BI_NERO_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
BI_NERO_HIGH_PD_CFG.actuators["bi_nero_arm"].stiffness = 500.0
BI_NERO_HIGH_PD_CFG.actuators["bi_nero_arm"].damping = 35.0
"""Configuration of Bi-Nero bimanual robot with stiffer PD control."""


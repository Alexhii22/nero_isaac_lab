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

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EMAJointPositionAction(JointPositionAction):
    """Joint position action term that applies EMA smoothing."""

    def __init__(self, cfg: EMAJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 解析 alpha
        if isinstance(cfg.alpha, float):
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        
        # 初始化上一时刻动作
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        # 重置时，将上一时刻动作初始化为当前真实关节位置，防止突跳
        if env_ids is None:
            self._prev_applied_actions[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            self._prev_applied_actions[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # 1. 基础处理 (scale, offset, clip)
        super().process_actions(actions)
        
        # 2. 应用 EMA 平滑: y = alpha * x + (1 - alpha) * y_prev
        self._processed_actions[:] = self._alpha * self._processed_actions + (1.0 - self._alpha) * self._prev_applied_actions
        
        # 3. 更新缓存
        self._prev_applied_actions[:] = self._processed_actions[:]


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for joint position action term with EMA smoothing."""

    class_type: type = EMAJointPositionAction
    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average [0, 1]. 1.0 means no smoothing."""

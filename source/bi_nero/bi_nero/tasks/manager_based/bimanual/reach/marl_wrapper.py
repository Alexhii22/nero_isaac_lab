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

"""Multi-agent wrapper：将单 agent ManagerBasedRLEnv 适配为 SKRL MAPPO 期望的接口。

SKRL multi_agent_train() 的调用约定：
  - env.num_agents > 1           → 选择 multi_agent_train 分支
  - env.state()                  → 返回全局 critic 状态 [N, 60]
  - env.reset() → (obs_dict, info)   obs_dict = {"left": ..., "right": ...}
  - env.step(actions_dict)           actions_dict = {"left": ..., "right": ...}
  - env.agents                   → 活跃 agent 列表（非空表示 episode 未结束）

数据映射：
  obs["policy_left"]  (30D) → obs_dict["left"]    → policy_left  输入
  obs["policy_right"] (30D) → obs_dict["right"]   → policy_right 输入
  obs["critic"]       (60D) → env.state()         → shared_value 输入
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from typing import Any


class BimanualMARLWrapper:
    """将 ManagerBasedRLEnv 包装为 SKRL MAPPO 的 multi-agent 接口。

    关键：SKRL multi_agent_train() 在 reset/step 后单独调用 env.state()
    获取全局状态，而不是从 obs_dict 里取。因此：
    - observation_spaces 只包含局部 obs 空间（不含 states）
    - shared_observation_spaces 包含全局状态空间（供 MAPPO 构造函数使用）
    - state() 返回最近一步的 critic obs [N, 60]
    """

    possible_agents = ["left", "right"]

    OBS_LEFT_DIM  = 30
    OBS_RIGHT_DIM = 30
    CRITIC_DIM    = 60
    ACT_DIM       = 7

    def __init__(self, env: Any) -> None:
        self.env = env
        inf = float("inf")

        # 各 agent 局部 obs 空间（policy 输入）
        self.observation_spaces: dict[str, gym.Space] = {
            "left":  gym.spaces.Box(-inf, inf, shape=(self.OBS_LEFT_DIM,),  dtype=np.float32),
            "right": gym.spaces.Box(-inf, inf, shape=(self.OBS_RIGHT_DIM,), dtype=np.float32),
        }
        # 全局状态空间（集中化 critic 输入），每个 agent 视角相同
        self.shared_observation_spaces: dict[str, gym.Space] = {
            "left":  gym.spaces.Box(-inf, inf, shape=(self.CRITIC_DIM,), dtype=np.float32),
            "right": gym.spaces.Box(-inf, inf, shape=(self.CRITIC_DIM,), dtype=np.float32),
        }
        self.action_spaces: dict[str, gym.Space] = {
            "left":  gym.spaces.Box(-1.0, 1.0, shape=(self.ACT_DIM,), dtype=np.float32),
            "right": gym.spaces.Box(-1.0, 1.0, shape=(self.ACT_DIM,), dtype=np.float32),
        }

        self.num_envs: int = env.unwrapped.num_envs
        self.device:   str = env.unwrapped.device

        # SKRL SequentialTrainer 检查 env.num_agents > 1 来选择 multi_agent_train
        self.num_agents: int = 2

        # 活跃 agent 列表（SKRL 用 `not self.env.agents` 判断是否 episode 结束）
        self._agents: list[str] = list(self.possible_agents)

        # reward 拆分 index，懒初始化（第一次 step 后建立）
        self._left_idx:   list[int] | None = None
        self._right_idx:  list[int] | None = None
        self._shared_idx: list[int] | None = None

        # 最近一次的原始 obs，供 state() 方法提取全局状态
        self._last_raw_obs: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # SKRL multi_agent_train() 要求的接口
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[str]:
        """返回当前活跃的 agent 列表。
        SKRL 用 `not self.env.agents` 判断是否需要 reset。
        Isaac Lab 的 reset 由 terminated/truncated 驱动，这里统一返回所有 agent。
        """
        return self._agents

    def state(self) -> torch.Tensor:
        """返回集中化 critic 的全局状态 [N, 60]。
        SKRL multi_agent_train() 在 reset() 和 step() 后分别调用此方法，
        分别获取 shared_states（当前）和 shared_next_states（下一步）。
        """
        assert self._last_raw_obs is not None, "state() called before reset()"
        return self._last_raw_obs["critic"]  # [N, 60]

    # ------------------------------------------------------------------
    # Standard gym-like multi-agent 接口
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        raw_obs, info = self.env.reset()
        self._last_raw_obs = raw_obs
        self._agents = list(self.possible_agents)
        return self._split_obs(raw_obs), info

    def step(
        self,
        actions: dict[str, torch.Tensor],
    ) -> tuple[dict, dict, dict, dict, dict]:
        # 1. 合并：left(0-6) 在前，right(7-13) 在后，与 ActionsCfg 声明顺序一致
        combined = torch.cat([actions["left"], actions["right"]], dim=-1)  # [N, 14]

        # 2. 底层 step
        raw_obs, combined_reward, terminated, truncated, info = self.env.step(combined)

        # 3. 保存最新 obs（供 state() 在 step 后被调用时使用）
        self._last_raw_obs = raw_obs

        # 4. 拆分局部 obs（不含全局状态，全局状态由 state() 单独返回）
        obs_dict = self._split_obs(raw_obs)

        # 5. 拆分 reward
        reward_dict = self._split_reward(combined_reward)

        # 6. 统一终止（单仿真体，两 agent 同生同灭）
        # Isaac Lab 返回 [N] 的 1D tensor，SKRL memory 期望 [N, 1]，需要 unsqueeze
        terminated_1d = terminated.view(-1, 1)
        truncated_1d  = truncated.view(-1, 1)
        terminated_dict = {"left": terminated_1d, "right": terminated_1d}
        truncated_dict  = {"left": truncated_1d,  "right": truncated_1d}

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _split_obs(self, raw_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """只返回各 agent 的局部 obs，全局状态通过 state() 单独获取。"""
        return {
            "left":  raw_obs["policy_left"],   # [N, 30]
            "right": raw_obs["policy_right"],  # [N, 30]
        }

    def _split_reward(self, combined_reward: torch.Tensor) -> dict[str, torch.Tensor]:
        rm = self.env.unwrapped.reward_manager

        # 懒初始化：按 term 名前缀建立列 index 映射
        if self._left_idx is None:
            names = rm._term_names
            self._left_idx   = [i for i, n in enumerate(names) if n.startswith("left_")]
            self._right_idx  = [i for i, n in enumerate(names) if n.startswith("right_")]
            self._shared_idx = [
                i for i, n in enumerate(names)
                if not n.startswith("left_") and not n.startswith("right_")
            ]

        # _step_reward: [N, num_terms]，reward_manager.compute() 在 env.step() 内已填充
        step_r = rm._step_reward

        r_left_terms  = step_r[:, self._left_idx].sum(dim=1)
        r_right_terms = step_r[:, self._right_idx].sum(dim=1)
        r_shared      = step_r[:, self._shared_idx].sum(dim=1)

        r_left  = (r_left_terms  + r_shared * 0.5).unsqueeze(-1)  # [N, 1]
        r_right = (r_right_terms + r_shared * 0.5).unsqueeze(-1)  # [N, 1]

        return {"left": r_left, "right": r_right}

    # ------------------------------------------------------------------
    # 属性透传
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def close(self) -> None:
        self.env.close()

    def render(self) -> Any:
        return self.env.render()

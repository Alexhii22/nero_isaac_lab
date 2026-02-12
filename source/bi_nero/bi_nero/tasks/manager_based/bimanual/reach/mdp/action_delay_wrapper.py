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

"""Gymnasium wrapper that introduces action delay for sim2real training."""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import torch


class ActionDelayWrapper(gym.Wrapper):
    """Delays actions by a configurable number of steps to simulate real-robot command latency.

    When ``delay_steps >= 1`` the wrapper buffers incoming actions and replays older
    actions so that the environment always executes an action that is ``delay_steps``
    steps old.  This is useful for sim2real transfer where the real robot experiences
    communication / actuator latency.

    Args:
        env: The environment to wrap.
        delay_seconds: Desired delay in seconds.
        step_dt: Simulation step duration in seconds (used to convert seconds -> steps).
    """

    def __init__(self, env: gym.Env, delay_seconds: float, step_dt: float):
        super().__init__(env)
        self.delay_seconds = delay_seconds
        self.step_dt = step_dt
        self.delay_steps: int = max(1, math.ceil(delay_seconds / step_dt))

        # Ring-buffer that holds the last ``delay_steps`` actions.
        self._action_buffer: deque[torch.Tensor] = deque(maxlen=self.delay_steps + 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the buffer with zeros so the first ``delay_steps`` steps use a
        # zero action (i.e. hold default pose).
        zero_action = torch.zeros_like(
            torch.as_tensor(self.action_space.sample())
        )
        self._action_buffer.clear()
        for _ in range(self.delay_steps + 1):
            self._action_buffer.append(zero_action)
        return obs, info

    def step(self, action):
        # Store the latest commanded action …
        self._action_buffer.append(
            action if isinstance(action, torch.Tensor) else torch.as_tensor(action)
        )
        # … but actually apply the delayed (older) action.
        delayed_action = self._action_buffer[0]
        return self.env.step(delayed_action)

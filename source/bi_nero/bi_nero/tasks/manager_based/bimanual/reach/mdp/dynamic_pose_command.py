"""Dynamic sweeping pose command for bimanual reach tasks.

状态机行为（per-env）：
  1. [moving]  : 目标沿 Y 轴匀速移动，机械臂跟踪
  2. [waiting] : 目标切换到固定"休息位置"，机械臂达到后保持
  3. 计时结束  : 新动态目标在 start_pos_y 随机 X 处出现，循环

等待期使用固定 rest_pos 而非停在终点，确保 reward 稳定、观测 in-distribution。
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DynamicSweepPoseCommand(UniformPoseCommand):
    """动态扫描姿态命令生成器。

    • 运动期：目标从 start_pos_y 沿负 Y 匀速移动到 end_pos_y
    • 等待期：目标切换到固定 rest_pos，机械臂保持在该位置
    • 等待结束：目标以新随机 X 从 start_pos_y 再次出现
    """

    cfg: DynamicSweepPoseCommandCfg

    def __init__(self, cfg: DynamicSweepPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        num_envs = env.num_envs
        device = env.device

        # 每个环境的运动状态
        self.is_moving = torch.ones(num_envs, dtype=torch.bool, device=device)
        # 等待倒计时（秒）
        self.wait_timer = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 内部辅助：移动目标到固定休息位置（等待期使用）
    # ------------------------------------------------------------------
    def _set_rest_target(self, env_ids: torch.Tensor):
        """将指定环境的目标切换到固定休息位置。"""
        if len(env_ids) == 0:
            return
        self.pose_command_b[env_ids, 0] = self.cfg.rest_pos_x
        self.pose_command_b[env_ids, 1] = self.cfg.rest_pos_y
        self.pose_command_b[env_ids, 2] = self.cfg.rest_pos_z

    # ------------------------------------------------------------------
    # 内部辅助：将目标初始化到动态运动起点（随机 X）
    # ------------------------------------------------------------------
    def _reset_target(self, env_ids: torch.Tensor):
        """随机化 X，固定 Z，将 Y 设置为起始位置，进入运动状态。"""
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        device = self._env.device

        # 随机 X ∈ [pos_x_min, pos_x_max]
        x_rand = (
            torch.rand(n, device=device)
            * (self.cfg.pos_x_max - self.cfg.pos_x_min)
            + self.cfg.pos_x_min
        )

        self.pose_command_b[env_ids, 0] = x_rand
        self.pose_command_b[env_ids, 1] = self.cfg.start_pos_y
        self.pose_command_b[env_ids, 2] = self.cfg.fixed_pos_z

        # 标记为运动中
        self.is_moving[env_ids] = True

    # ------------------------------------------------------------------
    # 覆盖父类：每次 episode 重置时调用
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor):
        # 先让父类采样姿态（Roll/Pitch/Yaw）
        super()._resample_command(env_ids)
        # 覆盖位置到初始状态
        self._reset_target(env_ids)
        # 等待计时清零
        self.wait_timer[env_ids] = 0.0

    # ------------------------------------------------------------------
    # 覆盖父类：每个控制步调用
    # ------------------------------------------------------------------
    def _update_command(self):
        """每步更新：移动目标 / 切换休息位置 / 管理等待倒计时。"""
        dt: float = self._env.step_dt
        device = self._env.device

        # ---- 1. 正在运动的环境：更新 Y 坐标 ----
        moving_ids = self.is_moving.nonzero(as_tuple=False).flatten()
        if len(moving_ids) > 0:
            self.pose_command_b[moving_ids, 1] -= self.cfg.velocity * dt

            # 检查是否到达终点
            reached = (
                self.pose_command_b[:, 1] <= self.cfg.end_pos_y
            ) & self.is_moving

            reached_ids = reached.nonzero(as_tuple=False).flatten()
            if len(reached_ids) > 0:
                # ✅ 切换到固定休息位置（reward 稳定，观测正常）
                self._set_rest_target(reached_ids)
                # 更改状态为等待
                self.is_moving[reached_ids] = False
                # 随机等待时间
                wait_time = (
                    torch.rand(len(reached_ids), device=device)
                    * (self.cfg.wait_time_max - self.cfg.wait_time_min)
                    + self.cfg.wait_time_min
                )
                self.wait_timer[reached_ids] = wait_time

        # ---- 2. 等待中的环境：倒计时 ----
        waiting_ids = (~self.is_moving).nonzero(as_tuple=False).flatten()
        if len(waiting_ids) > 0:
            self.wait_timer[waiting_ids] -= dt

            # 计时结束：新目标出现
            done_waiting = (self.wait_timer <= 0.0) & (~self.is_moving)
            done_ids = done_waiting.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                self._reset_target(done_ids)


# ---------------------------------------------------------------------------
# 配置类
# ---------------------------------------------------------------------------

@configclass
class DynamicSweepPoseCommandCfg(UniformPoseCommandCfg):
    """动态扫描 + 固定休息位置 命令配置。

    运动期参数：
      start_pos_y / end_pos_y : Y 轴运动范围
      fixed_pos_z             : 运动期固定 Z 高度
      pos_x_min / pos_x_max   : 每次出现时随机选 X（在此范围内）
      velocity                : Y 轴移动速度（m/s）

    休息期参数（无动态目标时显示的固定目标）：
      rest_pos_x / y / z      : 固定休息目标位置（机器人基坐标系）

    等待时间：
      wait_time_min / max     : 到达终点后等待时间范围（秒）
    """

    class_type: type = DynamicSweepPoseCommand

    # --- 运动参数 ---
    velocity: float = 0.20
    """Y 轴移动速度（m/s），内部取负，目标向负 Y 方向移动。"""

    start_pos_y: float = MISSING
    """动态目标出现时的 Y 初始坐标。Left: 0.0, Right: 0.3"""

    end_pos_y: float = MISSING
    """到达该 Y 坐标后进入等待。Left: -0.3, Right: 0.0"""

    fixed_pos_z: float = 0.40
    """运动期固定 Z 高度（m）。"""

    pos_x_min: float = MISSING
    """动态目标 X 随机范围下限（m）。"""

    pos_x_max: float = MISSING
    """动态目标 X 随机范围上限（m）。"""

    # --- 休息位置（等待期目标）---
    rest_pos_x: float = MISSING
    """等待期固定目标 X 坐标（m）。"""

    rest_pos_y: float = MISSING
    """等待期固定目标 Y 坐标（m）。"""

    rest_pos_z: float = MISSING
    """等待期固定目标 Z 坐标（m）。"""

    # --- 等待时间 ---
    wait_time_min: float = 1.0
    """到达终点后最短等待时间（秒）。"""

    wait_time_max: float = 2.0
    """到达终点后最长等待时间（秒）。"""

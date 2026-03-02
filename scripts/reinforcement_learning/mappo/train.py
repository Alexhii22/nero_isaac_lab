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

"""MAPPO 双臂 Reach 训练脚本。

用法：
    cd <project_root>
    python scripts/reinforcement_learning/mappo/train.py \\
        --task Isaac-Reach-BiNero-MAPPO-v0 \\
        --num_envs 4096 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------ #
# Argparse
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Train bimanual reach with SKRL MAPPO.")
parser.add_argument("--num_envs",       type=int,   default=None,  help="Number of parallel environments.")
parser.add_argument("--task",           type=str,   default="Isaac-Reach-BiNero-MAPPO-v0")
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--max_iterations", type=int,   default=2000,  help="Number of MAPPO update iterations.")
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,   default=200)
parser.add_argument("--video_interval", type=int,   default=2000)
parser.add_argument("--checkpoint",     type=str,   default=None,  help="Path to model checkpoint to resume training.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (Isaac Sim is now running)."""

import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

import skrl
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.multi_agents.torch.mappo import MAPPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

import bi_nero.tasks  # noqa: F401  — 触发 gym.register
from bi_nero.tasks.manager_based.bimanual.reach.marl_wrapper import BimanualMARLWrapper

# ------------------------------------------------------------------ #
# 超参
# ------------------------------------------------------------------ #
ROLLOUTS       = 24
LEARNING_RATE  = 3e-4
HIDDEN_DIMS    = [256, 256, 128]
OBS_LEFT_DIM   = 30
OBS_RIGHT_DIM  = 30
CRITIC_DIM     = 60
ACT_DIM        = 7


# ------------------------------------------------------------------ #
# 模型工厂
# ------------------------------------------------------------------ #

def build_policy(obs_dim: int, act_dim: int, device: str):
    """去中心化 policy：输入局部 obs（30D），输出关节 Δpos（7D）。"""
    return gaussian_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network=[{
            "name":        "net",
            "input":       "STATES",   # SKRL 从 inputs["states"] 取局部 obs
            "layers":      HIDDEN_DIMS,
            "activations": "elu",
        }],
        output="ACTIONS",
    )


def build_value(state_dim: int, device: str):
    """集中化 critic：输入全局状态（60D），输出标量 V(s)。
    network[0]["input"] = "STATES" → SKRL 从 inputs["states"] 取全局状态。
    """
    return deterministic_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),  # dummy
        device=device,
        clip_actions=False,
        network=[{
            "name":        "net",
            "input":       "STATES",
            "layers":      HIDDEN_DIMS,
            "activations": "elu",
        }],
        output="ONE",
    )


# ------------------------------------------------------------------ #
# 主函数
# ------------------------------------------------------------------ #

def main():
    device = args_cli.device if args_cli.device else "cuda:0"

    # ---- 随机种子 ----
    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- 日志目录 ----
    log_root = os.path.abspath(os.path.join("logs", "skrl", "bi_nero_mappo"))
    log_dir  = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)

    # ---- 环境配置 ----
    from bi_nero.tasks.manager_based.bimanual.reach.config.joint_pos_env_cfg_mappo import (
        BiNeroReachMAPPOEnvCfg,
    )
    env_cfg = BiNeroReachMAPPOEnvCfg()
    env_cfg.seed = seed
    env_cfg.sim.device = device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # ---- 创建 Isaac Lab 环境 ----
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # ---- 视频录制（可选）----
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # ---- 套上 multi-agent wrapper ----
    env = BimanualMARLWrapper(env)

    # ---- 模型 ----
    policy_left  = build_policy(OBS_LEFT_DIM,  ACT_DIM, device)
    policy_right = build_policy(OBS_RIGHT_DIM, ACT_DIM, device)
    # 每个 agent 独立的 value 网络（两者都输入全局状态 60D，仍为集中化 critic）
    # 注意：不可共用同一 Python 对象！SKRL 为每个 agent 创建独立 Adam 优化器，
    # 若共享对象会导致两套 Adam 动量估计同时更新同一参数，步长计算错误。
    value_left  = build_value(CRITIC_DIM, device)
    value_right = build_value(CRITIC_DIM, device)

    # ---- Memory（每 agent 独立）----
    memories = {
        "left":  RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device),
        "right": RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device),
    }

    # ---- MAPPO agent ----
    models = {
        "left":  {"policy": policy_left,  "value": value_left},
        "right": {"policy": policy_right, "value": value_right},
    }
    agent_cfg = {
        "rollouts":              ROLLOUTS,
        "learning_epochs":       8,
        "mini_batches":          4,
        "discount_factor":       0.99,
        "lambda":                0.95,
        "grad_norm_clip":        1.0,
        "ratio_clip":            0.2,
        "value_clip":            0.2,
        "clip_predicted_values": True,
        "value_loss_scale":      1.0,
        # 局部 obs 归一化（policy 输入 30D）
        "state_preprocessor":              RunningStandardScaler,
        "state_preprocessor_kwargs":       {
            "size": env.observation_spaces["left"],
            "device": device,
        },
        # 集中化 critic 全局状态归一化（60D）
        "shared_state_preprocessor":       RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {
            "size": env.shared_observation_spaces["left"],
            "device": device,
        },
        # value 输出归一化
        "value_preprocessor":              RunningStandardScaler,
        "value_preprocessor_kwargs":       {"size": 1, "device": device},
        # 增大熵系数：防止策略过早收敛到次优的局部极值
        "entropy_loss_scale":    0.02,
        # 提高学习率：加快从好轨迹中学习的速度
        "learning_rate":         3e-4,
        "experiment": {
            "directory":           log_root,
            "experiment_name":     os.path.basename(log_dir),
            "write_interval":      "auto",
            "checkpoint_interval": "auto",
        },
    }

    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=agent_cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.shared_observation_spaces,  # 全局状态空间（集中化 critic）
    )

    # ---- Trainer ----
    timesteps = ROLLOUTS * args_cli.max_iterations
    trainer_cfg = {
        "timesteps":               timesteps,
        "headless":                True,
        "close_environment_at_exit": False,
        "environment_info":        "log",
    }
    # ---- Load Checkpoint ----
    if args_cli.checkpoint:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
            agent.load(checkpoint_path)
        else:
            print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
            exit(1)

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

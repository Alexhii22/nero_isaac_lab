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

"""将 MAPPO checkpoint 导出为 Gazebo sim2sim 可用的 policy_left.pt、policy_right.pt 与 preprocessor.npz。

在 Isaac Lab 环境中运行（需先启动 Isaac Sim）：
    cd <nero_isaac_lab_project_root>
    python scripts/reinforcement_learning/mappo/export_for_gazebo.py \\
        --checkpoint logs/skrl/bi_nero_mappo/2026-02-24_12-00-00/checkpoints/agent_48000.pt \\
        --output-dir ./exported_mappo_gazebo

导出产物：
    <output_dir>/policy_left.pt   - 左臂 policy 网络（输入已归一化的 30D，输出 7D mean）
    <output_dir>/policy_right.pt   - 右臂 policy 网络
    <output_dir>/preprocessor.npz  - left_mean(30), left_var(30), right_mean(30), right_var(30)

Gazebo 端用 gazebo_mappo_play.py 加载上述文件，构建 30D 左/右观测、归一化后分别推理，动作解码与单 PPO 一致。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export MAPPO checkpoint for Gazebo sim2sim.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to agent_*.pt. Auto-detected if omitted.")
parser.add_argument("--output-dir", "-o", type=str, default="./exported_mappo_gazebo", help="Output directory.")
parser.add_argument("--task", type=str, default="Isaac-Reach-BiNero-MAPPO-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest after Isaac Sim is running."""

import numpy as np
import torch
import gymnasium as gym
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

import bi_nero.tasks  # noqa: F401
from bi_nero.tasks.manager_based.bimanual.reach.marl_wrapper import BimanualMARLWrapper

# Must match train.py / play.py
OBS_LEFT_DIM = 30
OBS_RIGHT_DIM = 30
CRITIC_DIM = 60
ACT_DIM = 7
HIDDEN_DIMS = [256, 256, 128]
LOG_ROOT = os.path.abspath(os.path.join("logs", "skrl", "bi_nero_mappo"))


def build_policy(obs_dim: int, act_dim: int, device: str):
    return gaussian_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network=[{"name": "net", "input": "STATES", "layers": HIDDEN_DIMS, "activations": "elu"}],
        output="ACTIONS",
    )


def build_value(state_dim: int, device: str):
    return deterministic_model(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        device=device,
        clip_actions=False,
        network=[{"name": "net", "input": "STATES", "layers": HIDDEN_DIMS, "activations": "elu"}],
        output="ONE",
    )


def find_latest_checkpoint(log_root: str) -> str:
    pattern = os.path.join(log_root, "**", "checkpoints", "agent_*.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {log_root}. Run train.py or set --checkpoint.")
    return max(checkpoints, key=os.path.getmtime)


def get_policy_net(policy_model):
    """从 SKRL policy Model 中取出输入 states 输出 mean 的 net（30D -> 7D）。"""
    if hasattr(policy_model, "instances") and "policy" in policy_model.instances:
        inner = policy_model.instances["policy"]
        net = getattr(inner, "net", None) or getattr(inner, "actor", None)
        if net is not None:
            return net
    for attr in ("net", "actor", "policy_net"):
        net = getattr(policy_model, attr, None)
        if isinstance(net, torch.nn.Module):
            return net
    raise AttributeError("Could not find policy net in model. Inspect agent.agents['left'].policy structure.")


def get_preprocessor_stats(preprocessor):
    """从 RunningStandardScaler 取 running_mean 与 running_var（numpy 一维）。"""
    if preprocessor is None:
        return None, None
    state = getattr(preprocessor, "state_dict", lambda: {})()
    if not state:
        mean = getattr(preprocessor, "running_mean", None)
        var = getattr(preprocessor, "running_var", None)
    else:
        mean = state.get("running_mean", state.get("mean"))
        var = state.get("running_var", state.get("var"))
    if mean is not None and hasattr(mean, "cpu"):
        mean = mean.cpu().numpy().ravel()
    if var is not None and hasattr(var, "cpu"):
        var = var.cpu().numpy().ravel()
    return mean, var


def main():
    device = getattr(args_cli, "device", None) or "cuda:0"
    resume_path = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else find_latest_checkpoint(LOG_ROOT)
    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Output dir: {out_dir}")

    from bi_nero.tasks.manager_based.bimanual.reach.config.joint_pos_env_cfg_mappo import BiNeroReachMAPPOEnvCfg

    env_cfg = BiNeroReachMAPPOEnvCfg()
    env_cfg.seed = 42
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = 2  # minimal for building spaces

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = BimanualMARLWrapper(env)

    policy_left = build_policy(OBS_LEFT_DIM, ACT_DIM, device)
    policy_right = build_policy(OBS_RIGHT_DIM, ACT_DIM, device)
    value_left = build_value(CRITIC_DIM, device)
    value_right = build_value(CRITIC_DIM, device)

    models = {
        "left": {"policy": policy_left, "value": value_left},
        "right": {"policy": policy_right, "value": value_right},
    }
    agent_cfg = {
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_spaces["left"], "device": device},
        "shared_state_preprocessor": RunningStandardScaler,
        "shared_state_preprocessor_kwargs": {"size": env.shared_observation_spaces["left"], "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=None,
        cfg=agent_cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.shared_observation_spaces,
    )
    env.close()

    agent.init(trainer_cfg={"timesteps": 0, "headless": True})
    agent.load(resume_path)
    agent.set_running_mode("eval")

    # Export policy nets (normalized 30D -> 7D mean)
    eps = 1e-8
    for uid in ("left", "right"):
        policy_obj = agent.agents[uid].policy
        net = get_policy_net(policy_obj)
        net.eval()
        prep = getattr(agent.agents[uid], "state_preprocessor", None)
        mean, var = get_preprocessor_stats(prep)
        if mean is None or var is None:
            mean = np.zeros(OBS_LEFT_DIM, dtype=np.float32)
            var = np.ones(OBS_LEFT_DIM, dtype=np.float32)
        if mean.size != OBS_LEFT_DIM or var.size != OBS_LEFT_DIM:
            mean = np.broadcast_to(np.asarray(mean).ravel()[:OBS_LEFT_DIM], (OBS_LEFT_DIM,)).astype(np.float32)
            var = np.broadcast_to(np.asarray(var).ravel()[:OBS_LEFT_DIM], (OBS_LEFT_DIM,)).astype(np.float32)

        # Trace: input (1, 30) normalized, output (1, 7)
        with torch.no_grad():
            example = torch.from_numpy((np.zeros(30, dtype=np.float32) - mean) / np.sqrt(var + eps)).unsqueeze(0).to(device)
            traced = torch.jit.trace(net, example)

        out_pt = os.path.join(out_dir, f"policy_{uid}.pt")
        traced.save(out_pt)
        print(f"[INFO] Saved {out_pt}")

    # Save preprocessor stats (both agents)
    left_mean, left_var = get_preprocessor_stats(getattr(agent.agents["left"], "state_preprocessor", None))
    right_mean, right_var = get_preprocessor_stats(getattr(agent.agents["right"], "state_preprocessor", None))
    if left_mean is None:
        left_mean = np.zeros(OBS_LEFT_DIM, dtype=np.float32)
        left_var = np.ones(OBS_LEFT_DIM, dtype=np.float32)
    if right_mean is None:
        right_mean = np.zeros(OBS_RIGHT_DIM, dtype=np.float32)
        right_var = np.ones(OBS_RIGHT_DIM, dtype=np.float32)
    np.savez(
        os.path.join(out_dir, "preprocessor.npz"),
        left_mean=np.asarray(left_mean).ravel()[:OBS_LEFT_DIM].astype(np.float32),
        left_var=np.asarray(left_var).ravel()[:OBS_LEFT_DIM].astype(np.float32),
        right_mean=np.asarray(right_mean).ravel()[:OBS_RIGHT_DIM].astype(np.float32),
        right_var=np.asarray(right_var).ravel()[:OBS_RIGHT_DIM].astype(np.float32),
    )
    print(f"[INFO] Saved {os.path.join(out_dir, 'preprocessor.npz')}")
    print("[INFO] Done. Use gazebo_mappo_play.py with --export-dir", out_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()

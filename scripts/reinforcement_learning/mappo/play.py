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

"""MAPPO 双臂 Reach 可视化推理脚本。

用法：
    # 自动加载最新 checkpoint，打开 4 个环境
    python scripts/reinforcement_learning/mappo/play.py \\
        --task Isaac-Reach-BiNero-MAPPO-v0 \\
        --num_envs 4

    # 指定 checkpoint 路径
    python scripts/reinforcement_learning/mappo/play.py \\
        --task Isaac-Reach-BiNero-MAPPO-v0 \\
        --num_envs 4 \\
        --checkpoint logs/skrl/bi_nero_mappo/2026-02-24_12-00-00/checkpoints/agent_48000.pt

    # 通过 ZMQ 发布 policy 指令与关节状态（PlotJuggler 可订阅）
    python scripts/reinforcement_learning/mappo/play.py \\
        --task Isaac-Reach-BiNero-MAPPO-v0 \\
        --num_envs 4 \\
        --zmq_addr tcp://*:5556
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------ #
# Argparse
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Play a trained MAPPO bimanual reach agent.")
parser.add_argument("--num_envs",   type=int,  default=4,    help="Number of environments to visualize.")
parser.add_argument("--task",       type=str,  default="Isaac-Reach-BiNero-MAPPO-v0")
parser.add_argument("--seed",       type=int,  default=42)
parser.add_argument("--checkpoint", type=str,  default=None, help="Path to checkpoint (.pt). Auto-detected if omitted.")
parser.add_argument("--video",      action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--real_time",  action="store_true", default=False, help="Slow down to real-time speed.")
parser.add_argument(
    "--zmq_addr",
    type=str,
    default=None,
    help="ZMQ PUB address for streaming policy commands and joint state as JSON (e.g. tcp://*:5556). PlotJuggler can subscribe.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (Isaac Sim is now running)."""

import glob
import os
import time
import json

import gymnasium as gym
import numpy as np
import torch

from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

import bi_nero.tasks  # noqa: F401  — 触发 gym.register
from bi_nero.tasks.manager_based.bimanual.reach.marl_wrapper import BimanualMARLWrapper

# 关节名（与 reach_env_cfg / asset 一致）
LEFT_JOINT_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
]
RIGHT_JOINT_NAMES = [
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
]

# ------------------------------------------------------------------ #
# 超参（需与 train.py 保持一致，否则模型结构不匹配无法加载 checkpoint）
# ------------------------------------------------------------------ #
HIDDEN_DIMS   = [256, 256, 128]
OBS_LEFT_DIM  = 30
OBS_RIGHT_DIM = 30
CRITIC_DIM    = 60
ACT_DIM       = 7
LOG_ROOT      = os.path.abspath(os.path.join("logs", "skrl", "bi_nero_mappo"))


# ------------------------------------------------------------------ #
# 模型工厂（与 train.py 完全一致）
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Checkpoint 自动查找
# ------------------------------------------------------------------ #

def find_latest_checkpoint(log_root: str) -> str:
    """在 log_root 下递归查找最新修改的 agent_*.pt checkpoint。"""
    pattern = os.path.join(log_root, "**", "checkpoints", "agent_*.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        raise FileNotFoundError(
            f"在 {log_root} 下未找到任何 checkpoint。\n"
            "请先运行 train.py，或通过 --checkpoint 手动指定路径。"
        )
    # 按文件修改时间排序，取最新
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


# ------------------------------------------------------------------ #
# ZMQ 发布：policy 指令 + 关节状态 → JSON，供 PlotJuggler 订阅
#
# JSON 每步一条，格式示例：
#   {
#     "t": 1234567890.123,       // 本机时间戳 (s)
#     "timestep": 42,            // 仿真步数
#     "left_action": [0.1,-0.2,...],   // 左臂 policy 输出 7D，[-1,1] 归一化
#     "right_action": [...],             // 右臂 policy 输出 7D
#     "left_joint_pos": [1.0,1.0,...],   // 左臂当前关节角 (rad)，7D
#     "right_joint_pos": [...]           // 右臂当前关节角 (rad)，7D
#   }
# PlotJuggler：Streaming → Add ZMQ Subscriber，URL 填 tcp://localhost:5556
# ------------------------------------------------------------------ #

def _zmq_publisher_impl(zmq_addr: str, env, actions: dict, step_time: float, env_id: int = 0):
    """从 env 取 robot 关节状态，与 actions 一起序列化为 JSON 并通过 ZMQ 发送。"""
    try:
        import zmq
    except ImportError:
        return

    if not hasattr(_zmq_publisher_impl, "_socket"):
        try:
            ctx = zmq.Context()
            _zmq_publisher_impl._socket = ctx.socket(zmq.PUB)
            # 解决 Address already in use 报错
            _zmq_publisher_impl._socket.setsockopt(zmq.LINGER, 0)
            _zmq_publisher_impl._socket.bind(zmq_addr)
            print(f"[ZMQ] Success: Publishing JSON to {zmq_addr}")
            print(f"[ZMQ] Tip: In PlotJuggler, add 'ZMQ Subscriber', set URL to '{zmq_addr.replace('*', 'localhost')}', and leave 'Topic' empty.")
        except Exception as e:
            print(f"[ZMQ] [ERROR] Failed to bind to {zmq_addr}: {e}")
            # 防止重复报错
            _zmq_publisher_impl._socket = None
            return

    sock = _zmq_publisher_impl._socket
    if sock is None:
        return

    try:
        robot = env.unwrapped.scene["robot"]
        # 记录关节 ID 以避免重复查找
        if not hasattr(_zmq_publisher_impl, "_left_ids"):
            _zmq_publisher_impl._left_ids, _ = robot.find_joints(LEFT_JOINT_NAMES, preserve_order=True)
            _zmq_publisher_impl._right_ids, _ = robot.find_joints(RIGHT_JOINT_NAMES, preserve_order=True)
            if torch.is_tensor(_zmq_publisher_impl._left_ids):
                _zmq_publisher_impl._left_ids = _zmq_publisher_impl._left_ids.cpu().numpy()
            if torch.is_tensor(_zmq_publisher_impl._right_ids):
                _zmq_publisher_impl._right_ids = _zmq_publisher_impl._right_ids.cpu().numpy()

        jpos = robot.data.joint_pos[env_id].cpu().numpy()
        left_joint_pos = jpos[_zmq_publisher_impl._left_ids].tolist()
        right_joint_pos = jpos[_zmq_publisher_impl._right_ids].tolist()
    except Exception as e:
        if not hasattr(_zmq_publisher_impl, "_state_warned"):
            print(f"[ZMQ] [WARN] Could not get robot state: {e}")
            _zmq_publisher_impl._state_warned = True
        left_joint_pos = [0.0] * 7
        right_joint_pos = [0.0] * 7

    # 动作：取 env_id 对应的 mean_actions（已转为 tensor），转 list
    def to_list(t):
        if t is None:
            return [0.0] * 7
        try:
            # 统一处理 torch.Tensor 和 np.ndarray
            if torch.is_tensor(t):
                x = t[env_id] if t.dim() > 1 else t
                return x.detach().cpu().numpy().tolist()
            elif isinstance(t, np.ndarray):
                x = t[env_id] if t.ndim > 1 else t
                return x.tolist()
            elif isinstance(t, (list, tuple)):
                return list(t)
            return [float(t)]
        except Exception:
            return [0.0] * 7

    # 计算绝对目标角度 (Target Joint Position) - 简化逻辑
    # 依据 BiNeroReachEnvCfg 中的配置：scale=0.5, use_default_offset=True
    DEFAULT_LEFT = [1.0, 1.0, -1.0, 1.5, 1.0, 0.0, 0.0]
    DEFAULT_RIGHT = [-1.0, 1.0, 1.0, 1.5, -1.0, 0.0, 0.0]
    SCALE = 0.5

    left_target_joint_pos = DEFAULT_LEFT
    right_target_joint_pos = DEFAULT_RIGHT

    try:
        # 获取动作 tensor
        l_raw = actions.get("left")
        r_raw = actions.get("right")
        
        # 调试输出：每 100 步打印一次数据状态
        ts = getattr(_zmq_publisher_impl, "_timestep", 0)
        show_debug = (ts % 100 == 0)

        if l_raw is not None and r_raw is not None:
            # 取当前环境 ID 的动作并转为 list
            def extract_act(raw, default):
                try:
                    if torch.is_tensor(raw):
                        a = raw[env_id] if raw.dim() > 1 else raw
                        return a.detach().cpu().numpy().tolist()
                    elif isinstance(raw, np.ndarray):
                        a = raw[env_id] if raw.ndim > 1 else raw
                        return a.tolist()
                    elif isinstance(raw, (list, tuple)):
                        return list(raw)
                    return default
                except Exception:
                    return default

            l_act = extract_act(l_raw, [0.0]*7)
            r_act = extract_act(r_raw, [0.0]*7)
            
            # 直接计算绝对目标：target = default + action * scale
            left_target_joint_pos = [d + a * SCALE for d, a in zip(DEFAULT_LEFT, l_act)]
            right_target_joint_pos = [d + a * SCALE for d, a in zip(DEFAULT_RIGHT, r_act)]
            
            if show_debug:
                print(f"[ZMQ] [DEBUG] Step {ts}:")
                print(f"      l_act (first 3): {l_act[:3]}")
                print(f"      l_target (first 3): {left_target_joint_pos[:3]}")
        else:
            if show_debug:
                print(f"[ZMQ] [WARN] Actions missing: left={l_raw is None}, right={r_raw is None}")
    except Exception as e:
        if not hasattr(_zmq_publisher_impl, "_error_printed"):
            print(f"[ZMQ] [ERROR] Target calculation failed at step {ts}: {e}")
            import traceback
            traceback.print_exc()
            _zmq_publisher_impl._error_printed = True

    payload = {
        "t": step_time,
        "timestep": getattr(_zmq_publisher_impl, "_timestep", 0),
        "left_action": to_list(actions.get("left")),
        "right_action": to_list(actions.get("right")),
        "left_target_joint_pos": left_target_joint_pos,
        "right_target_joint_pos": right_target_joint_pos,
        "left_joint_pos": left_joint_pos,
        "right_joint_pos": right_joint_pos,
    }
    
    try:
        sock.send_string(json.dumps(payload))
        # 调试：每 100 步打印一次确认
        ts = getattr(_zmq_publisher_impl, "_timestep", 0)
        if ts % 100 == 0:
            print(f"[ZMQ] Sent packet {ts} at {step_time:.3f}")
        _zmq_publisher_impl._timestep = ts + 1
    except Exception as e:
        if not hasattr(_zmq_publisher_impl, "_send_warned"):
            print(f"[ZMQ] [ERROR] Send failed: {e}")
            _zmq_publisher_impl._send_warned = True


def maybe_publish_zmq(zmq_addr: str | None, env, actions: dict):
    if not zmq_addr:
        return
    try:
        import zmq  # noqa: F401
    except ImportError:
        if not getattr(maybe_publish_zmq, "_warned", False):
            print("\n" + "="*50)
            print("[ERROR] ZMQ Data Streaming requested but 'pyzmq' not found!")
            print("Please run: pip install pyzmq")
            print("="*50 + "\n")
            maybe_publish_zmq._warned = True
        return
    
    step_time = time.time()
    _zmq_publisher_impl(zmq_addr, env, actions, step_time, env_id=0)


def main():
    device = args_cli.device if args_cli.device else "cuda:0"

    # ---- Checkpoint 路径 ----
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = find_latest_checkpoint(LOG_ROOT)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # ---- 环境配置 ----
    from bi_nero.tasks.manager_based.bimanual.reach.config.joint_pos_env_cfg_mappo import (
        BiNeroReachMAPPOEnvCfg_PLAY,
    )
    env_cfg = BiNeroReachMAPPOEnvCfg_PLAY()
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs

    # ---- 创建 Isaac Lab 环境 ----
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = BimanualMARLWrapper(env)

    # ---- 获取 step_dt 用于实时限速 ----
    try:
        step_dt = env.unwrapped.step_dt
    except AttributeError:
        step_dt = None

    # ---- 重建模型（结构须与 train.py 一致）----
    policy_left  = build_policy(OBS_LEFT_DIM,  ACT_DIM, device)
    policy_right = build_policy(OBS_RIGHT_DIM, ACT_DIM, device)
    value_left   = build_value(CRITIC_DIM, device)
    value_right  = build_value(CRITIC_DIM, device)

    models = {
        "left":  {"policy": policy_left,  "value": value_left},
        "right": {"policy": policy_right, "value": value_right},
    }

    # ---- 实例化 MAPPO（eval 模式不需要 memory）----
    # 关键：预处理器配置必须与 train.py 完全一致！
    # checkpoint 保存了 RunningStandardScaler 的均值/方差统计量。
    # 若此处不配置预处理器，agent.load() 会跳过预处理器状态，
    # 推理时观测不经归一化 → 策略收到与训练完全不同的输入分布 → 动作错误。
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=None,
        cfg={
            # 局部 obs 归一化（policy 输入 30D）—— 必须与 train.py 匹配
            "state_preprocessor":        RunningStandardScaler,
            "state_preprocessor_kwargs": {
                "size":   env.observation_spaces["left"],
                "device": device,
            },
            # 全局状态归一化（critic 输入 60D）—— 必须与 train.py 匹配
            "shared_state_preprocessor":        RunningStandardScaler,
            "shared_state_preprocessor_kwargs": {
                "size":   env.shared_observation_spaces["left"],
                "device": device,
            },
            # value 输出归一化 —— 推理时不影响动作，但需存在以匹配 checkpoint 结构
            "value_preprocessor":        RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": device},
            "experiment": {
                "write_interval":      0,
                "checkpoint_interval": 0,
            },
        },
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.shared_observation_spaces,
    )

    # ---- 初始化并加载 checkpoint ----
    agent.init(trainer_cfg={"timesteps": 0, "headless": True})
    agent.load(resume_path)
    agent.set_running_mode("eval")
    print("[INFO] Agent loaded and set to eval mode.")

    # ---- 推理循环 ----
    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # act() 返回 (actions_dict, log_prob_dict, outputs_dict)
            actions_raw, _, outputs = agent.act(obs, timestep=0, timesteps=0)

            # 确定性推理：使用 Gaussian policy 的均值动作
            actions = {
                uid: outputs[uid].get("mean_actions", actions_raw[uid])
                for uid in env.possible_agents
            }

            obs, _, terminated, _, _ = env.step(actions)

        # ZMQ：发布 policy 指令与关节状态（PlotJuggler 可订阅）
        maybe_publish_zmq(args_cli.zmq_addr, env, actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        # 实时限速（可选）
        if args_cli.real_time and step_dt is not None:
            elapsed = time.time() - start_time
            if step_dt > elapsed:
                time.sleep(step_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

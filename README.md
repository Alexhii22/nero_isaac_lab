# OpenArm Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository provides simulation and learning environments for the **OpenArm robotic platform**, built on **NVIDIA Isaac Sim** and **Isaac Lab**.
It enables research and development in **reinforcement learning (RL)**, **imitation learning (IL)**, **teleoperation**, and **sim-to-real transfer** for both **unimanual (single-arm)** and **bimanual (dual-arm)** robotic systems.

### What this repo offers
- **Isaac Sim models** 使用OpenArm的软件架构构建的NERO机械臂双臂动态reach任务，软件包使用MAPPO多智能体算法作为基础，mappo超参微调直接在train文件中修改
- **Isaac Lab training environments** for RL tasks (reach, lift a cube, open a drawer).
- **Imitation learning**, **teleoperation interfaces**, and **sim-to-sim / sim-to-real transfer pipelines** are currently under development and will be available soon.


This repository has been tested with:
- **Ubuntu 22.04**
- **Isaac Sim v5.1.0**
- **Isaac Lab v2.3.0**
- **Python 3.11**

---

## Installation Guide


### (使用说明) Local installation

It is assumed that you have created a virtual environment named env_isaaclab using miniconda or anaconda and will be working within that environment.

1. Clone git at your HOME directory
```bash
cd ~
git clone https://github.com/Alexhii22/nero_isaac_lab.git
```

2. Activate your virtual env which contains Isaac Lab package
```bash
conda activate nero
```

3. Install python package with
```bash
cd nero_isaac_lab
python -m pip install -e source/nero
```

4. With this command, you can verify that OpenArm package has been properly installed and check all the environments where it can be executed.
```bash
python ./scripts/tools/list_envs.py
```

## Reinforcement Learning (RL)
Skrl环境下的mappo 实机部署可使用gymnasium导入gym文件自带

### Training Model

```bash
python /home/databeyond/nero/nero_isaac_lab/scripts/reinforcement_learning/mappo/train.py --task Isaac-Reach-BiNero-MAPPO-v0 --num_envs 4096  --headless
```

### Replay Trained Model 使用zmq队列发布可视化数据，plotjuggler host同一地址接收

```bash
python scripts/reinforcement_learning/mappo/play.py \
    --task Isaac-Reach-BiNero-MAPPO-v0 \
    --num_envs 4 \
    --zmq_addr tcp://*:5556
```

### Analyze logs

```bash
python -m tensorboard.main --logdir=logs
```

And open the google and go to `http://localhost:6006/`



## License

[Apache License 2.0](LICENSE.txt)

Copyright 2025 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
基于松灵NERO机械臂搭建的双臂仿人REACH任务 
基于PPO的强化学习运动学逆解
基于open_arm isaaclab搭建的rl训练架构

first start:cd openarm_isaac_lab
python -m pip install -e source/bi_nero #已注释关于openarm的构建

tensorboard：
python -m tensorboard.main --logdir=logs

nero训练：
python /home/databeyond/agx_arm/openarm_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Reach-Nero-v0 --headless
  
nero 回放：
python /home/databeyond/agx_arm/openarm_isaac_lab/scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Reach-Nero-Play-v0 \
  --checkpoint /home/databeyond/agx_arm/openarm_isaac_lab/logs/rl_games/nero_reach/2026-02-02_18-44-06/nn/nero_reach.pth \
  --num_envs 1
  
python /home/databeyond/agx_arm/openarm_isaac_lab/scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Reach-Nero-Play-v0 \
  --num_envs 4
  
  双臂训练：
  python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-BiNero-Bi-v0 --headless
  训练加上lipsnet的ppo网络：
  python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-NeroV1-Bi-LipsNet-v0 --headless
 双臂回放 
  python ./scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-BiNero-Bi-v0 --num_envs 8
 双臂继续训练示例：
 python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-BiNero-Bi-v0 --headless --resume --load_run 2026-02-09_11-30-56 --checkpoint model_1999.pt

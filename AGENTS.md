# AMP_mjlab 项目知识库

## 项目概述

基于 **mjlab** (MuJoCo) + **rsl_rl** 框架的 **Unitree G1** 人形机器人 AMP 运动控制项目。

**核心创新**：单一 policy 同时学习 locomotion（走/跑）与 recovery（跌倒恢复），避免传统策略切换导致的行为不连续性。

**上游依赖**：
- [mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) — MuJoCo 仿真环境管理器 (v1.2.0)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — RL 训练框架（本仓库自带修改版）
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) — AMP 算法参考实现
- [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) — 部署集成代码（`MJAmp State`）

---

## 目录结构

```
AMP_mjlab/
├── src/                              # 主要源码包 (wbc_mjlab)
│   ├── tasks/
│   │   ├── amp_loco/                 # AMP 运动控制任务 **[核心]**
│   │   │   ├── amp_env_cfg.py        #   环境配置工厂函数
│   │   │   ├── ampmotion_loader.py   #   运动数据加载器 (env侧, MotionLoader)
│   │   │   ├── config/g1/            #   G1 机器人特定配置
│   │   │   │   ├── env_cfgs.py       #   环境参数 (rough/flat)
│   │   │   │   └── rl_cfg.py         #   RL 参数 (RslRlAmpRunnerCfg)
│   │   │   ├── mdp/                  #   MDP 组件
│   │   │   │   ├── events.py         #   初始化/重置事件 (MotionResetManager)
│   │   │   │   ├── observations.py   #   AMP 观测函数 (body pos/ori/vel)
│   │   │   │   ├── rewards.py        #   奖励函数
│   │   │   │   ├── terminations.py   #   终止条件 (DelayedTerminationManager)
│   │   │   │   ├── command.py        #   命令生成
│   │   │   │   ├── terrain.py        #   地形配置
│   │   │   │   └── metrics.py        #   评估指标
│   │   │   └── rl/                   #   RL 相关
│   │   │       ├── runner.py         #   AMPOnPolicyRunner (ONNX 导出)
│   │   │       └── wrapper.py        #   (空文件, 占位)
│   │   └── velocity/                 # 基础速度跟踪任务 (非AMP)
│   ├── assets/
│   │   ├── robots/unitree_g1/        # G1 机器人模型、执行器参数、碰撞配置
│   │   └── motions/g1/amp/           # 运动数据 (NPZ)
│   │       ├── WalkandRun/           #   行走/跑步数据
│   │       └── Recovery/             #   跌倒恢复数据
│   └── __init__.py                   # SRC_PATH 定义
├── rsl_rl/                          # 修改版 rsl_rl 框架 **[核心]**
│   ├── algorithms/
│   │   ├── ppo.py                    # PPO 算法
│   │   ├── amp_ppo.py                # AMP-PPO 算法 (含判别器训练)
│   │   └── distillation.py           # 策略蒸馏
│   ├── modules/
│   │   ├── actor_critic.py           # Actor-Critic 网络
│   │   ├── discriminator.py          # AMP 判别器网络
│   │   ├── normalizer.py             # EmpiricalNormalization
│   │   └── rnd.py                    # Random Network Distillation
│   ├── runners/
│   │   ├── on_policy_runner.py       # 标准 on-policy runner
│   │   └── amp_on_policy_runner.py   # AMP on-policy runner (含 AMPLoader)
│   ├── storage/
│   │   ├── rollout_storage.py        # Rollout 存储
│   │   └── replay_buffer.py          # 回放缓冲区 (AMP 用)
│   └── utils/
│       ├── motion_loader.py          # AMPLoader (rsl_rl侧)
│       ├── utils.py                  # Normalizer, 工具函数
│       ├── wandb_utils.py            # WandB 日志
│       └── neptune_utils.py          # Neptune 日志
├── scripts/
│   ├── train.py                      # 训练入口
│   ├── play.py                       # 回放 + ONNX 导出入口
│   ├── csv_to_npz.py                 # CSV → NPZ 数据转换
│   ├── list_envs.py                  # 列出已注册任务
│   ├── analyze_base_motion.py        # 基础运动分析
│   └── visualize_terrain.py          # 地形可视化
├── mjlab_patch/                      # mjlab 本地补丁
│   └── mjlab/managers/observation_manager.py  # history_ordering 支持
├── motion_data_csv/                  # 原始 CSV 运动数据
├── setup.py                          # 包安装配置
└── logs/                             # 训练日志输出 (git ignored)
```

---

## 核心架构

### 数据流

```
mjlab Env (MuJoCo)
  │
  ├─ obs: {"actor": tensor, "critic": tensor, "amp": tensor}
  │
  ▼
RslRlVecEnvWrapper
  │
  ▼
AmpOnPolicyRunner (rsl_rl)
  │
  ├─ AMPPPO.act(obs, critic_obs, amp_obs) → actions
  ├─ env.step(actions) → (obs, rewards, dones, infos)
  ├─ Discriminator.predict_amp_reward(amp_obs, next_amp_obs, task_reward) → combined_reward
  └─ AMPPPO.process_env_step() + AMPPPO.update() → policy update
```

### 观测空间结构

| 分组 | 内容 | 说明 |
|------|------|------|
| `actor` | base_ang_vel, projected_gravity, command, joint_pos, joint_vel, actions | 低维观测，含噪声，4帧历史 |
| `critic` | actor内容 + base_lin_vel, body_pos_b, body_ori_b | 特权观测，无噪声，4帧历史 |
| `amp` | body_pos_b, body_ori_b, body_lin_vel_b, body_ang_vel_b | 13个身体部位的局部位姿+速度 |

### AMP 判别器

- **输入**：当前帧 + 下一帧的 body pos/ori/lin_vel/ang_vel 拼接
- **维度**：`obs_dim * 2`，其中 `obs_dim = (3+6+3+3) * 13 = 195`
- **网络**：MLP `[1024, 512, 256]` + Linear(256, 1)
- **损失**：MSE (expert→1, policy→-1) + 梯度惩罚 (λ=10)
- **AMP 奖励**：`amp_reward_coef * clamp(1 - 0.25*(d-1)², min=0)`
- **混合奖励**：`(1-task_reward_lerp) * amp_reward + task_reward_lerp * task_reward`

### 关键机制

#### 1. 延迟终止 (DelayedTermination)
- 40% 的 env 在触发终止后不会立即 reset
- 这些 env 获得 `max_delay_steps=250` 步的恢复窗口
- 恢复窗口内从 Recovery 运动数据采样 reset 状态
- 恢复窗口超时后才允许 reset

#### 2. 运动数据准备
- 使用 `AMPLoader` (rsl_rl侧) 预处理运动数据到局部坐标系
- 使用 `MotionResetManager` (env侧) 管理 reset 时的状态采样
- 支持 WalkandRun + Recovery 两组数据，自动拼接

#### 3. ONNX 导出
- 训练时每个 save_interval 自动导出 `policy.onnx`
- 模型包含 obs_normalizer，C++ 部署无需单独实现归一化
- ONNX Metadata 包含 env/rl 配置信息

#### 4. 观测历史排序
- `history_ordering="time"` 表示按时间维展开历史
- 需要 mjlab patch 支持，否则只能使用 `"term"` 排序

---

## 配置体系

### 环境配置 (ManagerBasedRlEnvCfg)

通过 `make_amp_env_cfg()` 工厂函数创建，然后各类机器人覆写特定参数：

| 参数 | G1 Rough | G1 Flat |
|------|----------|---------|
| 地形 | 随机粗糙地形 (平坦40% + 随机粗糙60%) | 平面 |
| CCD | 128 iter | 50 iter |
| num_envs | 默认1 (CLI指定) | 默认1 |
| episode_length | 20s | 20s |
| terrain_scan | 有 (射线传感器) | 无 |

### RL 配置 (RslRlAmpRunnerCfg)

| 参数 | 值 | 说明 |
|------|-----|------|
| actor hidden_dims | [512, 256, 128] | 激活: ELU |
| critic hidden_dims | [512, 256, 128] | 激活: ELU |
| algorithm | AMPPPO | clip=0.2, entropy=0.005 |
| learning_rate | 1e-3 | adaptive schedule |
| num_steps_per_env | 24 | |
| max_iterations | 100001 | |
| amp_reward_coef | 0.1 | AMP 奖励系数 |
| amp_task_reward_lerp | 0.75 | 混合比例 |
| amp_discr_hidden_dims | [1024, 512, 256] | 判别器网络 |
| min_normalized_std | [0.05] × 29 | 策略噪声下限 |

---

## 常用命令

```bash
# 训练
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096

# 回放
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt

# 数据转换
python scripts/csv_to_npz.py --input motion_data_csv/amp --output src/assets/motions/g1/amp

# 列出任务
python scripts/list_envs.py --keyword AMP
```

---

---

## Git 工作流

### 原子化本地提交规则

每次对话中有较大改动完成时（bugfix、feature、配置变更、文档更新等），AI Agent **必须自动在本地创建一个原子级 commit**。

- 每个 commit 应当逻辑完整、可独立存在（一个改动一个 commit）
- commit message 应当清晰说明做了什么、为什么
- 不强制 push，仅作本地保存

### Push 到远程需要用户确认

**严禁 AI Agent 擅自推送任何 commit 到远程仓库。**

- 所有推送操作（`git push`）必须先向用户列出将要推送的 commit 内容、影响范围
- 在获得用户明确批准后方可执行 push
- 默认推送到 `kennyp/23dof`（用户 fork 的 `23dof` 分支）
- `git push --force` 绝对禁止，必须由用户手动执行

### 分支说明

| 分支 | 用途 | 远程 |
|------|------|------|
| `23dof` | 当前开发分支（23-DOF AMP 功能） | `kennyp/23dof` |
| `main` | 上游稳定分支 | `origin/main` |

---

## 编程约定

### Python 风格
- Python 3.11+, 类型注解全开
- `torch`, `numpy`, `dataclasses` 广泛使用
- 配置类使用 `@dataclass(frozen=True)` 不可变模式
- MDP 函数签名：`(env, **params) -> torch.Tensor`

### MDP 函数规则
- 奖励函数返回 `(num_envs,)` 或 `(num_envs, 1)` 张量
- 观测函数返回 `(num_envs, obs_dim)` 张量
- 事件函数通过 `env` 对象访问 scene/manager

### 命名规则
- 任务 ID：`Unitree-G1-AMP-Rough`, `Unitree-G1-AMP-Flat`
- 日志目录：`logs/rsl_rl/g1_amp_locomotion/<timestamp>/`
- 模型文件：`model_<iteration>.pt` + `policy.onnx`

### 检查点格式
```python
{
  "model_state_dict": ...,        # Actor-Critic 参数
  "optimizer_state_dict": ...,     # 优化器状态
  "discriminator_state_dict": ..., # 判别器参数 (AMP only)
  "amp_normalizer": ...,           # AMP 归一化器 (AMP only)
  "iter": ...,                     # 当前迭代数
  "infos": ...,
  # 可选:
  "rnd_state_dict": ...,           # RND 参数
  "obs_norm_state_dict": ...,      # 观测归一化器
}
```

---

## 常见问题

1. **训练到 20k iter 附近指标突变** — 正常现象，通常是策略突然学会 recovery 行为
2. **需要 `history_ordering` 配置** — 必须打 mjlab_patch，否则去掉该配置
3. **多 GPU 训练** — 使用 `torchrunx`，自动检测 CUDA_VISIBLE_DEVICES
4. **NaN guard** — 可通过 `--enable-nan-guard` 开启

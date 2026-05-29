# 训练流程与机制分析

> 本文档详细解析 AMP_mjlab 训练管线的每一环节，阐明其数学原理、设计目的以及对最终策略的收益。

---

## 一、整体训练管线

```
                    ┌───────────────────────────────────────────┐
                    │              训练主循环                      │
                    │  for it = 0 → max_iterations (100001)      │
                    └───────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    ┌────────────────┐     ┌────────────────────┐     ┌──────────────────┐
    │  Phase A       │     │  Phase B           │     │  Phase C         │
    │  初始化配置     │     │  Rollout 收集经验   │     │  策略更新         │
    │  - 加载 NPZ    │     │  - 24 步与环境交互  │     │  - PPO 更新      │
    │  - 创建网络     │     │  - AMP 奖励混合     │     │  - AMP 判别器更新 │
    │  - 建立存储     │     │  - replay buffer    │     │  - 保存 checkpoint│
    └────────────────┘     └────────────────────┘     └──────────────────┘
```

---

## 二、Phase A：初始化配置（一次性）

### 2.1 运动数据加载

**文件**：`rsl_rl/utils/motion_loader.py` (AMPLoader) + `src/tasks/amp_loco/ampmotion_loader.py` (MotionLoader)

**流程**：
```
WalkandRun/*.npz ──┐
                   ├──→ AMPLoader (rsl_rl侧): 逐帧 preload 到局部坐标系
Recovery/*.npz   ──┘         ↓
                   存储 5 个列表: body_pos_b, body_quat_b, body_ori_b, body_lin_vel_b, body_ang_vel_b
                              ↓
                   MotionResetManager (env侧): 拼接成连续帧, 用于 reset
```

**每帧计算**（AMPLoader 的 preloading 循环）：
```
1. subtract_frame_transforms:  body_pos_b = R(anchor)⁻¹ · (body_pos_w - anchor_pos_w)
                              body_quat_b = R(anchor)⁻¹ · body_quat_w
2. matrix_from_quat:          提取旋转矩阵前两列 → 6D 旋转表示
3. quat_apply_inverse:        把速度转到 body 局部坐标系
```

**存储维度**：每个 body 15 维 = pos(3) + ori(6) + lin_vel(3) + ang_vel(3)，13 bodies = 195 维

**目的**：
- 提前把整个运动数据从世界坐标系转局部坐标系，避免训练时重复计算
- **最终收益**：AMP 判别器只需要比较局部坐标系下的 body 状态，对机器人朝向、位置具 有**平移/旋转不变性**

---

## 三、Phase B：Rollout 收集经验

### 3.1 采样 → 执行循环

```
for step in range(24):  # num_steps_per_env
    actions = alg.act(obs, critic_obs, amp_obs)
    obs, rewards, dones, infos = env.step(actions)
    next_amp_obs = infos["observations"]["amp"]
    
    # 修复 terminal state 的 AMP 观测
    next_amp_obs_with_term = clone(next_amp_obs)
    next_amp_obs_with_term[reset_env_ids] = amp_obs[reset_env_ids]
    
    # AMP 奖励混合
    rewards = discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards)
    
    alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
```

**目的与收益**：
- **24 步的短 rollout**：减少 on-policy 算法的样本过时程度，提高策略更新效率
- **terminal 状态修复**：mjlab 在 reset 后立即计算 obs，导致 reset 后的 AMP 观测污染了 terminal 数据；用 pre-step 的 amp_obs 代替，保证判别器看到真实 terminal 状态 → **防止判别器学到错误的 transition**

---

### 3.2 AMP 奖励混合

**公式**（`discriminator.py: predict_amp_reward`）：

```
d = D( [state, next_state] )            # 判别器输出, 范围无界

r_amp_raw = 1 - 0.25 * (d - 1)²         # 二次型奖励
r_amp = amp_reward_coef * clamp(r_amp_raw, 0)
       = 0.1 * max(0, 1 - 0.25*(d-1)²)

# 与 task_reward 线性插值
r_final = (1 - task_reward_lerp) * r_amp + task_reward_lerp * task_reward
        = 0.25 * r_amp + 0.75 * task_reward
```

**曲线分析**：

| d 值 | 含义 | r_amp_raw | r_amp |
|------|------|-----------|-------|
| d = 1 | 与 expert 一致 | 1.0 | 0.1 |
| d = 0 | 中性 | 0.75 | 0.075 |
| d = -1 | 与 policy 一致（差） | 0.0 | 0.0 |
| d < -1 | 比 policy 还差 | < 0 | 0 (clamped) |

**目的**：
- 判别器输出 `d` 衡量 transition 与 expert 数据的**风格相似度**
- 二次型 `1 - 0.25(d-1)²` 在 d=1（expert 似然最高）时给予最大奖励
- `clamp(0)` 防止负奖励干扰
- 75% task + 25% AMP 的风格混合：**策略主要优化任务性能，AMP 提供风格正则化**

**最终收益**：
- 策略学会在完成任务（速度跟踪）的同时产生接近 expert motion 的自然步态
- AMP 奖励充当**隐式运动先验**，不需要显式编写步态周期奖励

---

### 3.3 域随机化（Domain Randomization）

**配置**（`amp_env_cfg.py` events）：

| 随机化项 | 类型 | 参数范围 | 目的 |
|---------|------|---------|------|
| `push_robot` | 间隔扰动 (1-3s) | 线速度 ±1.0/±0.5/±0.4 m/s, 角速度 ±0.52/±0.52/±0.78 rad/s | 抗扰动鲁棒性 |
| `foot_friction` | 启动时随机 | 0.3 ~ 1.2 | 适应不同地面材质 |
| `encoder_bias` | 启动时随机 | ±0.015 rad | 传感器噪声鲁棒 |
| `base_com` | 启动时随机 | ±0.025/±0.025/±0.03 m | 质量分布不确定鲁棒 |

**目的**：
- Sim-to-Real 迁移的关键技术
- 在仿真中暴露策略到**分布外情况**，迫使学到鲁棒策略

**最终收益**：
- 策略在部署到真实 G1 机器人时，面对传感器偏差、地面变化、外部推力时表现稳定
- `push_robot` 尤其对 recovery 能力至关重要 — 训练中的扰动就是测试时的跌倒

---

### 3.4 观测噪声

**配置**（`amp_env_cfg.py observations`）：

| 观测项 | 噪声范围 | 目的 |
|-------|---------|------|
| base_ang_vel | ±0.2 rad/s | IMU 角速度噪声 |
| projected_gravity | ±0.05 | IMU 加速度噪声 |
| joint_pos | ±0.01 rad | 关节编码器噪声 |
| joint_vel | ±0.5 rad/s | 关节速度观测噪声 |
| command | 无噪声 | 命令来自控制器 |

**目的**：
- 策略必须学会在噪声观测下做决策，而不是过拟合到精确值
- **最终收益**：部署时真实传感器噪声不会导致性能退化

---

## 四、Phase B.5：延迟终止（Delayed Termination）

### 4.1 机制

```
reset_triggered ──→ is_delay_env? (40%)
                        │
                    ┌───┴───┐
                    │ YES   │ NO
                    ▼       ▼
            计数器++     立即 reset
               │
        ┌──────┴──────┐
        │ < 250 steps │ = 250 steps
        ▼              ▼
    抑制 reset     允许 reset
    从 Recovery    计数器归零
    数据采样
```

**代码逻辑**（`terminations.py: DelayedTerminationManager.compute()`）：
```python
delay_and_done = delay_mask & dones
self._delay_counters[delay_and_done] += 1

not_ready = delay_and_done & (counter < max_delay_steps)
self._terminated_buf[not_ready] = False      # 抑制 reset

ready = delay_and_done & (counter >= max_delay_steps)
self._delay_counters[ready] = 0              # 允许 reset

# 如果 delay env 自己恢复了（没有 done），计数器归零
self._delay_counters[delay_mask & ~dones] = 0
```

### 4.2 Reset 状态采样

**正常 env**：从 WalkandRun 数据中随机采帧
**Delay env**：优先从 Recovery 数据中采帧（fallback 到 WalkandRun）

**目的**：
- 传统做法：跌倒立即 reset → 策略从未学会从跌倒中恢复
- 40% 的 env 获得 250 步的"恢复窗口"：如果策略自己站起来了，恢复成功；否则超时后 reset 并重新从 Recovery 数据中采样
- **最终收益**：单一 policy 自然学会 recovery 行为，不需要单独训练一个 recovery 策略并切换

---

## 五、Phase C：策略更新

### 5.1 AMP 判别器训练

**输入数据流**：
```
AMPLoader (NPZ expert) ──→ expert_generator ──→ (expert_state, expert_next_state)
                                                      │
amp_storage (replay buffer) ──→ policy_generator ──→ (policy_state, policy_next_state)
                                                      │
                                                      ▼
                                          normalize_by_amp_normalizer
                                                      │
                                                      ▼
                                           Discriminator: D([s, s'])
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                         d_expert         d_policy
                                        target=1          target=-1
```

**损失函数**：
```
L_expert = MSE(d_expert, 1)              # expert → 判为 1
L_policy = MSE(d_policy, -1)             # policy → 判为 -1
L_amp = 0.5 * (L_expert + L_policy)

# 梯度惩罚（WGAN-GP 风格）
L_grad_pen = λ * ||∇D(ε·expert + (1-ε)·policy)||²    # λ = 10

# 总判别器损失
L_disc = L_amp + L_grad_pen
```

**优化器分组**（`amp_ppo.py: __init__`）：
```python
params = [
    {"params": policy.parameters(),                     "name": "policy"},
    {"params": discriminator.trunk.parameters(),        "weight_decay": 10e-4, "name": "amp_trunk"},
    {"params": discriminator.amp_linear.parameters(),   "weight_decay": 10e-2, "name": "amp_head"},
]
```

| 分组 | 权重衰减 | 目的 |
|------|---------|------|
| Policy | 无 | 不限制策略表达能力 |
| Trunk | 0.001 | 温和正则化特征提取器 |
| Head | 0.01 | 强正则化输出层，防止判别器过拟合到 expert 数据 |

### 5.2 梯度惩罚分析

**公式**（`discriminator.py: compute_grad_pen`）：
```python
expert_data = cat([expert_state, expert_next_state])
expert_data.requires_grad = True

grad = autograd.grad(outputs=D(expert_data), inputs=expert_data, ...)[0]
grad_pen = λ * mean(||grad||² - 0)²    # 强制梯度趋于 0
```

**目的**：
- 标准 GAN 的梯度惩罚通常强制 L2 范数接近 1（WGAN-GP）
- 这里强制接近 0：这是一种**简化的 1-Lipschitz 约束**
- **最终收益**：防止判别器在 expert 数据附近产生剧烈梯度，稳定训练

### 5.3 AMP Normalizer 更新

```
每步 update() 后:
    amp_normalizer.update(policy_state.cpu().numpy())
    amp_normalizer.update(expert_state.cpu().numpy())
```

这是 `Normalizer`（不是 `EmpiricalNormalization`）— 跟踪 running mean/std 并用于 normalize 判别器输入。

**目的**：
- 判别器输入是 195 维的 body 位姿/速度，各维度量纲不同（位置 0.1m vs 角速度 10 rad/s）
- 归一化后各维度都在相似范围，判别器不会偏向大数值维度
- **最终收益**：更快更稳定的判别器收敛

### 5.4 PPO 策略更新

**更新前**：`compute_returns()` 用 GAE(λ=0.95, γ=0.99) 计算 advantage

**更新循环**（5 epochs × 4 mini-batches = 20 次更新/迭代）：
```
1. 重新计算 policy.act(obs_batch) → 新 action dist
2. PPO 损失:
   L_clip = max(ratio * A, clip(ratio, 1±ε) * A)   # clipped surrogate
   L_value = MSE(V, returns)                         # 或 clipped MSE
   L_entropy = entropy_bonus * 0.005                 # 探索

3. AMP 损失 + 梯度惩罚（见 5.1）

4. 总损失:
   L_total = L_clip + 1.0 * L_value - 0.005 * L_entropy + 1.0 * L_amp + 1.0 * L_grad_pen
```

**min_normalized_std** (`[0.05] × 29`)：
```
if std < 0.05: clamp to 0.05   # 防止策略过早确定，保留探索能力
```

**自适应学习率**（基于 KL 散度）：
```
if KL > 2 * desired_kl:   lr /= 1.5    # 策略变化太大，降低学习率
if KL < 0.5 * desired_kl: lr *= 1.5    # 策略变化太小，加快学习
```

**目的与收益**：
- PPO 保证稳定更新（clipped objective）
- AMP 损失引导策略产生 expert-like 的 body 运动学
- `task_reward_lerp=0.75` 确保任务完成优先，AMP 只做风格正则化
- **最终收益**：单一 loss 同时优化任务性能和运动风格，训练稳定，无需调参

---

## 六、奖励函数详细分析

### 6.1 任务奖励（需最大化）

| 奖励项 | 公式 | 权重 | 参数 | 作用目标 |
|-------|------|------|------|---------|
| `track_anchor_linear_velocity` | `exp(-||cmd_vel - body_vel||² / σ²)` | 1.0 | σ=1.0 | torso_link 线速度 |
| `track_anchor_angular_velocity` | `exp(-(ω_z_err² + ω_xy²) / σ²)` | 1.0 | σ=3.14 | torso_link 角速度 |
| `track_root_height` | `exp(-(h_desired - h_actual)² / σ²)` | 1.0 | σ=0.3 | pelvis 高度 |
| `body_ang_vel_xy_l2` | `exp(-||ω_xy||² / σ²)` | 0.5 | σ=3.14 | pelvis 的 xy 角速度 |

**分析**：
- 线速度和角速度跟踪使用高斯核：**误差为 0 时奖励最大=1；误差增大时指数衰减**
- `σ` 控制"宽容度"：σ 越大，相同误差的惩罚越小
- `body_ang_vel_xy_l2` 显式惩罚身体 xy 轴角速度 → 保持上身稳定，抑制摇晃

### 6.2 惩罚项

| 惩罚项 | 公式 | 权重 | 目的 |
|-------|------|------|------|
| `is_terminated` | `done ? -200 : 0` | -200 | 强惩罚跌倒 |
| `joint_acc_l2` | `mean(á²)` | -2.5e-7 | 抑制关节急动，产生平滑运动 |
| `joint_pos_limits` | `sum(pos超出limit)/...` | -10.0 | 防止关节超限 |
| `action_rate_l2` | `mean((a_t - a_{t-1})²)` | -0.01 | 抑制动作抖振 |
| `foot_slip` | `sum(||v_xy||² * in_contact)` | -0.25 | 惩罚脚着地时滑动 |
| `self_collisions` | 撞击计数 | -0.1 | 防止自碰撞 |

**分析**：
- `is_terminated = -200` 是高权重惩罚 — 跌倒直接影响生存
- `joint_acc_l2` 权重极小 -2.5e-7，但关节加速度通常很大，乘积后产生合理惩罚
- `foot_slip` 只在有速度指令时激活（`command_threshold=0.1`），站立时不予惩罚

### 6.3 延迟环境奖励缩放

```
delay_env_reward_scaling:  delay_mask ? reward * ratio : reward
delay_env_reward_mask_only: delay_mask ? reward * ratio : 0
```

`_apply_delay_env_reward_scaling` — 用于速度跟踪等主要奖励，延迟环境下**降低**权重
`_apply_delay_env_reward_mask_only` — 用于高度跟踪，延迟环境下**只有**缩放后的奖励（非延迟环境得到 0）

**目的**：
- 延迟环境中机器人可能处于跌倒/恢复的非正常状态
- 此时不应该期望它完美跟踪速度，所以降低 velocity tracking 的奖励权重
- 高度跟踪只在延迟环境中有效（正常环境由其他机制保持高度）

---

## 七、训练监控与检测点（Checkpoint）

### 7.1 保存格式

```python
{
    "model_state_dict": ...,            # Actor-Critic 参数（可加载到 play.py）
    "optimizer_state_dict": ...,         # 优化器状态（支持 resume）
    "discriminator_state_dict": ...,     # 判别器参数
    "amp_normalizer": ...,               # AMP 归一化器 running state
    "iter": current_iteration,           # 当前迭代数
    "infos": ...,
}
```

### 7.2 ONNX 导出（每个 save_interval = 100 次迭代）

```python
class OnnxPolicyWrapper(nn.Module):
    def forward(self, obs):
        if obs_normalizer:
            obs = obs_normalizer(obs)
        return actor_critic.act_inference(obs)    # 不含噪声的 mean action
```

**目的与收益**：
- **training-in-the-loop ONNX 导出**：训练过程中持续生成 ONNX，部署侧可以随时取用
- **含归一化器的 ONNX**：C++ 部署时不需要自己实现归一化，输入原始 obs 即可
- **操作集 18，dynamic batch**：支持各种部署环境

---

## 八、训练曲线特征与故障诊断

### 8.1 约 20k 迭代的指标突变

```
              │
reward        │   ↗ 策略突然学会 recovery
              │  /
              │ /
              │/
              └──────────────────── iter
                     ~20k
```

**原因**：
- 此时策略第一次成功从倒地位姿站起来
- 延迟终止的 250 步窗口中，策略偶然发现了 recovery 行为
- 一旦发现，AMP 奖励（从 Recovery 数据中看到的行为）开始正向反馈

**这是正常现象**，不是训练发散。

### 8.2 关键监控指标

| 指标 | 正常范围 | 含义 |
|------|---------|------|
| `Loss/amp` | ~0.5-1.0 | 判别器区分 policy vs expert 的难度 |
| `Loss/amp_policy_pred` | < 0 | 策略状态被判为 -1 的平均值 |
| `Loss/amp_expert_pred` | ~1.0 | expert 状态被判为 1 的平均值 |
| `Loss/amp_grad_pen` | 小 | 梯度惩罚值 |
| `Policy/mean_noise_std` | ~0.05-0.3 | 策略探索噪声 |
| `Train/mean_reward` | 持续增长 | 总体奖励趋势 |
| `Metrics/mean_delay_steps` | < 250 | 延迟环境平均恢复时间 |

---

## 九、设计收益总结

| 机制 | 解决的核心问题 | 对最终策略的收益 |
|------|-------------|----------------|
| AMP 判别器 | 难以手工设计步态奖励 | 自动从 expert motion 中学习自然步态风格 |
| 延迟终止 + Recovery 采样 | 传统 RL 跌倒立即 reset，学不会恢复 | 单一 policy 统一 locomotion + recovery |
| 域随机化 | 仿真与现实的差距 (Sim-to-Real gap) | 部署到真实 G1 机器人时表现鲁棒 |
| 观测噪声 | 策略过拟合精确仿真值 | 对真实传感器噪声不敏感 |
| 25% AMP + 75% Task 混合 | AMP 可能过度约束任务性能 | 任务性能优先，风格作为正则化 |
| 梯度惩罚 (λ=10) | 判别器训练不稳定 | 更稳定、更快收敛的对抗训练 |
| 自适应学习率 (KL) | PPO 步长敏感 | 自动调节学习率，减少调参 |
| 局部坐标系 AMP 观测 | 对全局位置/朝向的依赖 | 平移旋转不变性，更好的泛化 |
| ONNX 内含归一化器 | 部署时需要额外实现归一化 | 一键部署，C++ 侧零工作 |
| min_normalized_std=0.05 | 策略过早确定导致探索不足 | 保留持续探索能力 |

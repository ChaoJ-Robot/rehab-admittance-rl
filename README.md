# rl_admittance_rehab_ws

基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统。

当前仓库已完成 **Phase 0：仓库初始化**、**Phase 1：MuJoCo 三自由度机器人**、**Phase 2：固定参数导纳控制**、**Phase 3：虚拟患者**、**Phase 4：Gymnasium 训练环境**、**Phase 5：SAC 训练** 和 **Phase 6：安全策略部署层**。页面、Agent 与 ROS2 业务逻辑仍按后续 Phase 顺序实现。

## 环境

- Python 3.10+
- 配置格式：YAML
- 测试：pytest
- 代码检查与格式化：ruff
- 类型检查配置：mypy

## 安装

```bash
cd rl_admittance_rehab_ws
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Phase 0 验证

```bash
python3 -m scripts.check_config
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check .
ruff format --check .
mypy rehab_sim scripts
```

`check_config` 会加载 `configs/` 下的六个 YAML 文件并输出配置摘要。当前配置中的机器人、控制器和安全阈值仍是明确标记的占位值，不能用于控制真实机器人。

## Phase 1：MuJoCo 三自由度机器人

模型与网格位于 `assets/mujoco/`，机器人任务空间为 `[x, y, theta]`，交互力接口为 `[Fx, Fy, Tz]`。三个关节均为绕 MuJoCo +Z 轴的转动关节，零位按 CAD 校正为三连杆共线。`tool_tip` 是末端交互 site；其平面偏置来自 CAD 网格包络估计，真实机器人使用前必须重新标定。

运动学和 MuJoCo 运行时接口分别位于：

- `rehab_sim/robot/kinematics.py`
- `rehab_sim/robot/mujoco_robot.py`

启动 1000 步无界面验证：

```bash
python3 -m scripts.run_phase1_sim --headless --steps 1000
```

启动可视化：

```bash
python3 -m scripts.run_phase1_sim --top
```

施加恒定末端外力进行模型测试：

```bash
python3 -m scripts.run_phase1_sim --headless --steps 1000 --wrench 0.5 -0.2 0.1
```

Phase 1 的位置执行器和外力接口仅用于数字孪生验证，不是导纳控制器，也不允许 RL 直接替代底层伺服输出。

## Phase 2：固定参数导纳控制

控制器位于 `rehab_sim/controllers/admittance_controller.py`，实现对角三自由度模型：

```text
M * ddX + D * dX + K * (X - Xr) = F_effective + F_assist
```

已实现：

- 一阶力信号低通滤波；
- 软死区；
- 速度和加速度限幅；
- 任务空间工作空间裁剪；
- 固定参数的按需辅助项接口；
- 与 Phase 1 MuJoCo 模型连接的阻尼最小二乘 IK 目标映射。

仿真基线参数位于 `configs/admittance.yaml`，明确标记为仿真占位值，不能直接用于真实机器人。

运行三类基线实验：

```bash
python3 -m scripts.run_phase2_baseline --experiment step --duration 3
python3 -m scripts.run_phase2_baseline --experiment sine --duration 3
python3 -m scripts.run_phase2_baseline --experiment reverse --duration 3
```

结果默认写入 `experiments/reports/phase2_baseline/`：

- `*_baseline.csv`：力、期望/实际位姿、速度、加速度和关节目标；
- `*_baseline.svg`：力、速度和位置曲线；
- `*_baseline.json`：样本数、峰值速度、峰值力和漂移摘要。

Phase 2 不包含虚拟患者、RL 策略、随机探索、安全策略部署或 Agent。

## Phase 3：虚拟患者

虚拟患者位于 `rehab_sim/patients/`，根据以下模型生成交互力：

```text
F_h = K_h(X_r-X) + D_h(dX_r-dX)
      + F_bias + F_noise + F_tremor
```

已实现：

- 轻度、中度、重度三类 YAML 配置；
- 主动阻抗力、方向偏置、随机噪声和周期性震颤；
- 反应延迟队列；
- 基于患者主动功率的疲劳累积和休息恢复；
- 独立随机种子、重置和可重复输出；
- 患者主动功率、疲劳和力分项状态记录。

运行患者力生成演示：

```bash
python3 -m scripts.run_phase3_patient_demo --duration 8 --sample-time 0.01
```

结果默认写入 `experiments/reports/phase3_patients/`，每个患者配置包含 CSV 力/状态数据和 SVG 曲线。该脚本是开环患者力生成演示，不是 Gymnasium 环境，也不直接控制机器人。

## Phase 4：Gymnasium 训练环境

环境位于 `rehab_sim/envs/`，提供 `PointReachEnv`、`CircleTrackingEnv` 和 `Figure8TrackingEnv` 三类连续控制任务。动作是低频的导纳参数增量 `[damping_xy, damping_theta, assist_gain, velocity_scale]`，不是关节力矩；观测包含关节状态、末端状态、患者交互力、参考轨迹、跟踪误差、当前导纳参数、任务进度和仿真安全状态。

奖励由进度、归一化跟踪误差、过大交互力、运动突变、辅助能量、参数变化、患者主动功率、任务成功和不安全终止等分项组成，并通过 `info["reward_components"]` 独立记录。环境支持 Gymnasium `reset/step` 接口、有限时域、成功终止和仿真异常终止。

运行 Gymnasium/SB3 检查及短随机策略验证：

```bash
python3 -m scripts.check_phase4_envs
```

Phase 4 的集成测试还覆盖 SB3 `check_env`、三个任务的随机策略有限步运行、无 NaN 和固定种子可重复性：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

该阶段只提供环境和固定控制器基线接口，不包含 SAC 训练脚本或真实机器人安全部署。

## Phase 5：SAC 训练

训练入口为 `scripts/train_sac.py`。SAC 使用 `MultiInputPolicy` 读取 Phase 4 的结构化观测，通过 `VecNormalize` 归一化观测和奖励；策略动作仍然是低频的四维导纳参数增量，不是关节力矩或电机命令。训练脚本支持配置文件中的多随机种子，并支持命令行覆盖训练步数、任务、患者、评估频率和设备。

按配置运行五个随机种子：

```bash
python3 -m scripts.train_sac
```

快速仿真 smoke run：

```bash
python3 -m scripts.train_sac \
  --run-name smoke \
  --total-timesteps 2000 \
  --seeds 0 \
  --learning-starts 256 \
  --device cpu
```

每个 run 目录包含：

- `final_model.zip` 和 `vecnormalize.pkl`；
- 定期 SAC checkpoint、replay buffer 和归一化统计；
- TensorBoard event 文件；
- 自动评估历史、成功率、交互力和参数变化/振荡指标；
- 配置 SHA-256、Git commit 和命令行元数据。

加载已保存模型进行评估：

```bash
python3 -m scripts.evaluate_sac \
  --model experiments/trained_models/phase5_sac/<run>/seed_0000/final_model.zip \
  --vecnormalize experiments/trained_models/phase5_sac/<run>/seed_0000/vecnormalize.pkl
```

## Phase 6：安全策略部署层

独立安全层位于 `rehab_sim/safety/`，不导入 SAC 或 Stable-Baselines3。策略动作经过以下固定顺序处理：动作裁剪、参数变化率限制、参数边界投影、稳定性检查和安全状态检查。策略输出仍只会转化为 `[Dx, Dy, Dtheta, Ka, velocity_scale]`，不会发布关节力矩或电机命令。

安全运行时会在模型加载失败、推理异常/超时、动作或状态出现 NaN、传感器掉线、交互力接近阈值、力矩/速度/加速度超限时切换到配置的保守固定参数。安全配置中的阈值目前全部是仿真占位值，`hardware_validation_required: true`，不能用于真实患者或真机运行。

运行 Phase 6 安全单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/unit/test_safety.py
```

Phase 6 不包含交互页面、Agent 或 ROS2 接口。

## 项目规范

完整设计报告保存在 [PROJECT_SPEC.md](PROJECT_SPEC.md)。每次只推进一个 Phase，并在对应阶段完成测试和文档更新。

# rl_admittance_rehab_ws

基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统。

当前仓库已完成 **Phase 0：仓库初始化**、**Phase 1：MuJoCo 三自由度机器人**、**Phase 2：固定参数导纳控制**、**Phase 3：虚拟患者**、**Phase 4：Gymnasium 训练环境**、**Phase 5：SAC 训练**、**Phase 6：安全策略部署层**、**Phase 7：交互页面**、**Phase 8：交互 Agent**、**Phase 9：ROS2 接口** 和 **Phase 10：系统实验与项目包装**。

## 环境

- Python 3.10+
- 配置格式：YAML
- 测试：pytest
- 代码检查与格式化：ruff
- 类型检查配置：mypy

新手学习入口：[新手学习手册.docx](新手学习手册.docx)，对应 Markdown 源文件为 [docs/新手学习手册.md](docs/新手学习手册.md)。

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

`check_config` 会加载 `configs/` 下的九个 YAML 文件并输出配置摘要。当前配置中的机器人、控制器、安全阈值、Agent 阈值、ROS2 硬件参数和实验参数仍是明确标记的仿真/开发占位值，不能用于控制真实机器人。

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

## Phase 7：交互页面

Phase 7 提供仿真专用的 FastAPI + WebSocket 后端和 React + TypeScript + Vite 前端。后端会话服务维护等待、训练、暂停、完成和停止状态，按 20 Hz 推送任务空间遥测；当前数据源是确定性的 simulation-only provider，不连接真实机器人。Phase 8 在此页面上层接入只读交互 Agent。

启动后端：

```bash
python3 -m uvicorn backend.app.main:app --reload --port 8000
```

另开终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

页面支持：

- 点到点、圆轨迹和八字轨迹任务选择；
- 轻度/中度/重度患者配置选择；
- 固定导纳与 RL 参数调节模式切换；
- 开始、暂停、继续和停止训练；
- 参考/实际轨迹、`Fx/Fy/Tz` 力曲线和导纳参数曲线；
- 任务进度、得分、安全状态、患者主动功率和疲劳估计；
- 训练完成后的误差、峰值力、平滑度、主动做功和辅助做功摘要。

Phase 7 验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/integration/test_phase7_backend.py
cd frontend && npm run build
```

Phase 7 本身不包含 LLM 调用、ROS2 接口或真实机器人控制；交互 Agent 在 Phase 8 独立接入。

## Phase 8：交互 Agent

Agent 位于 `rehab_sim/agent/`，第一版采用规则引擎，不依赖 LLM 或外部 API。它只读取遥测并生成文本反馈，不修改导纳参数、不发布控制命令，也不进入机器人控制闭环。跟踪误差、交互力、速度、患者主动功率和疲劳阈值位于 `configs/agent.yaml`。

已实现：

- 任务开始、跟踪良好、跟踪误差过大、交互力过大、速度过快/过慢、患者不活跃、疲劳和安全停止事件；
- 事件冷却、结构化事件日志和可注入的可选语音播报回调；
- 训练结束总结模板，包含亮点、风险提示和下一步建议；
- Agent 异常隔离，Agent 失败不会中断遥测或安全控制；
- WebSocket 实时反馈、`GET /api/agent/events` 审计接口和前端反馈提示/总结卡片。

验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
cd frontend && npm run build
```

Phase 8 不包含 ROS2 接口；后续如接入 LLM，也只能生成解释/语音文本，不能进入控制链路。

## Phase 9：ROS2 与真实机器人接口

ROS2 工作区位于 `ros2_ws/`，包含：

- `rehab_interfaces`：导纳参数、策略动作、末端状态、力、任务状态、安全状态和实时指标消息，以及任务控制服务；
- `rehab_robot_bridge`：仿真和 ROS2 驱动共用的 `RobotAdapter` 接口。桥接层订阅 `/joint_states`、`/rehab_robot/end_effector_state` 和 `/rehab_robot/wrench`，接收并审计任务空间导纳参数，不提供电机力矩或电流接口；具体硬件驱动仍需经过验证后接入；
- `rehab_policy_node`：固定参数模式和确定性参数策略，所有参数经过既有独立安全监督器、通信看门狗和低速测试限幅；
- `rehab_task_manager`：启动、暂停、停止、复位、患者配置和策略模式服务。

ROS2 参数位于 `configs/ros2.yaml`。默认配置为固定参数、仿真输入、低速测试模式和 `hardware.enabled: false`；真实驱动接入前必须完成硬件限位、力矩/速度阈值、急停、使能和看门狗验证。低速模式只限制任务空间速度缩放，不改变控制器接口为电机命令。

构建和启动：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install -e .
colcon build --base-paths ros2_ws --symlink-install
source install/setup.bash

ros2 run rehab_task_manager task_manager_node --ros-args -p config_dir:=$PWD/configs
ros2 run rehab_robot_bridge robot_bridge_node --ros-args -p config_dir:=$PWD/configs
ros2 run rehab_policy_node policy_node --ros-args -p config_dir:=$PWD/configs
```

Phase 9 验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
source /opt/ros/humble/setup.bash
colcon build --base-paths ros2_ws --symlink-install
```

当前阶段完成的是 ROS2 接口和无人体接触的仿真/台架准备；没有连接具体真实机器人驱动，也没有进行人体接触测试。任何通信超时都会选择安全回退参数，固定参数模式可以独立于策略运行。

## Phase 10：系统实验与项目包装

Phase 10 提供统一的一键对比入口 `scripts/run_phase10_experiments.py`。默认实验矩阵包含五种方法和三类虚拟患者：

- 固定导纳 `fixed_admittance`；
- 规则自适应 `rule_adaptive`；
- 模糊规则 `fuzzy_control`；
- SAC；
- PPO。

所有方法使用同一个 MuJoCo/Gymnasium 任务和同一个四维低频导纳参数动作空间。默认配置、方法阈值、患者矩阵、随机种子和训练步数位于 `configs/phase10.yaml`，SAC/PPO 训练只调整导纳参数，不发布电机命令。

完整实验：

```bash
python3 -m scripts.run_phase10_experiments
```

新环境快速验收：

```bash
python3 -m scripts.run_phase10_experiments --quick \
  --output-dir experiments/reports/phase10_quick
```

可选录制无头 MuJoCo 演示视频：

```bash
python3 -m scripts.run_phase10_experiments --quick --record-video
```

输出目录包含：

- `episode_metrics.csv`：逐 episode 原始指标；
- `summary.csv` / `summary.json`：按方法和患者聚合的均值、标准差、成功率、安全率、误差、峰值力、患者主动功率、辅助能量和参数振荡率；
- `success_rate.png`、`tracking_force_comparison.png`、`parameter_stability.png`：自动生成图表；
- `phase10_report.md`：带配置哈希、Git commit 和限制说明的实验报告；
- `models/`：短周期 SAC/PPO 模型及 VecNormalize 统计；
- `demo_*.mp4`：可选的无头 MuJoCo 轨迹演示视频。

项目报告和简历描述分别位于 [docs/phase10_project_report.md](docs/phase10_project_report.md) 和 [docs/resume_description.md](docs/resume_description.md)。Phase 10 的仿真统计不能替代真实硬件阈值验证、人体实验或临床结论。

## 项目规范

完整设计报告保存在 [PROJECT_SPEC.md](PROJECT_SPEC.md)。每次只推进一个 Phase，并在对应阶段完成测试和文档更新。

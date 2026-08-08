# 简历项目描述

## 基于安全强化学习与交互 Agent 的上肢康复机器人自适应导纳训练系统

### 一、项目背景

平面三自由度上肢康复机器人通常采用**固定参数导纳控制**：阻尼设置过小会导致响应振荡、放大异常扰动，设置过大会让机器人运动迟缓、患者主动参与度下降；而传统人工调参依赖经验，难以连续适配不同患者、不同训练阶段以及同一患者在疲劳前后的运动能力变化。

本项目构建了一套**安全强化学习（Safe RL）驱动的自适应导纳训练系统**：让强化学习作为"大脑"在安全约束下低频在线调整导纳参数（阻尼、辅助增益、速度限制），底层柔顺控制仍由传统导纳控制器执行；同时引入**交互 Agent** 实时感知训练状态，为患者提供任务引导、过程反馈和训练总结，形成"感知—决策—执行—反馈"的完整人机交互闭环。项目覆盖从 MuJoCo 数字孪生、虚拟患者建模、Gymnasium 训练环境、SAC/PPO 策略训练，到安全部署层、实时交互页面与 ROS2 真机接口的完整链路。

### 二、技术栈

| 层次 | 技术 |
|---|---|
| 仿真建模 | MuJoCo（MJCF 数字孪生）、正逆运动学、DLS IK |
| 强化学习 | Gymnasium、Stable-Baselines3（SAC 主算法 / PPO 对比）、`MultiInputPolicy`、`VecNormalize`、TensorBoard |
| 控制算法 | 任务空间三自由度导纳控制、阻抗虚拟患者模型、疲劳模型、模糊/规则自适应基线 |
| 后端 | Python 3.10+、FastAPI、WebSocket、pydantic、YAML 配置体系 |
| 前端 | React + TypeScript + Vite、实时图表 |
| 机器人接口 | ROS2 Humble（自定义接口消息、RobotAdapter 桥接、策略节点、任务管理器） |
| 工程化 | pytest（单元/集成/回归）、ruff、mypy、多随机种子可复现实验、配置哈希与 Git commit 记录 |

### 三、主要工作

1. **强化学习问题定义与训练环境**：将 RL 定位为低频导纳参数调节器（1–5 Hz），动作空间为四维参数增量 `[ΔDxy, ΔDθ, ΔKa, Δλv]`，观测采用 0.5s 滑动窗口统计特征（轨迹误差、交互力、患者主动功率、运动平滑度、疲劳估计等）；设计多分量奖励函数（任务进度、跟踪误差、过大交互力、运动突变、辅助能量、患者主动功率、成功与不安全终止），实现点到点、圆轨迹、"8"字轨迹三类 Gymnasium 连续控制环境，通过 SB3 `check_env` 验证并支持固定种子复现。

2. **虚拟患者建模**：构建参数化阻抗虚拟患者（轻/中/重度配置），模拟主动力、反应延迟、方向偏置、力噪声、周期性震颤及基于主动功率的疲劳累积与休息恢复，为 RL 提供可重复、可随机化的训练与泛化测试环境。

3. **多算法训练与对比实验**：以 SAC（`MultiInputPolicy` + `VecNormalize`，多随机种子、TensorBoard 日志、checkpoint）为主算法训练，与 PPO、固定导纳、规则自适应、模糊控制共五种方法在轻/中/重度虚拟患者上对比，评估成功率、安全终止率、跟踪误差、峰值交互力、患者主动做功占比、辅助能量与参数振荡率，一键流水线自动输出逐 episode 数据、统计表、曲线、Markdown 报告与演示视频。

4. **安全强化学习部署层**：实现独立于策略的安全监督器，对策略动作依次执行裁剪、参数变化率限制、边界投影与稳定性检查；在推理异常/超时、NaN、传感器掉线、交互力超限时自动回退到保守固定参数；RL 只输出任务空间导纳参数、不进入电机控制闭环，训练期探索与部署期确定性推理完全解耦，安全逻辑不依赖任何 RL 模型。

5. **交互 Agent 设计**：实现规则驱动的交互 Agent（不依赖 LLM/外部 API），事件检测器识别跟踪误差过大、交互力过大、速度过快/过慢、患者不活跃、疲劳与安全停止等 8 类事件，带事件冷却与结构化审计日志，通过 WebSocket 向前端推送实时反馈并在训练结束后生成总结（亮点/风险提示/下一步建议），支持可注入的语音播报回调；Agent 与控制链路完全解耦，异常隔离不影响遥测与安全控制。

6. **交互系统与机器人接口**：FastAPI + WebSocket 后端与 React + TypeScript 前端实现 20 Hz 实时遥测（参考/实际轨迹、Fx/Fy/Tz 力曲线、导纳参数、安全状态、疲劳估计与训练摘要），支持任务/患者/控制模式切换与训练控制；ROS2 工作区提供自定义接口消息、仿真/真机统一 `RobotAdapter` 桥接、确定性策略节点与任务管理器，支持固定参数模式独立运行与通信看门狗安全回退。

---

### English Version

**Safe RL-Based Adaptive Admittance Control and Interactive Agent System for a Planar 3-DoF Upper-Limb Rehabilitation Robot**

**Background**: Fixed-parameter admittance control cannot adapt to patients of varying capability or to fatigue-induced performance changes, while manual tuning is experience-dependent. This project builds a safe reinforcement learning framework that adjusts admittance parameters (damping, assist gain, speed limit) at low frequency under hard safety constraints, plus a rule-driven interactive Agent providing real-time coaching feedback and session summaries.

**Tech Stack**: MuJoCo, Gymnasium, Stable-Baselines3 (SAC/PPO), task-space admittance control, virtual impedance patient models with fatigue dynamics, FastAPI + WebSocket, React + TypeScript, ROS2 Humble, pytest/ruff/mypy, YAML-driven reproducible experiments.

**Key Work**:
- Formulated RL as a 1–5 Hz admittance-parameter regulator with 4-D action deltas `[ΔDxy, ΔDθ, ΔKa, Δλv]`, sliding-window state features, and a multi-component reward; implemented three Gymnasium continuous-control tasks (point-reach, circle, figure-8) validated with SB3 `check_env`.
- Built mild/moderate/severe virtual impedance patients simulating reaction delay, directional bias, noise, tremor, and power-based fatigue accumulation/recovery for reproducible training and generalization testing.
- Trained SAC (MultiInputPolicy + VecNormalize, multi-seed, TensorBoard, checkpointing) and PPO against fixed, rule-adaptive, and fuzzy baselines; evaluated success rate, tracking RMSE, peak interaction force, patient active-work ratio, assistive energy, and parameter oscillation via an automated one-command experiment pipeline.
- Implemented a model-free safety layer (action clipping, rate limits, parameter projection, stability checks, conservative fallback on inference failure/NaN/sensor loss/force limits), decoupling training-time exploration from deterministic deployment-time inference; RL never enters the motor control loop.
- Built a rule-based interactive coaching Agent (LLM-free) detecting 8 event types (high tracking error, excessive force, speed anomalies, inactivity, fatigue, safety stop) with cooldown and audit logging, delivering real-time WebSocket feedback and post-session summaries with full isolation from the control loop.
- Integrated a FastAPI/WebSocket backend with a React+TypeScript dashboard streaming 20 Hz telemetry and a ROS2 workspace (custom interfaces, sim/real RobotAdapter bridge, deterministic policy node, task manager) with watchdog-based safe fallback.

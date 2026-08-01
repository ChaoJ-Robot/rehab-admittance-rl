# 基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统
## 工程设计与实施报告（Codex 执行版）

---

## 1. 项目名称

**中文名称：** 基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统

**英文名称：** Safe Reinforcement Learning-Based Adaptive Admittance and Interactive Agent System for a Planar 3-DoF Upper-Limb Rehabilitation Robot

**建议工作空间名称：**

```text
rl_admittance_rehab_ws
```

**建议 Git 仓库名称：**

```text
planar-rehab-safe-rl
```

---

## 2. 项目背景

现有平面三自由度上肢康复机器人通常采用固定参数导纳控制。固定参数能够保证基本柔顺性，但难以同时适配不同使用者、不同训练阶段和同一使用者在疲劳前后的运动能力变化。

当导纳阻尼或力—速度映射增益设置过小时，机器人可能响应过快、产生振荡或放大使用者的异常扰动；设置过大时，机器人运动迟缓，使用者需要施加较大的交互力，主动参与度下降。传统人工调参依赖经验，难以实现连续、个体化的辅助。

本项目拟在现有平面三自由度机器人基础上，构建一个包含以下模块的完整系统：

1. MuJoCo 数字孪生和虚拟患者模型；
2. 固定参数导纳控制基线；
3. 安全强化学习参数调节器；
4. 面向患者的训练交互页面；
5. 用于任务引导、反馈和训练总结的交互 Agent；
6. ROS2 或现有控制系统接口；
7. 仿真训练、硬件推理和实验评估闭环。

强化学习不直接输出电机力矩或轴位置，而是在安全约束下低频调整导纳阻尼、辅助增益和速度限制。底层伺服和导纳控制仍由传统控制算法执行。

---

## 3. 项目目标

### 3.1 总体目标

实现一个可在 MuJoCo 中训练、可迁移至真实平面三自由度机器人、能够根据患者运动表现在线调整柔顺参数和辅助等级的上肢康复训练系统。

### 3.2 功能目标

系统应具备以下功能：

- 建立平面三自由度机器人的 MuJoCo 模型；
- 支持末端平面位置和转角三自由度运动；
- 模拟或接入末端六维力传感器中的 `Fx、Fy、Tz`；
- 实现固定参数导纳控制；
- 建立多种能力等级的虚拟患者模型；
- 使用 SAC 或 PPO 在仿真中训练参数调节策略；
- 在运行过程中根据轨迹误差、交互力、运动平滑性和主动做功调整控制参数；
- 具备参数限幅、参数变化率限制、交互力保护和故障回退；
- 提供康复任务页面、实时参数显示、轨迹显示和训练统计；
- Agent 根据训练状态提供语音或文本反馈；
- 记录完整实验数据并自动生成评估结果。

### 3.3 就业项目目标

项目最终应能够展示以下能力：

- MuJoCo 机器人建模；
- 人机交互与导纳控制；
- 强化学习环境设计；
- SAC/PPO 训练和部署；
- 安全强化学习与参数约束；
- ROS2 系统集成；
- React/FastAPI 实时交互页面；
- 工程实验设计和性能评估；
- 仿真到真机的接口迁移设计。

---

## 4. 项目范围与假设

本报告默认机器人自由度定义为：

```text
q = [x_axis, y_axis, theta_axis]
```

末端任务空间定义为：

```text
X = [x, y, theta]
```

主要交互力信号为：

```text
F_h = [Fx, Fy, Tz]
```

若实际机器人第三自由度不是末端平面转角，可在配置文件中修改自由度语义，但软件架构保持不变。

本项目第一阶段只考虑平面训练，不建立完整人体肌骨模型。患者行为通过参数化虚拟患者模型表示。

---

## 5. 总体系统架构

```text
┌─────────────────────────────────────────────┐
│                交互页面与 Agent              │
│  任务显示、轨迹显示、实时参数、得分、反馈     │
└───────────────────┬─────────────────────────┘
                    │ WebSocket / ROS2
                    ↓
┌─────────────────────────────────────────────┐
│             任务管理与训练状态机              │
│  等待 → 校准 → 训练 → 暂停 → 完成 → 总结      │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          安全强化学习参数调节器               │
│ 状态 → SAC/PPO → 参数增量 → 安全投影          │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                导纳控制器                    │
│  Fx、Fy、Tz → 期望速度/位置                  │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              底层运动控制器                  │
│  位置环、速度环、限位、急停、故障处理         │
└───────────────────┬─────────────────────────┘
                    ↓
        MuJoCo 仿真机器人 / 真实三自由度机器人
```

---

## 6. 分层控制频率

建议采用多频率架构：

| 模块 | 推荐频率 |
|---|---:|
| 底层伺服控制 | 500–1000 Hz |
| 导纳控制 | 250–500 Hz |
| 力信号滤波与安全监控 | 250–500 Hz |
| RL 状态统计窗口 | 10–20 Hz |
| RL 参数更新 | 1–5 Hz |
| 页面数据刷新 | 20–30 Hz |
| Agent 反馈决策 | 0.2–1 Hz 或事件触发 |

强化学习只能低频调整参数，禁止直接进入 1 ms 级底层闭环。

---

## 7. 导纳控制器设计

### 7.1 三自由度导纳模型

定义：

\[
X=[x,y,\theta]^T
\]

\[
F_h=[F_x,F_y,T_z]^T
\]

采用以下导纳模型：

\[
M_d\ddot X_d+D_d\dot X_d+K_d(X_d-X_r)
=
F_h+F_{assist}
\]

其中：

- \(M_d\)：虚拟质量矩阵；
- \(D_d\)：虚拟阻尼矩阵；
- \(K_d\)：虚拟刚度或轨迹引导矩阵；
- \(X_r\)：参考轨迹；
- \(F_{assist}\)：按需辅助力。

第一版建议使用对角矩阵：

\[
M_d=\operatorname{diag}(M_x,M_y,M_\theta)
\]

\[
D_d=\operatorname{diag}(D_x,D_y,D_\theta)
\]

\[
K_d=\operatorname{diag}(K_x,K_y,K_\theta)
\]

### 7.2 第一版参数策略

第一版不建议让 RL 同时调整全部 \(M,D,K\)。优先采用：

- 固定虚拟质量 \(M_d\)；
- RL 调整阻尼 \(D_x,D_y,D_\theta\)；
- RL 调整辅助增益 \(K_a\)；
- RL 调整速度上限缩放系数 \(\lambda_v\)。

原因是虚拟质量在线快速变化容易影响系统稳定性。完成第一版后，再扩展为安全约束下的 \(M,D\) 联合调节。

### 7.3 按需辅助力

定义轨迹误差：

\[
e=X_r-X
\]

按需辅助力可写为：

\[
F_{assist}=K_a e+D_a\dot e
\]

其中 \(K_a\) 由强化学习策略动态调整。

### 7.4 必须实现的控制保护

- 力信号低通滤波；
- 软死区；
- 速度限幅；
- 加速度限幅；
- 参数变化率限制；
- 工作空间边界保护；
- 交互力超限回退；
- 异常状态下切换到固定保守参数；
- RL 输出失效或超时处理。

---

## 8. MuJoCo 数字孪生

### 8.1 仿真模型要求

MuJoCo 模型至少包含：

- 三个主动关节；
- 机器人底座；
- 末端握把；
- 末端交互点 `site`；
- 关节限位；
- 速度和驱动力范围；
- 质量、阻尼和摩擦参数；
- 可选的平面约束；
- 接触力或等效外力输入接口。

### 8.2 仿真环境观测

环境应输出：

```text
joint_position
joint_velocity
end_effector_pose
end_effector_velocity
interaction_force
reference_pose
tracking_error
admittance_parameters
task_progress
safety_status
```

### 8.3 外力模拟

虚拟患者通过 MuJoCo 外力接口向握把施加：

```text
Fx
Fy
Tz
```

外力不应直接随机生成，应由虚拟患者控制模型计算。

### 8.4 仿真验证

在强化学习训练前必须完成：

1. 三轴正向和反向运动测试；
2. 末端位姿与真实机器人坐标定义一致；
3. 外力方向与真实力传感器方向一致；
4. 固定导纳参数下响应稳定；
5. 阶跃力、正弦力和反向力测试；
6. 不同仿真步长下响应一致性测试；
7. 参数变化时无数值发散。

---

## 9. 虚拟患者模型

### 9.1 设计目的

虚拟患者不是完整人体动力学模型，而是一个可参数化的人机交互行为生成器，用于产生不同运动能力、反应延迟、疲劳和异常扰动条件。

### 9.2 基础患者控制模型

虚拟患者希望末端沿参考轨迹运动，其作用力定义为：

\[
F_h =
K_h(X_r-X)
+
D_h(\dot X_r-\dot X)
+
F_{noise}
+
F_{tremor}
\]

其中：

- \(K_h\)：患者主动控制能力；
- \(D_h\)：患者运动阻尼；
- \(F_{noise}\)：随机运动噪声；
- \(F_{tremor}\)：周期性异常扰动。

### 9.3 患者能力参数

每个患者配置至少包含：

```yaml
strength_scale: 0.0-1.0
coordination_scale: 0.0-1.0
reaction_delay_ms: 0-500
directional_bias: [bx, by, btheta]
force_noise_std: [sx, sy, stheta]
tremor_amplitude: [ax, ay, atheta]
tremor_frequency_hz: 0-8
fatigue_rate: 0.0-1.0
recovery_rate: 0.0-1.0
```

### 9.4 三种基础患者类型

#### 轻度受限

- 主动力较强；
- 反应延迟小；
- 轨迹误差较小；
- 需要少量辅助。

#### 中度受限

- 主动力中等；
- 存在方向偏差；
- 反应延迟明显；
- 需要动态辅助。

#### 重度受限

- 主动力较弱；
- 容易停滞；
- 轨迹误差大；
- 需要较强辅助和更保守的阻尼。

### 9.5 疲劳模型

使用时间相关的能力衰减：

\[
K_h(t)=K_{h0}(1-\alpha_f f_t)
\]

\[
f_{t+1}=
\operatorname{clip}
(f_t+c_1P_h-c_2r_{rest},0,1)
\]

其中 \(P_h\) 为患者主动功率，\(f_t\) 为疲劳程度。

---

## 10. 康复训练任务

### 10.1 第一阶段任务

先实现三种基础任务：

1. 点到点到达；
2. 圆形轨迹跟踪；
3. “8”字轨迹跟踪。

### 10.2 第二阶段任务

后续扩展：

- 随机目标点击；
- 目标尺寸自适应；
- 障碍物绕行；
- 速度控制任务；
- 方向性力量训练；
- 旋转自由度协调任务。

### 10.3 任务难度参数

```text
target_distance
target_radius
reference_speed
path_width
assist_level
resistance_level
task_duration
```

初期只允许人工选择任务难度。强化学习参数控制稳定后，再考虑由高层策略调整任务难度。

---

## 11. 强化学习问题定义

### 11.1 强化学习定位

强化学习只负责：

```text
根据最近一段时间的运动表现，调整导纳和辅助参数
```

强化学习不负责：

- 直接输出电机电流；
- 直接输出关节力矩；
- 绕过安全控制器；
- 在真实患者训练时执行随机探索。

### 11.2 状态空间

建议采用 0.5 秒滑动窗口的统计特征，而不是只使用瞬时值。

第一版状态可定义为：

\[
s_t=[
e_x,e_y,e_\theta,
\dot e_x,\dot e_y,\dot e_\theta,
F_x,F_y,T_z,
\dot F_x,\dot F_y,\dot T_z,
v_x,v_y,\omega_z,
P_h,
J_{smooth},
p_{task},
D_x,D_y,D_\theta,
K_a,
a_{t-1}
]
\]

建议实际实现以下特征：

- 当前轨迹误差；
- 窗口平均轨迹误差；
- 最大轨迹误差；
- 末端速度；
- 速度波动；
- 交互力；
- 交互力变化率；
- 患者主动功率；
- 运动平滑度；
- 任务进度；
- 上一周期参数；
- 上一周期动作；
- 疲劳估计值；
- 安全状态。

### 11.3 动作空间

第一版连续动作：

\[
a_t=[
\Delta D_{xy},
\Delta D_\theta,
\Delta K_a,
\Delta \lambda_v
]
\]

其中：

- \(\Delta D_{xy}\)：平移阻尼统一增量；
- \(\Delta D_\theta\)：旋转阻尼增量；
- \(\Delta K_a\)：辅助增益增量；
- \(\Delta \lambda_v\)：速度上限缩放增量。

第二版可扩展为：

\[
a_t=[
\Delta D_x,\Delta D_y,\Delta D_\theta,
\Delta M_x,\Delta M_y,\Delta M_\theta,
\Delta K_a
]
\]

### 11.4 参数更新方式

使用增量方式而不是直接输出绝对参数：

\[
p_{t+1}=
\operatorname{clip}
(p_t+\alpha a_t,p_{min},p_{max})
\]

并增加变化率限制：

\[
|p_{t+1}-p_t|\leq \Delta p_{max}
\]

### 11.5 奖励函数

奖励应同时考虑任务完成、患者参与、安全性和平滑性：

\[
r_t=
w_p r_{progress}
-w_e E_{track}
-w_f P_{force}
-w_j J_{motion}
-w_a A_{robot}
-w_c C_{change}
+w_h P_{human}
+r_{success}
-r_{unsafe}
\]

各项含义：

- \(r_{progress}\)：任务进度；
- \(E_{track}\)：轨迹误差；
- \(P_{force}\)：过大交互力惩罚；
- \(J_{motion}\)：速度或加速度不平滑惩罚；
- \(A_{robot}\)：机器人辅助量惩罚；
- \(C_{change}\)：参数频繁变化惩罚；
- \(P_{human}\)：患者主动功率奖励；
- \(r_{success}\)：完成任务奖励；
- \(r_{unsafe}\)：超力、越界或不稳定惩罚。

参考初始形式：

```text
reward =
+ 2.0 * progress
- 1.5 * normalized_tracking_error
- 1.0 * excessive_force_penalty
- 0.3 * motion_jerk_penalty
- 0.5 * robot_assistance_energy
- 0.2 * parameter_change
+ 0.5 * positive_human_power
+ 20.0 * task_success
- 30.0 * unsafe_termination
```

所有权重必须通过配置文件管理。

### 11.6 终止条件

成功终止：

- 到达目标；
- 完成轨迹；
- 任务时间达到要求且误差合格。

失败终止：

- 交互力超限；
- 速度超限；
- 越出工作空间；
- 数值发散；
- 连续停滞超过阈值；
- RL 参数超出安全范围；
- 仿真异常。

### 11.7 算法选择

主算法：

```text
SAC
```

原因：

- 连续状态和动作；
- 样本利用率高；
- 适合 MuJoCo；
- 容易使用 Stable-Baselines3 实现。

对比算法：

```text
PPO
```

基线方法：

1. 固定参数导纳；
2. 规则式自适应导纳；
3. 模糊控制参数调节；
4. SAC 参数调节；
5. PPO 参数调节。

---

## 12. “在线强化学习”的实现定义

本项目中的在线强化学习应分为两个阶段。

### 12.1 仿真训练阶段

在 MuJoCo 中允许策略探索，完成策略训练和随机化。

### 12.2 真实机器人运行阶段

真实机器人只执行确定性推理：

```text
状态统计
→ 策略推理
→ 安全投影
→ 参数更新
```

禁止在真实患者训练中使用高熵随机动作。

若后续需要在线更新模型，只允许：

- 使用健康受试者数据；
- 采用离线回放更新；
- 在仿真中重新验证；
- 通过人工批准后部署新模型。

因此项目中的“在线”主要指参数在线自适应，而不是在患者身上在线试错训练。

---

## 13. 安全约束设计

### 13.1 参数安全集合

所有参数必须位于配置的安全区间：

```yaml
admittance:
  mass:
    x: [min, max]
    y: [min, max]
    theta: [min, max]
  damping:
    x: [min, max]
    y: [min, max]
    theta: [min, max]
  assist_gain: [min, max]
  velocity_scale: [min, max]
```

### 13.2 RL 动作安全投影

执行顺序：

```text
策略原始动作
→ 动作裁剪
→ 参数变化率限制
→ 参数边界投影
→ 稳定性规则检查
→ 安全监督器批准
→ 导纳控制器
```

### 13.3 故障回退

出现以下情况时，立即切换至保守固定参数：

- 策略推理超时；
- 模型文件缺失；
- 输入状态存在 NaN；
- 参数连续振荡；
- 交互力接近阈值；
- 传感器掉线；
- 通信异常。

### 13.4 硬件保护

必须保留独立于 RL 的：

- 机械限位；
- 软件限位；
- 速度限制；
- 加速度限制；
- 力/力矩限制；
- 急停；
- 使能状态；
- 看门狗；
- 零力校准检查。

---

## 14. 交互页面设计

### 14.1 技术方案

推荐：

```text
前端：React + TypeScript + Vite
后端：FastAPI
实时通信：WebSocket
机器人通信：ROS2 或 TCP/ADS 适配器
```

### 14.2 页面组成

#### 训练主页面

显示：

- 当前训练任务；
- 参考轨迹；
- 实际运动轨迹；
- 当前目标点；
- 末端位置；
- 训练时间；
- 得分；
- 完成进度；
- Agent 提示。

#### 实时参数页面

显示：

- `Fx、Fy、Tz`；
- `x、y、theta`；
- 末端速度；
- 轨迹误差；
- `D_x、D_y、D_theta`；
- 辅助增益；
- 速度上限；
- 患者主动功率；
- 疲劳估计；
- 当前安全状态；
- 当前 RL 动作。

#### 治疗师设置页面

允许设置：

- 任务类型；
- 训练时间；
- 轨迹速度；
- 目标尺寸；
- 初始导纳参数；
- 参数安全范围；
- 力和速度上限；
- 是否启用 RL；
- 患者配置；
- Agent 反馈等级。

#### 训练报告页面

输出：

- 完成率；
- 平均轨迹误差；
- 峰值交互力；
- 平均交互力；
- 运动平滑度；
- 患者主动做功；
- 机器人辅助做功；
- RL 参数变化曲线；
- 与历史训练对比。

---

## 15. 交互 Agent 设计

### 15.1 Agent 定位

Agent 只负责：

- 任务说明；
- 训练开始提示；
- 运动过程鼓励；
- 速度过快或过慢提示；
- 力过大提醒；
- 休息提示；
- 训练总结。

Agent 不得直接控制机器人，也不得直接修改导纳参数。

### 15.2 第一版实现

第一版使用规则驱动 Agent，不依赖大语言模型。

事件示例：

```text
task_started
tracking_good
tracking_error_high
force_too_high
patient_inactive
fatigue_detected
task_completed
safety_stop
```

反馈示例：

```text
“保持当前节奏，轨迹控制得很好。”
“请稍微放松握把，当前交互力偏大。”
“机器人将适当增加辅助，请继续主动完成动作。”
“检测到连续疲劳趋势，建议暂停休息。”
```

### 15.3 第二版实现

后续可以接入语言模型生成更自然的反馈，但必须使用结构化输入：

```json
{
  "event": "tracking_error_high",
  "task": "circle_tracking",
  "duration": 42.5,
  "tracking_error": 0.031,
  "force_level": "normal",
  "fatigue_level": "medium"
}
```

语言模型输出只用于文本或语音，不进入控制链。

---

## 16. 软件目录结构

```text
rl_admittance_rehab_ws/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── robot.yaml
│   ├── admittance.yaml
│   ├── safety.yaml
│   ├── patient_profiles.yaml
│   ├── tasks.yaml
│   └── rl_sac.yaml
├── assets/
│   └── mujoco/
│       ├── planar_rehab_robot.xml
│       ├── handle.xml
│       └── scene.xml
├── rehab_sim/
│   ├── envs/
│   │   ├── planar_rehab_env.py
│   │   ├── point_reach_env.py
│   │   ├── circle_tracking_env.py
│   │   └── figure8_tracking_env.py
│   ├── robot/
│   │   ├── robot_model.py
│   │   └── kinematics.py
│   ├── controllers/
│   │   ├── admittance_controller.py
│   │   ├── baseline_controller.py
│   │   ├── safety_supervisor.py
│   │   └── parameter_projector.py
│   ├── patients/
│   │   ├── patient_base.py
│   │   ├── impedance_patient.py
│   │   ├── fatigue_model.py
│   │   └── profiles.py
│   ├── tasks/
│   ├── rewards/
│   ├── observations/
│   ├── randomization/
│   └── logging/
├── rl/
│   ├── train_sac.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── export_policy.py
│   ├── callbacks.py
│   └── hyperparameter_search.py
├── deployment/
│   ├── policy_runtime.py
│   ├── state_normalizer.py
│   ├── fallback_manager.py
│   └── model_registry.py
├── ros2_ws/
│   └── src/
│       ├── rehab_robot_bridge/
│       ├── rehab_policy_node/
│       ├── rehab_task_manager/
│       ├── rehab_data_logger/
│       └── rehab_interfaces/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── websocket.py
│   │   ├── api/
│   │   ├── services/
│   │   └── schemas/
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── charts/
│   │   ├── services/
│   │   └── types/
│   └── package.json
├── agent/
│   ├── event_detector.py
│   ├── feedback_rules.py
│   ├── message_templates.py
│   └── speech_adapter.py
├── scripts/
│   ├── run_simulation.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   ├── run_dashboard.py
│   └── run_hardware.py
├── experiments/
│   ├── baselines/
│   ├── trained_models/
│   ├── evaluation_results/
│   └── reports/
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

---

## 17. ROS2 接口建议

### 17.1 订阅话题

```text
/joint_states
/rehab_robot/end_effector_state
/rehab_robot/wrench
/rehab/task/reference
/rehab/system/state
```

### 17.2 发布话题

```text
/rehab/admittance/parameters
/rehab/policy/action
/rehab/task/status
/rehab/safety/status
/rehab/metrics/realtime
```

### 17.3 服务

```text
/rehab/start_task
/rehab/pause_task
/rehab/stop_task
/rehab/reset_task
/rehab/enable_rl
/rehab/load_policy
/rehab/set_patient_profile
```

### 17.4 消息定义

建议自定义：

```text
AdmittanceParameters.msg
RehabTaskState.msg
SafetyState.msg
RealtimeMetrics.msg
PolicyAction.msg
PatientProfile.msg
```

---

## 18. 数据记录

每次训练必须记录：

```text
timestamp
task_id
patient_profile
reference_pose
actual_pose
pose_error
end_effector_velocity
interaction_force
human_power
robot_assistance
admittance_parameters
rl_observation
rl_action_raw
rl_action_safe
reward_components
safety_state
agent_event
task_result
```

推荐保存格式：

- 原始高频数据：HDF5 或 Parquet；
- 实验摘要：CSV；
- 配置快照：YAML；
- 模型信息：JSON；
- 可视化图表：PNG；
- 最终报告：Markdown 或 HTML。

---

## 19. 实验设计

### 19.1 对比方法

必须比较：

1. 固定参数导纳；
2. 规则式自适应导纳；
3. 模糊自适应导纳；
4. SAC 自适应导纳；
5. PPO 自适应导纳。

### 19.2 虚拟患者条件

至少测试：

- 轻度；
- 中度；
- 重度；
- 疲劳增加；
- 方向偏差；
- 反应延迟；
- 力噪声；
- 震颤干扰。

### 19.3 核心指标

\[
\text{轨迹 RMSE}
\]

\[
\text{任务完成时间}
\]

\[
\text{任务成功率}
\]

\[
\text{峰值交互力}
\]

\[
\text{平均交互力}
\]

\[
\text{速度平滑度}
\]

\[
\text{患者主动做功占比}
\]

\[
\text{机器人辅助做功}
\]

\[
\text{参数变化总量}
\]

\[
\text{安全触发次数}
\]

### 19.4 泛化测试

训练时随机化：

- 患者能力；
- 反应延迟；
- 力噪声；
- 机器人摩擦；
- 控制延迟；
- 力传感器零漂；
- 轨迹速度；
- 初始位置。

测试时加入训练范围外参数，评估策略泛化能力。

### 19.5 统计要求

每种方法、每种患者条件：

```text
至少 5 个随机种子
每个随机种子至少 50 个 episode
报告均值 ± 标准差
```

---

## 20. 分阶段实施计划

## Phase 0：仓库初始化

### 任务

- 建立目录结构；
- 配置 Python 项目；
- 配置格式化、类型检查和单元测试；
- 建立 YAML 配置系统；
- 编写日志模块。

### 验收标准

- `pytest` 可运行；
- `ruff` 或 `flake8` 无错误；
- 配置文件可加载；
- 示例脚本可启动。

---

## Phase 1：MuJoCo 三自由度机器人

### 任务

- 建立机器人 MJCF；
- 配置关节、质量、阻尼和限位；
- 建立末端 `site`；
- 实现运动学接口；
- 实现外力注入；
- 实现可视化。

### 验收标准

- 三个关节可独立控制；
- 末端位姿输出正确；
- 外力方向正确；
- 关节和工作空间限位有效；
- 1000 步仿真无数值错误。

---

## Phase 2：固定参数导纳控制

### 任务

- 实现三自由度导纳方程；
- 实现滤波、死区、限速和限加速度；
- 实现阶跃力测试；
- 实现正弦力跟随；
- 建立基线实验脚本。

### 验收标准

- 静止无明显漂移；
- 阶跃力响应无发散；
- 反向力切换稳定；
- 参数修改后响应方向符合预期；
- 输出力、速度和位置曲线。

---

## Phase 3：虚拟患者

### 任务

- 实现基础阻抗患者；
- 实现轻度、中度、重度配置；
- 实现延迟、噪声、疲劳和震颤；
- 实现患者主动功率计算。

### 验收标准

- 三类患者表现明显不同；
- 相同配置可重复；
- 不同随机种子产生合理差异；
- 疲劳随训练时间增加；
- 输出患者状态曲线。

---

## Phase 4：Gymnasium 环境

### 任务

- 封装 `reset()`、`step()`；
- 定义 observation 和 action space；
- 实现奖励分项；
- 实现成功和失败终止；
- 通过 Stable-Baselines3 `check_env`。

### 验收标准

- 环境接口符合 Gymnasium；
- 不出现 NaN；
- 随机策略可连续运行；
- 每个奖励分项可单独记录；
- 固定控制器可作为环境基线运行。

---

## Phase 5：SAC 训练

### 任务

- 实现 SAC 训练脚本；
- 状态归一化；
- TensorBoard 日志；
- checkpoint；
- 自动评估；
- 多随机种子训练。

### 验收标准

- 策略奖励明显高于随机策略；
- 成功率持续提升；
- 参数不频繁振荡；
- 交互力不超过安全阈值；
- 模型可保存和加载。

---

## Phase 6：安全策略部署

### 任务

- 实现参数安全投影；
- 实现动作变化率限制；
- 实现策略推理超时检测；
- 实现保守参数回退；
- 实现安全单元测试。

### 验收标准

- 任意策略动作不能突破参数边界；
- NaN 输入触发回退；
- 模型加载失败触发回退；
- 力超限触发回退；
- 安全逻辑独立于 RL 模型运行。

---

## Phase 7：交互页面

### 任务

- FastAPI 后端；
- WebSocket 实时数据；
- React 训练页面；
- 参数曲线；
- 轨迹曲线；
- 训练控制按钮；
- 报告页面。

### 验收标准

- 页面实时刷新不少于 20 Hz；
- 可启动、暂停和停止任务；
- 可切换 RL 与固定参数模式；
- 可显示实时轨迹和力曲线；
- 训练结束生成摘要。

---

## Phase 8：交互 Agent

### 任务

- 建立事件检测；
- 建立规则反馈；
- 支持文本提示；
- 可选语音播报；
- 建立训练总结模板。

### 验收标准

- 跟踪误差过大时给出反馈；
- 交互力过大时给出提醒；
- 疲劳时给出休息建议；
- Agent 失效不影响机器人控制；
- 所有反馈事件写入日志。

---

## Phase 9：ROS2 与真实机器人接口

### 任务

- 建立仿真和真机统一接口；
- 订阅真实力和位姿；
- 发布导纳参数；
- 部署确定性策略；
- 实现看门狗；
- 实现真机低速测试模式。

### 验收标准

- RL 节点不直接发布电机命令；
- 参数更新频率符合设计；
- 通信中断触发回退；
- 固定参数模式可独立运行；
- 在不接触人体情况下完成台架测试。

---

## Phase 10：系统实验与项目包装

### 任务

- 完成对比实验；
- 自动生成图表；
- 录制演示视频；
- 编写 README；
- 编写项目报告；
- 编写简历描述。

### 验收标准

- 至少完成五种方法对比；
- 至少完成三类患者测试；
- 输出统计表和曲线；
- 提供一键运行脚本；
- 仓库新环境可复现。

---

## 21. 第一版最小可行产品

第一版不要一次性实现全部功能。最小版本只包含：

```text
MuJoCo 三自由度机器人
+ 固定导纳控制
+ 三类虚拟患者
+ 点到点训练任务
+ SAC 调整 Dxy、Dtheta、Ka
+ 参数安全投影
+ 简单实时页面
```

第一版完成后再加入：

```text
圆形和 8 字任务
+ 疲劳模型
+ PPO 对比
+ Agent
+ ROS2 真机接口
```

---

## 22. Codex 执行要求

Codex 必须遵守以下规则：

1. 不要一次性实现整个项目；
2. 严格按照 Phase 顺序推进；
3. 每个 Phase 单独建立分支；
4. 每个模块先写接口和测试，再写实现；
5. 所有参数放入 YAML，不允许散落硬编码；
6. 所有数组明确单位、坐标系和维度；
7. 所有环境随机数使用可配置种子；
8. 所有安全逻辑必须独立于 RL 策略；
9. 不允许 RL 直接发布电机力矩或电流；
10. 每个 Phase 完成后更新 `README.md` 和 `CHANGELOG.md`；
11. 每个训练脚本必须支持命令行参数；
12. 所有模型文件记录配置哈希和 Git commit；
13. 使用类型注解；
14. 对关键数学函数编写单元测试；
15. 任何不确定的机器人参数先使用配置占位，不自行编造真实硬件参数。

---

## 23. 可直接交给 Codex 的总提示词

```text
你现在需要实现一个“基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统”。

项目工作空间名称为 rl_admittance_rehab_ws。

机器人是平面三自由度机器人，任务空间状态为 [x, y, theta]，主要人机交互信号为 [Fx, Fy, Tz]。强化学习不得直接控制电机或输出关节力矩，只能低频调整导纳阻尼、辅助增益和速度限制。底层控制、参数限幅、交互力保护和故障回退必须独立于强化学习策略。

请阅读项目根目录中的 PROJECT_SPEC.md，严格按照 Phase 0 到 Phase 10 顺序实施。当前只执行我指定的 Phase，不要提前实现后续功能。

工程要求：
1. Python 3.10 或以上；
2. MuJoCo + Gymnasium + Stable-Baselines3；
3. 配置统一使用 YAML；
4. 使用 pytest、类型注解和结构化日志；
5. 每个模块必须有最小单元测试；
6. 所有坐标系、单位和数组维度写在 docstring 中；
7. 强化学习训练和真实策略推理解耦；
8. 所有安全逻辑独立于 RL 模型；
9. 代码应支持后续接入 ROS2；
10. 不要编造实际机器人参数，未知值放入配置文件并标记 TODO。

每次开始一个 Phase 前：
- 先分析当前仓库；
- 列出需要新增和修改的文件；
- 给出实施步骤；
- 再开始修改代码。

每次完成一个 Phase 后：
- 运行测试；
- 给出运行命令；
- 说明已完成内容；
- 说明未完成内容；
- 给出验收结果；
- 不要自动进入下一个 Phase。
```

---

## 24. 推荐从 Codex 开始执行的第一条指令

```text
请先执行 Phase 0：仓库初始化。

要求：
1. 创建 PROJECT_SPEC.md，并保存我提供的完整项目设计报告；
2. 创建报告中规定的基础目录结构，但暂时不要实现 MuJoCo、RL、前端和 ROS2 业务逻辑；
3. 建立 pyproject.toml、requirements.txt、pytest、ruff 和基础日志配置；
4. 建立 configs 目录及 robot.yaml、admittance.yaml、safety.yaml、patient_profiles.yaml、tasks.yaml、rl_sac.yaml 的占位模板；
5. 编写配置加载模块和对应单元测试；
6. 创建 README.md 和 CHANGELOG.md；
7. 给出安装、测试和代码检查命令；
8. 完成后停止，不要执行 Phase 1。
```

---

## 25. 最终交付物

项目最终应交付：

- MuJoCo 三自由度机器人数字孪生；
- 固定和自适应导纳控制器；
- 虚拟患者模型；
- SAC 和 PPO 训练代码；
- 安全策略运行时；
- React 康复训练页面；
- FastAPI 实时数据后端；
- 规则式交互 Agent；
- ROS2 接口；
- 对比实验数据；
- 实验报告；
- 演示视频；
- 完整 README；
- 可写入简历的项目描述。

---

## 26. 预期简历描述

**基于安全强化学习的上肢康复机器人自适应导纳控制系统**

- 面向自主研发的平面三自由度上肢康复机器人，构建 MuJoCo 数字孪生和参数化虚拟患者模型，模拟不同肌力、反应延迟、疲劳和异常扰动条件下的人机交互。
- 设计分层控制架构，以传统导纳控制和独立安全监督器作为底层控制基础，使用 SAC 根据轨迹误差、交互力、运动平滑性和患者主动做功，在线调整阻尼、辅助增益及速度限制。
- 实现参数安全投影、变化率限制、交互力保护和故障回退机制，避免强化学习策略直接进入电机控制闭环。
- 基于 FastAPI、WebSocket、React 和 ROS2 搭建训练交互系统，实现轨迹、交互力、导纳参数、辅助等级和训练指标的实时显示，并通过规则式 Agent 提供训练引导与状态反馈。
- 对固定导纳、规则自适应、模糊控制、SAC 和 PPO 方法进行对比，评估轨迹误差、峰值交互力、患者主动做功占比、辅助能量和安全触发次数。

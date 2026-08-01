# rl_admittance_rehab_ws

基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统。

当前仓库已完成 **Phase 0：仓库初始化** 和 **Phase 1：MuJoCo 三自由度机器人**。当前只实现数字孪生、运动学、末端外力注入、模型限位和可视化；导纳控制、虚拟患者、Gymnasium 环境、强化学习、页面、Agent 与 ROS2 业务逻辑仍按后续 Phase 顺序实现。

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

## 项目规范

完整设计报告保存在 [PROJECT_SPEC.md](PROJECT_SPEC.md)。每次只推进一个 Phase，并在对应阶段完成测试和文档更新。

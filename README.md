# rl_admittance_rehab_ws

基于安全强化学习与交互 Agent 的平面三自由度上肢康复机器人自适应导纳训练系统。

当前仓库处于 **Phase 0：仓库初始化**。本阶段只提供项目结构、YAML 配置模板、配置加载器、日志配置和基础测试；MuJoCo、导纳控制、虚拟患者、Gymnasium 环境、强化学习、页面、Agent 与 ROS2 业务逻辑将在后续 Phase 按顺序实现。

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

## 项目规范

完整设计报告保存在 [PROJECT_SPEC.md](PROJECT_SPEC.md)。每次只推进一个 Phase，并在对应阶段完成测试和文档更新。

# Changelog

## Unreleased

### Phase 0

- 初始化 `rl_admittance_rehab_ws` 项目结构。
- 添加完整项目规范副本 `PROJECT_SPEC.md`。
- 添加 YAML 配置模板、配置加载器、日志配置和基础测试。
- 暂未实现任何 MuJoCo、导纳控制、患者模型、RL、页面、Agent 或 ROS2 业务逻辑。

### Phase 1

- 纳入三自由度 MuJoCo 模型和 CAD mesh 资产。
- 添加平面 3R 正向运动学和解析 Jacobian。
- 添加末端 `tool_tip` site 与 `[Fx, Fy, Tz]` 外力注入接口。
- 添加关节目标、关节限位、工作空间检查和数值稳定性检测。
- 添加 Phase 1 可视化/无界面运行脚本及集成测试。

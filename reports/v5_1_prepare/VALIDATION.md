# 准备验证

状态：代码和无卡验证完成，尚未运行真实 v5 GPU 实验。

- 本机 Conda aidemo：129 tests passed。
- 服务器原 Conda / PyTorch 环境、禁用 CUDA：129 tests passed。
- Ruff 检查、格式检查、`make exact`、源码包和 wheel 构建通过。
- Pro 原包哈希核验通过；CPU 有限枚举重跑最大恒等式误差 2.38698e-15。
- 服务器 11 个模型资产文件哈希通过；旧 C 方向可读且已验证身份，缺失投影明确保留为 null。
- 小型随机模型实际验证采样 → 原始策略 score → LoRA 梯度 → 完整 macro 落盘 → 分析报告。
- 自动化测试验证无 GPU 不启动、阶段超时停止、旧预算不重复计费。

真实 GPU 的吞吐、数值误差与实验收益未测量，由下一窗口执行。入口见 `docs/experiments/LUNA_START.md`。

证据：本目录 `local_tests.log`、`remote_tests.log`、`lint.log`、`exact.log`、`package.log`、`remote_preflight.json`、`source_verification.json`、`deployment_manifest.json`。

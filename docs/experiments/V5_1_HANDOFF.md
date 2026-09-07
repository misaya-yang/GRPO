# 实验交接

2026-09-07：真实小规模实验已完成，当前不自动扩样或训练。定时检查按用户要求暂停。主结果为 INCONCLUSIVE；见[报告](../../reports/trend_pilot/REPORT.md)和[分析](../../reports/trend_pilot/ANALYSIS.md)。服务器连接及路径见[实验机记录](../operations/EXPERIMENT_HOST.md)。

在实验仓库中，用实际 Python 路径设置 `V5_PYTHON`，常用命令：

```bash
bash scripts/start_v5_1.sh status
bash scripts/start_v5_1.sh report
```

`status` 返回当前/最近 run；`report` 汇总已有产物，不启动生成。已完成 run 为 `runs/v5_1_20260907T094302Z`，32 个宏组、256 回答。不要把新的 `start` 当作读取结果。

后续用户要求新周期时，`start --seconds 秒数` 按 v4 修复、numeric、Dev、筛选的顺序执行；若复用已完成前序阶段，用 `start-screen --run 已准备目录 --seconds 秒数`。代码保留累计预算、独立输出目录和超时。现有 `configs/v5_1/screen.json` 是 16 题×8 重复的较大配置，**不是本次 4 题×4 重复的实测配置**；本次冻结配置见 `reports/trend_pilot/result/screen_config.json`。

模型可替换，原生运行即可。主线是同 K/N/参数空间下的 Strat-full 对 IID-all；虚拟组不算独立样本，LoRA 固定点方差不算在线收益。普通运行异常在授权范围内自行处理，不另搭监控系统。原理论来源保持原字节。

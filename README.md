# 相同反馈边际与不同标准化信用

本项目研究一个受控问题：actor 回答保持 IID，且每条回答的反馈边际完全相同，仅改变评分配置在组内共享（S）还是独立（I），组标准化是否会改变平均信用方向与局部学习后果。

当前主线是 [dependent feedback v4](docs/theory/dependent_feedback_v4.md)。固定候选银行保存每条回答在所有预定测试或 criterion 下的 verdict，再精确积分 shared、independent 与 RLOO 优势；同一组 token score 用不同权重累积全参数梯度。独立 C/D 数据、Dev 冻结读出以及同一初态的 `eta`/`eta/2` 更新用于区分代数机制、功能方向和实际后果。

模型只是实验载体，不锁定某个型号或参数规模。优先复用已有、能力和显存适配的权重；任何模型、精度或梯度存储变化都写入回执，并按实际配置限定结论。旧 dependent-rollouts 理论和实现保留为历史对照，不与 S/I 证据混写。

## 已实现内容

- 二值 Poisson-binomial 与离散奖励 `(sum, sumsq)` 精确积分。
- RLOO、standardized advantage、一般奖励不变性刻画与大组极限的 CPU 检查。
- IID actor 采样、逐 token score 核验、完整组所有权和不可覆盖回执。
- 原生 Python 逐测试执行；Docker 只是可选执行后端。
- shared/independent/RLOO/full-rubric 权重、全参数梯度、Dev 读出和局部步长比较。
- prompt 级统计、C/D 独立性检查、预算 ledger 与 GO/STOP/INCONCLUSIVE 门禁。

代码可运行不等于真实模型实验通过。当前证据、未运行项和报告入口见 [index.md](index.md) 与 [docs/STATUS.md](docs/STATUS.md)。

## 本地环境

| 机器 | 已知环境 | 本轮角色 |
|---|---|---|
| 当前 Mac | M4 MAX；macOS 26.6.2；Conda `aidemo` | CPU 测试、精确验证和文档；不是固定或唯一开发机 |
| Mac `aidemo` 快照 | Python 3.13.11、NumPy 2.4.2、PyTorch 2.10.0、Transformers 5.2.0、safetensors 0.7.0 | 2026-09-07 查询；后续以现场回执为准 |
| 家用 PC | 已确认存在；OS、CPU/GPU、内存和 Python 环境未知 | 暂不分配固定角色，不猜测配置 |

本轮 Mac 只运行 CPU 工作。算法不绑定机器或 Conda 路径。当前运行器已在 macOS/Linux 验证；原生测试 worker 与预算锁使用 POSIX 接口，家用 PC 的运行方式待环境确认。通过 `PYTHON`、配置中的 `device`、`model_path` 与 `test_python` 记录实际环境。

## 本地检查

```bash
conda activate aidemo
make test
make lint
make exact
PYTHONPATH=src python -m reward_coupling.cli --help
```

也可以用 `make test PYTHON=/path/to/python` 选择其他已准备环境。完整阶段、输入输出与停止条件见 [RUNBOOK](docs/experiments/RUNBOOK.md)。

## 文档职责

- `agent.md`：稳定的项目核心约束。
- `index.md`：代码、理论、运行记录和证据索引。
- `docs/experiments/PLAN.md`：当前实验优先级、预算边界和判据。
- `docs/experiments/MANIFEST.md`：冻结值与待补回执。

原始方案、Pro 建议和历史 dossier 保存在 `docs/theory/` 与 `docs/archive/`。不要改写这些原始来源，也不要把 NOT_RUN、准备完成或单点 probe 写成 pilot GO。

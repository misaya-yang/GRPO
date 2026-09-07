# What Dependent Rollouts Teach a Policy

ICLR 2027 实验仓库。理论依据是原样归档的
[research dossier](docs/theory/dependent_rollout_research_dossier.md)。
研究主线：采样依赖如何改变成功/失败轨迹间的学习对比；以保留成功计数的
重排、IID-only 预测和小步干预区分计数效应与轨迹身份选择。

## 当前可运行内容

- CPU 精确反例、score-subspace/Fisher 分解、谱安全界验证。
- IID / stratified / lattice 的序列 arithmetic decoder；随机组标签、延迟随机位、
  有理数区间，记录 float64 token CDF 的数值误差。
- RLOO 和二元 standardized score-point 估计器；同池 W/C/X，解析重排平均。
- IID 分层虚拟组积分预测，包含 Poisson-binomial 标准化预测。
- Transformers 完整轨迹采样、full-parameter 加权反传、梯度 safetensors。
- 固定 SGD 方向、整步/半步干预、方向有限差分、独立 IID likelihood-ratio reward。
- 原始组 bootstrap、函数空间 distortion、AST/Fraction verifier、固定数据拆分。
- 两组等预算的六臂在线 RLOO runner 和三种子配置生成器。
- [ICLR 官方模板论文初稿](paper/main.tex)：理论正文、证明与实验协议；LLM 结果尚未填写。

这是一套可执行的**参考实验实现**，吞吐尚未在预训练模型上测量。
CPU 检查和随机小模型 smoke 不构成预训练模型实验结果。最新证据见
[状态记录](docs/STATUS.md) 和 `artifacts/validation/`。

## 本地验证

```bash
uv sync --locked
make test
make lint
make exact
# 安装与服务器相同版本的可选 LLM 依赖，做无下载的小模型测试
uv sync --locked --extra llm
make smoke
```

## 服务器直接运行（已有 Conda）

已只读核对 `connect.westc.seetacloud.com:27741`：
Python 3.12.3，PyTorch 2.8.0+cu128，Transformers 5.15.1，NumPy 2.3.2。
详细记录：[server_environment.json](artifacts/validation/server_environment.json)。

把仓库放到服务器后，不需要安装 uv 或覆盖原有 Conda 包：

```bash
bash scripts/conda_run.sh exact --output runs/cpu-exact.json
/root/miniconda3/bin/python scripts/remote_env_probe.py
```

换 Conda 路径只需设置 `RESEARCH_PYTHON`。换 GPU 调整配置中的
`device` / `dtype` / `attention`，随后重新做 score-point 和吞吐检查。
机制审计默认 float32 + eager，避免把低精度扰动误差混入 mean-direction 结论。

完整操作顺序见 [RUNBOOK](docs/experiments/RUNBOOK.md)。
首个模型配置在 `configs/pilot_*.json`；如果 revision 尚未固定，用
`scripts/pin_model.py` 解析一次精确 SHA（仅查元数据，不下载权重）。
所有输出使用新目录，已有结果不会被覆盖。

## 项目入口

| 路径 | 内容 |
|---|---|
| `docs/theory/` | 原始 dossier 与校验和 |
| `docs/experiments/PLAN.md` | 按 dossier 固定的实验矩阵、判据与证据归属 |
| `docs/experiments/MANIFEST.md` | Pilot 冻结清单模板（dossier §18）；首次 GPU 采集前冻结 |
| `docs/CLAIMS.md` | 声明与证据对应表；状态升级需新增验证回执 |
| `configs/` | Pilot 和训练参数；训练默认 `NOT_RUN` |
| `src/dependent_rollouts/` | 可复用实验实现 |
| `tests/` | 独立枚举、采样边界、反传与干预恢复测试 |
| `data/example/` | 固定结构拆分的算术样例 |
| `paper/` | 正文、附录、参考文献与官方 2027 样式 |
| `runs/` | 忽略的大型运行产物、轨迹、梯度、检查点 |

`dependent-rollouts --help` 列出命令。Conda 等价入口是
`PYTHONPATH=src /root/miniconda3/bin/python -m dependent_rollouts.cli`。

# 项目核心约束

1. 当前研究主线是 v5.1：`docs/theory/v5_1_original/guidance.md` 与 `docs/experiments/V5_1_HANDOFF.md`。以独立分层 Strat-full 对固定 K/N 的 IID-all 为主要比较。v4 只做一次旧银行诊断修复；原始 v3/v4/v5/Pro 文本保持原字节，不在原文上修订。
2. 先冻结问题、estimand、反馈配置、数据划分、预算和判据，再运行实验。不得按结果换任务、换 mask、删组、补样或启动无依据的 GPU 扫描。
3. 模型可替换。优先使用现有权重与适配当前资源的模型；`Qwen2.5-Coder-1.5B-Instruct` 只是 v4 的历史默认示意，不是开跑条件或结论对象。准确记录实际 model/tokenizer revision、精度、参数范围和梯度存储策略。
4. 已获授权的实验机工作可以直接运行原生 Python 和 GPU。Docker、chroot、固定 Conda、固定主机或重复批准都不是前置条件。直接 Hugging Face 不通时，优先复用本地权重，或使用镜像和 AutoDL 支持的下载路径。
5. 机器角色不写死。当前 Mac M4 MAX 本轮只做 CPU 工作；`aidemo` 可用但不是唯一开发环境。家用 PC 存在，规格未知。主机现场信息只写在 `docs/operations/EXPERIMENT_HOST.md`。
6. v5.1 每层内及跨层样本必须相互独立，使用每条回答私有的 fresh RNG；score 对原始边际策略计算。Strat-full 不适用于共享随机数/相关块；cross 仅作有独立块前提的诊断。v4 旧 S/I 合同仍限定回答 IID、外生反馈同边际。全同奖励、截断和解析失败保留在分母中，基础设施失败单独记录。
7. 主机制 loss 是 detached advantage 乘 response-token logp **之和**，包含实际 EOS，不包含 padding，不补 EOS。on-policy、`ddof=0`、根号外 `epsilon=1e-6`；不使用 clipping、KL reward、长度平均或 off-policy epochs。Adam 等在线训练设置另报。
8. CPU 恒等式、随机模型 plumbing、预训练 score/梯度 probe、自然反馈效应、局部后果和在线训练是不同证据层级。均值不变不代表方差、Adam 状态或有限步轨迹不变；不得编造模型结果、吞吐或 GO。
9. v5.1 保留 prompt 与独立 macro-bank 两层不确定性，强对照保持 K 而不是改成 N。v4 C/D 两侧不确定性不能混同条件于旧 C 的诊断。虚拟组、mask、重排或精确积分不是新增独立样本。冻结度量、功能投影及实践收益阈值。
10. 每阶段使用新输出目录，保存命令、版本、配置与数据 hash、完整输出、失败原因和预算 ledger。不要覆盖已有结果。训练必须由带 hash 证据的 pilot GO 解锁。
11. `make test`、`make lint`、`make exact` 是代码检查入口，可用 `PYTHON`/`RUFF` 选择环境。检查通过不自动构成预训练实验或 pilot GO。
12. 修改尽量小，保护无关工作。优先完成已授权、可审计的下一阶段；遇到资源或合同失败先诊断，不用盲扫掩盖问题。

文档职责：`agent.md` 保存核心约束；`AGENTS.md` 只做自动发现入口；`index.md` 维护详细文件、理论、报告和结果索引；`README.md` 只写项目详情及 Mac/家用 PC 环境；服务器信息集中在独立运维记录。

当前状态与运行交接见 `docs/STATUS.md` 和 `docs/experiments/V5_1_HANDOFF.md`。普通异常在用户授权范围内自行处理；版本升级不重置累计预算。

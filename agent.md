# 项目核心约束

1. 研究主线是 `docs/theory/dependent_feedback_v4.md`，并结合 `docs/theory/FEEDBACK_THEORY.md` 与已核验的 Pro 建议。原始 v4、Pro 文本和旧 dependent-rollouts 档案保持原字节，不在原文上修订。
2. 先冻结问题、estimand、反馈配置、数据划分、预算和判据，再运行实验。不得按结果换任务、换 mask、删组、补样或启动无依据的 GPU 扫描。
3. 模型可替换。优先使用现有权重与适配当前资源的模型；`Qwen2.5-Coder-1.5B-Instruct` 只是 v4 的历史默认示意，不是开跑条件或结论对象。准确记录实际 model/tokenizer revision、精度、参数范围和梯度存储策略。
4. 已获授权的实验机工作可以直接运行原生 Python 和 GPU。Docker、chroot、固定 Conda、固定主机或重复批准都不是前置条件。直接 Hugging Face 不通时，优先复用本地权重，或使用镜像和 AutoDL 支持的下载路径。
5. 机器角色不写死。当前 Mac M4 MAX 本轮只做 CPU 工作；`aidemo` 可用但不是唯一开发环境。家用 PC 存在，规格未知。主机现场信息只写在 `docs/operations/EXPERIMENT_HOST.md`。
6. Actor 回答 IID；整个评分配置向量与整个回答向量独立。S/I 对每条固定回答保持完整反馈边际相同。一题一组 prompt；全同奖励、截断、解析失败和候选超时均按冻结规则保留，基础设施失败单独暂停并记录。
7. 主机制 loss 是 detached advantage 乘 response-token logp **之和**，包含实际 EOS，不包含 padding，不补 EOS。on-policy、`ddof=0`、根号外 `epsilon=1e-6`；不使用 clipping、KL reward、长度平均或 off-policy epochs。Adam 等在线训练设置另报。
8. CPU 恒等式、随机模型 plumbing、预训练 score/梯度 probe、自然反馈效应、局部后果和在线训练是不同证据层级。均值不变不代表方差、Adam 状态或有限步轨迹不变；不得编造模型结果、吞吐或 GO。
9. C 与 D 独立；prompt 是总体推断单位。原始 actor 组拥有不确定性，虚拟 mask、重排或精确积分不是新增独立样本。冻结有符号功能读出、正缩放对照和多重比较范围。
10. 每阶段使用新输出目录，保存命令、版本、配置与数据 hash、完整输出、失败原因和预算 ledger。不要覆盖已有结果。训练必须由带 hash 证据的 pilot GO 解锁。
11. `make test`、`make lint`、`make exact` 是代码检查入口，可用 `PYTHON`/`RUFF` 选择环境。检查通过不自动构成预训练实验或 pilot GO。
12. 修改尽量小，保护无关工作。优先完成已授权、可审计的下一阶段；遇到资源或合同失败先诊断，不用盲扫掩盖问题。

文档职责：`agent.md` 保存核心约束；`AGENTS.md` 只做自动发现入口；`index.md` 维护详细文件、理论、报告和结果索引；`README.md` 只写项目详情及 Mac/家用 PC 环境；服务器信息集中在独立运维记录。

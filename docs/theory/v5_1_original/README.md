# v4 诊断 / v5.1 修订交付包

首先阅读 `V4_DIAGNOSIS_AND_V5_1_EXECUTION.md` 的 §0、§2、§4、§10。

运行 CPU 参考实现：

```bash
python -m pip install numpy
python verify_v5_1.py
```

环境应锁定实际 NumPy/Python 版本。`verify_v5_1_results.json` 是本轮真实执行结果，重跑会覆盖同目录该文件。

参考函数：

- `full_stratified_weights(rewards, k)`：二值完整分层权重，输入 B×m，输出同形状。梯度按 `(weights[...,None]*scores).mean((0,1))` 聚合。适用于跨层、层内样本都独立的采样律。
- `full_stratified_grid_weights(rewards, k, scale)`：整数网格奖励，输出 B×m×3 的平均 A、A+、A-。真实奖励为整数/scale。
- `cross_weights`：旧一般独立块估计器。
- `iid_weights`：强 IID-all 对照；不可以直接对有信息的平衡分层银行当成 IID 使用。
- `brute_full_weights` / `brute_grid_weights`：独立枚举参照。

代码只包含 CPU 核验和权重参考，不包含模型采样器、训练引擎或服务器操作。它没有读取原 v4 的 d 或缺失失败投影。不要把脚本内人工概率模型的数值写成真实 LLM 结果。

`execution_contract.json` 是需接入本地 manifest 的合同模板，不是可直接运行的训练配置。null 字段由真实资产解析；不能猜测或填虚构值。模型实际实现应先冻结预算余额、版本、参数空间、随机流和数值合同。

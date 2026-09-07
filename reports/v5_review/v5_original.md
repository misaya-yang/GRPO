# Dependent Group RL v5 补充方案：恢复 IID 组更新，并识别相关采样的净收益

**用途：与 v3、v4 配套的理论补充、研究取舍及 Codex 实现合同。**  
**文献核验截止：2026-09-06。**  
**证据状态：本轮完成自含推导和 CPU 有限概率模型核验；没有运行预训练模型、GPU 训练、付费 judge 或真实吞吐测试。**

> 首轮预算仍为 10h RTX 5090 或 5h Pro 6000；完整实验上限仍为 200h / 100h。不同硬件分别计时。这些数字是支出上限，不是速度预测。
>
> 本文件的“证明”指列明假设下的数学论证，CPU 检查是有限实例核验。两者都不是优先权、模型收益或录用概率的证明。

---

## 0. 建议推进的核心

附件提出把 verifier coupling 扩大成“joint sampler + group credit operator”的一般 RL 问题。这个扩展有价值，但条件期望、score 正交性、控制变量、U-statistic、分层采样都是既有工具。仅把它们放在一个框架里，还不足以实质抬高贡献。

本补充建议把主问题明确为：

> **当 rollout 生成可以相关，而训练仍希望保留某个既定 IID 组算法时，能否用同样数量的实际回答，精确恢复该算法的平均更新？什么时候相关生成能在最强 IID 重组参照之上进一步降低梯度方差，并在实际计算预算下改善学习？**

建议的算法候选由三部分组成：

1. 独立生成若干采样块；块内允许相关。
2. 从不同块各取一条回答，组成虚拟训练组；对这些组做精确平均。
3. 用奖励计数或低维充分统计量，将这个平均化成每条实际回答的训练权重。

**需要争取的贡献是一个完整结果：平均更新保持、相对完整 IID 子组平均的净收益判据、可用于 GRPO/PPO 的精确权重计算，以及有实际效应的验证。**

原始 product-form estimator 和 U-statistic 不是本项目发明；分层 U-statistic 也有很早的统计文献。本轮新增的是面向本项目的完整推导、适用边界、GRPO 实现公式和决定性实验合同。其具体假设—结论组合的发表新颖性仍需谨慎陈述。[R1–R4]

### 0.1 三个研究版本的上限

| 版本 | 需要交付的实质内容 | 判断 |
|---|---|---|
| 统一解释 | 条件信用、baseline 偏差、更多反例 | 主要仍是机制与理论整理；附件中的一般恒等式不能独自支撑升级 |
| 本文主版本 | 固定 IID 参照，恢复原组更新；精确计算；说明相对完整 IID 平均何时获益 | 可以成为算法论文候选；真实收益尚未确认 |
| 后续加强版本 | 根据独立校准选择采样块结构，并在较多 prompt 与较多同题回答之间分配预算 | 可能影响更广泛的 RL 采样设计；必须先通过主版本，不能先扩充 GPU 项目 |

---

## 1. 对附件建议的逐项审查

附件指用户上传的《粘贴的 markdown (1)。md(20260907-033740)》，以下保留其主要问题组织。[U1]

| 附件建议 | 审计结论 | 本补充采用的处理 |
|---|---|---|
| 单轨迹 on-policy 不确定组更新 | 正确；QuasiMoTTo 已明确讨论相关生成与 RLOO 偏差 | 作为研究动机，不申报为首次发现 |
| 条件信用是完整接口 | 正确，是条件期望的直接应用，v3 已有 | 保留为统一记号，不作为新增主定理 |
| 两个算法等价当且仅当条件信用差投影为零 | 对平均槽位信用正确 | 非交换情形应先对槽位取平均；不能要求每个槽位分别为零而称其必要 |
| 条件均值对应“目标改变” | 需要收紧 | 它决定固定参数处的平均向量；不自动给出全局标量 objective |
| residual 只改变噪声，因此可保留以降方差 | 均值部分正确，降方差不自动成立 | 最优控制变量系数须对完整组梯度计算，包含跨样本协方差 |
| 用 cross-fitting 估计条件信用即可恢复目标 | 不成立 | cross-fitting 避免部分依赖，不能消掉模型误差造成的一阶偏差 |
| 所有真正依赖伙伴的 baseline 都不可能普遍安全 | 量词过强 | 总更新可以有槽位间抵消，也可以显式正交化；需要限定函数类和非退化条件 |
| 跨独立 block baseline | 正确，但 v3 §7 已有，经典独立 baseline 也覆盖 | 作为 RLOO 特例与对照，不当作新算法核心 |
| 最优 Bernoulli count coupling 带来最大梯度收益 | 前半是概率约束结果，后半推不出 | count coverage 不决定 score 梯度协方差；还要求生成时可实现 |
| 三动作 curl 反例与可积性作为新主线 | 查重风险很高 | SoftmaxGRPO 已有三个奖励水平的显式非保守反例；本补充不扩展此主线 |
| 同时做 tabular、控制任务、LLM、verifier、多代理 | 当前预算下过宽 | 一个有限 oracle、一条真实模型主线、通过后一个迁移场景 |

### 1.1 residual 修正的正确公式与局限

固定参数处令

\[
m_i(y)=\mathbb E[A_i\mid Y_i=y],\quad
T=\frac1K\sum_i t_i S_i,\quad
C=\frac1K\sum_i[A_i-m_i(Y_i)]S_i.
\]

则 \(\mathbb EC=0\)。当 \(\mathbb E\|C\|^2>0\) 时，完整组估计器 \(T+\lambda C\) 的 trace variance 最优标量为

\[
\lambda^*=-\frac{\mathbb E\langle T-\mathbb ET,C\rangle}{\mathbb E\|C\|^2}.
\tag{1}
\]

分子含所有 \(i,j\) 的交叉项。只计算单样本 \(t_iS_i\) 与自己的 residual 的协方差，通常不是这个优化问题。逐项条件化也不是对整个组梯度共同做一次 Rao–Blackwell 条件化，不能直接推出组方差下降。

若用独立训练的 \(\hat m_i\) 替代 \(m_i\)，修正产生的均值误差仍是

\[
\frac\lambda K\sum_i\mathbb E[(m_i(Y_i)-\hat m_i(Y_i))S_i].
\tag{2}
\]

它一般对回归误差一阶敏感。没有额外正交结构就不能称“双重稳健”，也不能声称 cross-fitting 使其严格无偏。对完整 LLM 回答学习 \(m_i(Y_i)\) 还需要大量同分布数据。因此本轮主方法不依赖这个回归器。[U1, §三；U2, §2]

---

## 2. 最接近文献及剩余空间

这里的“未见覆盖”仅指实际核对的正文范围；不代表穷尽检索或优先权确认。

| 文献 | 已有内容 | 本项目必须避开的重复 | 本补充具体要交付的部分 |
|---|---|---|---|
| QuasiMoTTo, 2607.01179v1 | 精确边际的相关 rollout；独立分层与 lattice；RLOO 配对修正及支持不匹配讨论 | 相关采样、baseline 偏差、用联合概率修正 | 不要求块内配对支持的组更新恢复；处理完整组归一化；测相对强 IID 参照的净收益 |
| CARMS；Coupled Gradient Estimators, NeurIPS 2021 | 耦合离散 score 梯度、重要性或条件化校正 | “相关采样 + 无偏梯度”这一宽泛想法 | 不依赖块内联合 PMF 的独立块方案及 normalized group kernel 的精确计算 |
| Product-form estimators, 2102.11575v3 | 经验边际的乘积、无偏性、方差改善、广义 U-statistic 与计算代价 | 虚拟笛卡尔积本身，或把组合数当有效样本数 | 有相关样本的块条件、完整 IID 竞争者、RL 权重计算与梯度影响结构 |
| Demystifying GRPO, 2603.01162v3 | 明确设定组内 IID output–reward pairs，使用 U-statistic 分析 | U-statistic 身份和经典 Hoeffding 展开 | 独立随机经验测度构成的 block U-statistic；精确比较的方差差额 |
| On Advantage Estimates for Max@K, 2606.06080v1 | 多种 Max@K 信号、all-subsets 关系与等价性 | 只做子组平均、只换有限批权重 | 明确保持标准化 K 组参照，与完整 IID K 子组平均比较 |
| SoftmaxGRPO, 2608.09271v1 | 二值有限组目标，以及多奖励水平下的非保守反例 | 三动作 curl 作为新发现 | 只将其说明的 objective 边界作为本研究的适用限制 |
| Taga, 1973, Generalized U statistics for stratified random samples | 原始出版记录与摘要表明分层 U-statistic 及其方差是早期主题 | 声称首次结合分层与 U-statistic | 本轮未取得正文，无法排除一般统计定理的直接等价；不把一般方差公式当独占 novelty |

**已有统计方法的简单移植，是本方案最大的 novelty 风险之一。**成功的论文不能只展示本文式 (4) 的构造。它需要有分量的 RL 特定结论：完整组更新的可计算恢复、比强 IID 参照更低的实际估计误差、可预测的胜负边界，以及真实学习中的收益。

本轮查询还覆盖了 `GRPO product-form`、`GRPO Tensor Monte Carlo`、`policy gradient independent blocks stratified`、`GRPO Rao-Blackwell group`、分层 U-statistic 和影响函数采样。没有查到全文可核验、直接同时给出本补充全部 RL 设定与实现公式的工作；这个“全部组合”本身不构成新颖性证明。[R1–R8]

---

## 3. 数学合同：先指定希望保持的更新

固定 prompt \(x\)、生成 checkpoint \(\theta_0\)、奖励规则、解码分布及长度上限。暂省略 \(x\)。设单样本

\[
Z=(Y,U,\xi)\sim P,
\qquad S(Y)=\nabla_\theta\log\pi_\theta(Y)\big|_{\theta_0}.
\]

\(U,\xi\) 是可选的外生反馈随机性。只用固定程序 verifier 时可直接令 \(Z=Y\)。

指定一个 **K 样本、置换对称、平方可积的向量核**

\[
H(Z_{1:K})=\frac1K\sum_{i=1}^K a_i(Z_{1:K})S(Y_i),
\qquad
G_K=\mathbb E_{P^{\otimes K}}H.
\tag{3}
\]

这里的核是“一组数据会产生什么原始梯度”。非对称实现可以先对槽位做显式对称化；后面的标准 U-statistic 方差公式使用对称性。

对于二值 GRPO，沿用 v3/v4：总体标准差 `ddof=0`，`epsilon=1e-6` 在根号外；全同奖励组优势置零并计入平均；对 token score 求和，对回答固定数量平均。

**保持 \(G_K\) 与保持原始 expected reward 梯度是不同合同。**RLOO 的 IID 参照等于 \(\nabla J\)；normalized GRPO 的 IID 参照一般包含 prompt 成功率重加权。本文不同时声称既完全保留 GRPO，又将它自动变成 REINFORCE。[U2, §1；U3, §2–3；R6]

完整回答的公共支持、可微性、EOS/截断、padding、数值精度、score 约定继承 v3/v4。不能对条件分层分布的 log probability 求导来代替原始策略 score。

---

## 4. 用独立采样块恢复 IID 组更新

### 4.1 构造

生成 \(B\ge K\) 个相互独立、同分布的采样块，每块含 \(m\) 条实际回答，总数为

\[
N=Bm.
\]

第 b 个块为 \(\mathcal B_b=(Z_{b1},\ldots,Z_{bm})\)，经验测度为

\[
\widehat P_b=\frac1m\sum_{j=1}^m\delta_{Z_{bj}}.
\]

要求 \(\mathbb E\widehat P_b=P\)。块内可以相关；如果需要每个槽位都具有 P 边际，可做与数据无关的随机排列。实际上，以下均值定理只需要经验测度无偏。

从 B 个块中选 K 个不同块，每块选一条回答，对全部选择平均：

\[
\boxed{
\widehat G_{B,m,K}
=\frac1{\binom BK}
\sum_{I\subseteq[B],\,|I|=K}
\int H\prod_{b\in I}d\widehat P_b.
}
\tag{4}
\]

有 \(\binom BK m^K\) 个虚拟组，但实际只有 N 条回答。\(B=K\) 是完整笛卡尔积；\(m=1,B=N\) 退化为完整 IID U-statistic。

### 4.2 命题：平均更新保持

在上述独立性与可积性条件下，

\[
\boxed{\mathbb E\widehat G_{B,m,K}=G_K.}
\tag{5}
\]

**证明。** 固定任意 K 个不同块。对独立随机经验测度依次取期望，利用 \(\mathbb E\widehat P_b=P\)，其乘积积分的期望为 \(\int H\,dP^{\otimes K}\)。对块子集取平均仍相同。□

它不要求块内 \(Q(z_i,z_j)>0\) 覆盖全部 \(P\otimes P\) 支持，不需要密度比，也不需要估计 \(\mathbb E[A_i\mid Y_i]\)。其统计结构属于 product-form 与 U-statistic 方法。[R2–R3]

**重要限制。**

- 不同块必须在给定 prompt、checkpoint 和已冻结历史后独立；共享 judge 随机数、共享环境状态或共享 lattice shift 都可能破坏条件。
- 一个虚拟组不能从同一相关块取两条回答。
- 奖励必须能作为所选样本的既定值重用。若 judge 看完整组后比较评分，重组后可能需要重新评分，本节低成本权重实现就不直接适用。
- 它保持指定的 IID K 组更新；将 K 换成 N 是另一个目标。

### 4.3 为什么比跨 block baseline 更一般

对 RLOO，式 (4) 会化为

\[
\widetilde a_{bj}=r_{bj}-\frac1{B-1}\sum_{c\ne b}\bar r_c.
\tag{6}
\]

这仍是经典独立 baseline，v3 已经有其两块版本。对组标准差、排名或其他非线性信用，式 (4) 平均的是整个组核，不仅是分子。因而不会犯“换了独立 baseline 却仍除自身组随机标准差”的错误。[U2, §7]

---

## 5. 什么时候能保证方差不增加

### 5.1 对采样块的充分条件

若对任意平方可积向量函数 f，每个块满足

\[
\operatorname{Cov}(\widehat P_b f)
\preceq\frac1m\operatorname{Cov}_{P}(f),
\tag{7}
\]

则可以将它用于后续的方差比较。正确边际本身不保证式 (7)。

一个可实现实例是独立等质量分层。写成

\[
P=\frac1m\sum_{j=1}^mP_j,
\qquad Z_{bj}\sim P_j\text{，各层独立}.
\]

则

\[
\boxed{
\operatorname{Cov}(\widehat P_b f)
=\frac1m\left[
\operatorname{Cov}_{P}(f)
-\operatorname{Cov}_{J\sim\mathrm{Unif}[m]}\big(\mathbb E_{P_J}f\big)
\right].
}
\tag{8}
\]

**证明。** 独立层的平均方差为 \(m^{-2}\sum_j\operatorname{Cov}_{P_j}f\)。对均匀混合分布 P 使用全协方差分解。□

v3 的独立算术分层给出这样的 \(P_j\)。即使同一终止回答横跨多个算术层、层标签不是回答的确定函数，这个混合测度论证仍成立。必须使用每层私有 fresh randomness；共享偏移 lattice 不自动满足式 (7)。[U2, §3、§6]

### 5.2 方差比较链

定义：

- \(\Sigma_{\mathrm{strat,cross}}\)：独立分层块加式 (4)。
- \(\Sigma_{\mathrm{iid,cross}}\)：每块含 m 个 IID 样本，加同一式 (4)。
- \(\Sigma_{\mathrm{iid,split}}\)：用 N 个 IID 样本组成互不重叠的 K 组后平均；假设 K 整除 N。

则

\[
\boxed{
\Sigma_{\mathrm{strat,cross}}
\preceq\Sigma_{\mathrm{iid,cross}}
\preceq\Sigma_{\mathrm{iid,split}}.
}
\tag{9}
\]

第一个不等式可逐块替换证明：给定其余块，式 (4) 对当前经验测度是一个常数加线性积分；条件均值不变，式 (7) 降低条件协方差；再用全协方差公式。第二个不等式也可由下一节各阶系数逐项比较证明。

**这个保证没有说比完整 IID 子组平均更好。**完整 IID U-statistic 是更强的对照，必须保留。

### 5.3 为什么“块内任意相关都更好”是错误的

若每个块只是把一条 IID 回答复制 m 次，仍满足正确边际与跨块独立。\(B=K\) 时，式 (4) 等于一个普通 IID K 组核；它的方差是同 N 预算、平均 m 个 IID K 组的 m 倍。

因此，式 (5) 适用的采样器类比式 (9) 更宽。不能将均值保持定理扩大成普遍效率保证。

---

## 6. 核心加强：与完整 IID 子组平均的精确比较

这节提供一个可以判定收益与失败的理论对象。其证明使用经典 Hoeffding 分解；不声称发明这一分解。[R3–R5]

### 6.1 核的各阶部分

对 \(P^{\otimes K}\) 做对称 Hoeffding 展开：

\[
H=G_K+\sum_{s=1}^K\sum_{I\subseteq[K],|I|=s}h_s(Z_I),
\tag{10}
\]

其中每个 \(h_s\) 在任意一个自变量上积分均为零。定义

\[
\Sigma_s=\mathbb E[h_sh_s^\top].
\]

把 \(h_s\) 对 s 个独立块的经验测度积分，记为 \(\widetilde h_s\)，并令

\[
T_s=\operatorname{Cov}(\widetilde h_s).
\]

\(\widetilde h_s\) 同样在任一块变量上条件均值为零。因此 block U-statistic 的精确协方差为

\[
\boxed{
\Sigma_{\mathrm{strat,cross}}
=\sum_{s=1}^K\frac{\binom Ks^2}{\binom Bs}T_s.
}
\tag{11}
\]

若块内 IID，则 canonical 性使不同样本索引的交叉项消失，

\[
T_s^{\mathrm{iid}}=\frac{\Sigma_s}{m^s}.
\tag{12}
\]

N 个 IID 回答的完整 K 子组平均则满足

\[
\Sigma_{\mathrm{iid,all}}
=\sum_{s=1}^K\frac{\binom Ks^2}{\binom Ns}\Sigma_s.
\tag{13}
\]

**证明要点。** 对式 (4) 代入 (10)，每个 s 块项出现 \(\binom{B-s}{K-s}\) 次，故其系数为 \(\binom Ks/\binom Bs\)。不同 canonical 块子集正交，方差累加得到 (11)。IID 块中，对任一索引不匹配的乘积项在该单样本变量上积分为零，得到 (12)。式 (13) 为普通完整 U-statistic 的同一计数证明。□

### 6.2 净收益等式

独立分层满足

\[
\Delta_s:=\frac{\Sigma_s}{m^s}-T_s\succeq0.
\]

于是

\[
\boxed{
\Sigma_{\mathrm{iid,all}}-\Sigma_{\mathrm{strat,cross}}
=\underbrace{\sum_{s=1}^K\frac{\binom Ks^2}{\binom Bs}\Delta_s}_{\text{分层减少的方差}}
-\underbrace{\sum_{s=2}^K\binom Ks^2
\left(\frac1{\binom Bs m^s}-\frac1{\binom Ns}\right)\Sigma_s}_{\text{限制跨块重组的代价}}.
}
\tag{14}
\]

右侧第二项半正定，因为 \(\binom Bs m^s\le\binom{Bm}s\)。s=1 两个系数相同，所以代价从二阶开始。

**这就是主研究判据。**只有分层收益足以覆盖重组限制的代价，才能超过完整 IID 对照。它直接阻止了“去偏后一定保留原采样收益”的跳跃。

对于固定读出矩阵 L，取 \(\operatorname{tr}(L\cdot L^\top)\) 即得所测功能空间的精确标量比较。只测几个方向时，不得宣称已经检验整个参数空间的半正定序。

### 6.3 一阶影响函数与大样本边界

令

\[
\psi(z)=h_1(z)=\mathbb E[H(z,Z_2,\ldots,Z_K)]-G_K,
\]

并定义层间协方差

\[
D_1=\operatorname{Cov}_{J}\big(\mathbb E_{P_J}\psi\big).
\]

由式 (8)，\(\Delta_1=D_1/m\)。固定 K、m、P 与层分布，让 B 增大，则

\[
\boxed{
\Sigma_{\mathrm{iid,all}}-\Sigma_{\mathrm{strat,cross}}
=\frac{K^2}{N}D_1+O(N^{-2}).
}
\tag{15}
\]

在任何 \(d^\top D_1d>0\) 的固定方向，足够大的 B 下存在严格改善。该结论不要求 \(D_1\) 满秩，不保证有限 N=8/16 已进入渐近区间，也不声称有全方向严格改善。

一个有限 N 的充分条件是：

\[
\frac{K^2}{N}\operatorname{tr}(LD_1L^\top)
>
\sum_{s=2}^K\binom Ks^2
\left(\frac1{\binom Bs m^s}-\frac1{\binom Ns}\right)
\operatorname{tr}(L\Sigma_sL^\top).
\tag{16}
\]

它忽略了高阶的非负分层收益，因此只充分、不必要。CPU 例子也会展示 \(D_1=0\) 但高阶收益仍然存在的情况。

### 6.4 对 sampler 设计的实际含义

最该预测的是 \(\psi\)，即一条回答对**整组更新**的条件影响。只按通过率 q、回答是否重复、语义距离或奖励计数选择 sampler，没有得到式 (14) 的充分信息。

这也把 v3 的身份—信用理论接到正向算法设计：同样的 reward counts，可以对应不同的 \(\psi\) 层均值与梯度方差。

---

## 7. 二值 GRPO 的影响函数可以直接计算

记

\[
a(r,n)=\frac{r-(r+n)/K}{\sqrt{(r+n)(K-r-n)}/K+\varepsilon},
\]

全同组取 0。令 \(p=\mathbb E r\)、\(g=\mathbb E[rS]\)，定义

\[
u_r=\mathbb E_{M\sim\mathrm{Bin}(K-1,p)}a(r,M),
\]

\[
t_r=\mathbb E_{L\sim\mathrm{Bin}(K-2,p)}[a(1,L+r)-a(0,L+r)].
\]

则

\[
\boxed{
\psi(y)=\frac{u_{r(y)}}K S(y)
+\frac{K-1}{K}t_{r(y)}g-G_K.
}
\tag{17}
\]

**证明。** 固定第一条回答。它自己的信用贡献为 \(u_rS/K\)。对另一个槽位，奖励 1 对应的无条件 reward-score 期望为 g，奖励 0 对应为 \(-g\)，其余 K−2 个奖励计数服从二项分布，因此该伙伴槽位的条件贡献是 \(t_rg/K\)。合并 K−1 个伙伴并减去均值。□

这同时需要 \(\mathbb ES=0\)、固定二值奖励以及 IID 参照。连续奖励和任意 mark 特征不能直接套这个简式。

沿用 v3 的层统计 \(q_j=\mathbb E_{P_j}r\)、\(m_j=\mathbb E_{P_j}S\)、\(c_j=\mathbb E_{P_j}[rS]\)：

\[
\mathbb E_{P_j}\psi
=\frac{u_0m_j+(u_1-u_0)c_j}{K}
+\frac{K-1}{K}[t_0+(t_1-t_0)q_j]g-G_K.
\tag{18}
\]

因此完整确认阶段可以由独立校准银行预测一阶方差收益；训练算法本身不需要 p、q、m、c 的回归估计。

**校准限制。**有限样本中的 p、g 与层均值存在误差。独立校准、方向冻结及误差传播必须保留；不能把同一确认银行上的方差差额代回 (18) 再称为成功预测。首轮的少量 prompt/重复也不支持声称获得总体严格数值证书。

---

## 8. 精确权重：不枚举海量虚拟模型计算

### 8.1 二值奖励的动态规划

对已生成的 B×m 奖励银行，记 \(q_b=m^{-1}\sum_j r_{bj}\)。这只是**固定银行的经验通过比例**，不是对总体 p 的 plug-in 逆权校正。

给定自身块 b，均匀选择 K−1 个其他块，每块均匀取一条回答。伙伴成功数的 PMF 为

\[
\pi_{-b}(n)=
\frac{[t^{K-1}z^n]\prod_{c\ne b}\{1+t[(1-q_c)+q_cz]\}}
{\binom{B-1}{K-1}}.
\tag{19}
\]

自身奖励为 r 时，平均优势为

\[
\widetilde a_b(r)=\sum_{n=0}^{K-1}\pi_{-b}(n)a(r,n).
\]

于是式 (4) 精确化成

\[
\boxed{
\widehat G_{B,m,K}
=\frac1N\sum_{b=1}^B\sum_{j=1}^m
\widetilde a_b(r_{bj})S(Y_{bj}).
}
\tag{20}
\]

**证明中的系数。**每个块进入均匀 K 子组的概率为 K/B；组内平均有 1/K；自身样本选择概率为 1/m。因此每条实际回答的基础系数为 1/N。

直接实现 (19) 的状态为“已选择的块数、成功数”，所有 focal blocks 总计 \(O(B^2K^2+N)\) 标量操作，空间可用 \(O(K^2+N)\)。K、B 较小时完全在 CPU 上处理。最终只需要对 N 条实际回答完成普通加权 loss 的一次训练计算；允许显存受限的梯度累积，不代表额外模型调用完全免费。

### 8.2 完整 IID 对照同样可以低成本计算

N 条 IID 回答共 C 个成功。固定自身 r 后，其他 K−1 个回答的成功数服从超几何分布：

\[
\widetilde a_{\mathrm{all}}(r)
=\sum_n
\frac{\binom{C-r}{n}\binom{N-C-(1-r)}{K-1-n}}
{\binom{N-1}{K-1}}a(r,n).
\tag{21}
\]

完整 IID U-statistic 为 \(N^{-1}\sum_i\widetilde a_{\mathrm{all}}(r_i)S_i\)。越界组合数定义为 0。

它不需要 \(\binom NK\) 次 backward。这个强对照必须实现，不能以其“计算昂贵”为理由删掉。[R3、R5]

### 8.3 有限网格奖励

若奖励是明确的 \(r\in\{0,1/L,\ldots,1\}\)，伙伴集合的标准差只依赖奖励和与平方和。将 DP 状态扩成

\[
(s,q,q_2)=\left(\#\text{selected blocks},\sum Lr,\sum (Lr)^2\right)
\]

即可精确处理网格奖励的 normalized advantage。epsilon 必须与奖励缩放一致；用整数单位时分母中的 epsilon 为 \(L\varepsilon\)。

复杂度取决于实际可达状态数。任意实数奖励不保证具有小状态空间；人为量化改变了 estimand，不能继续称同一连续奖励算法的精确实现。

### 8.4 PPO clipping 需要正负权重分别积分

设 \(\rho\) 为该实际回答或 token 的固定概率比，\(c(\rho)=\operatorname{clip}(\rho,1-\epsilon_c,1+\epsilon_c)\)。定义

\[
A^+=\max(A,0),\qquad A^-=\min(A,0).
\]

则

\[
\min(\rho A,c(\rho)A)
=A^+\min(\rho,c(\rho))+A^-\max(\rho,c(\rho)).
\tag{22}
\]

DP 分别累积 \(w^+=\mathbb E[A^+]\)、\(w^-=\mathbb E[A^-]\)，就能把完整虚拟组 clipped loss 化为

\[
\frac1N\sum_{bj}\left[
 w^+_{bj}\min(\rho_{bj},c(\rho_{bj}))
+w^-_{bj}\max(\rho_{bj},c(\rho_{bj}))\right].
\tag{23}
\]

对逐 token ratio，逐 token 使用同一回答的两项权重。权重、奖励、旧策略概率均 detach；边界的次梯度规则保持一致。

**反例。** A 以相同概率取 ±1，\(\rho=1.5\)，clip 上界为 1.2。真正平均 surrogate 为 −0.15；先平均 A 得到 0，再 clipping 则是 0。不能把平均优势直接送入任意非线性优化步骤。二值奖励下自身优势符号固定，单个均值权重恰好足够；一般网格奖励不具备这一简化。

式 (23) 对每个固定 actor 参数精确等于该银行完整虚拟 loss；对其梯度也成立。若 actor 参数经过当前银行多轮优化，不应把“随机选择后的参数处无偏”从固定参数结论直接外推。动态 Adam、聚合梯度裁剪和步长选择还属于后续非线性操作。

---

## 9. 更广的 RL 意义：同题采样和增加题目也必须比较

保持目标并降低同 prompt 方差，仍不保证是最好的计算使用方式。

固定 N，假设 K 整除 N。令

\[
V_{\mathrm{prompt}}=\operatorname{tr}\operatorname{Cov}_{x\sim\nu}(LG_K(x)),
\]

\[
V_{\mathrm{group}}=\mathbb E_x\operatorname{tr}\operatorname{Cov}(LH_x\mid x),
\quad
V_{\mathrm{cross}}=\mathbb E_x\operatorname{tr}\Sigma_{\mathrm{strat,cross},L}(x).
\]

同一 prompt 使用 N 条回答的方案，其总方差为

\[
V_{\mathrm{prompt}}+V_{\mathrm{cross}}.
\]

把同样 N 条回答分给 N/K 个独立 prompt，每题一个 IID K 组，得到

\[
\frac KN(V_{\mathrm{prompt}}+V_{\mathrm{group}}).
\]

所以同题相关方案在这一响应数量预算下胜出的精确条件是

\[
\boxed{
V_{\mathrm{cross}}<\frac KN V_{\mathrm{group}}
-\left(1-\frac KN\right)V_{\mathrm{prompt}}.
}
\tag{24}
\]

如果右侧非正，任何非负的 \(V_{\mathrm{cross}}\) 都无法满足这个比较。增加 prompt 的竞争者可能更合算。

这个结论使用相同 token 上限与响应数量，实际 wall time 还包含 prefix 复用、prompt 长度、不同回答长度和批处理利用率；需要实测成本。不能将响应数量等价为 GPU 小时。

**因此完整实验要有“同计算预算增加 prompt”的参照。**只有同 prompt 图更漂亮，仍不足以证明 RL 训练预算利用率提高。

### 9.1 有条件的学习联系

二值 IID GRPO 在相应假设下可写成 \(G_K=\nabla\Phi_K\)，这一目标解释已有文献。若 \(\Phi_K\) 为 L-smooth，SGD 条件无偏，则

\[
\mathbb E\Phi_K(\theta+\eta\widehat G)\ge
\Phi_K(\theta)+\eta\|G_K\|^2
-\frac{L\eta^2}{2}\left(\|G_K\|^2+\operatorname{tr}\operatorname{Cov}(\widehat G)\right).
\tag{25}
\]

更低方差改善这个保证中的噪声项；它不自动证明任意真实非线性模型的一步收益排序。若局部目标恰为凹二次函数，协方差半正定下降才直接给出该二次目标的一步期望收益改善。真实模型需要小步与在线结果，不把经典 SGD 界当新收敛理论。

---

## 10. 本轮 CPU 实际结果

文件 `verify_v5.py` 只依赖 NumPy。它使用实际 categorical/Bernoulli score，核验有限银行 DP、Hoeffding 分解、协方差公式与边界例子。

| 检查 | 结果 |
|---|---|
| 二值 DP 对完全枚举 | 通过，覆盖 B=K 与 B>K |
| 网格 DP 对完全枚举 | 通过，包含奖励和、平方和 |
| 完整 IID 超几何权重对全部 K 子集枚举 | 通过 |
| 正负权重的 clipped surrogate 及对 ratio 的导数 | 通过 |
| RLOO 化为其他独立块的平均奖励 baseline | 通过 |
| 完整组影响函数式 (17) | 通过 |
| 各阶正交、完整方差与净收益式 (14) | 通过 |
| 4^8=65,536 个 IID 回答银行的均值及方差核验 | 通过 |
| 正确边际但复制采样块导致方差增加 | 复现，m=2 时为同预算普通 IID 平均的 2 倍 |

本轮等式检查最大绝对误差：`1.4682699500667695e-14`，assertion 容差 `3e-11`。以交付 JSON 中的实际运行结果为准。

### 10.1 强 IID 参照上的正反例

下表均为数学 oracle，不是真实模型效果。方差指标为所定义 score 空间的协方差 trace。

| 模型与分层 | K,B,m,N | 完整 IID 方差 | 分层加跨块平均方差 | 比值 |
|---|---|---:|---:|---:|
| 四类 softmax；奖励对齐的两层 | 4,4,2,8 | 0.05168475 | 0.04080511 | 0.7895 |
| 四类 softmax；奖励对齐的两层 | 4,8,2,16 | 0.02293629 | 0.02040255 | 0.8895 |
| 四类 softmax；两层成功率都为 1/2 | 4,8,2,16 | 0.02293629 | 0.01218708 | 0.5313 |
| 四类 softmax；不对称奖励 | 4,4,2,8 | 0.03848554 | 0.02488220 | 0.6465 |
| 标量 Bernoulli；层标签与奖励/score 无关 | 4,4,2,8 | 0.00503432 | 0.00591194 | 1.1743 |

第三行尤其有用：四类均匀，奖励为 `(1,1,0,0)`，层分别为 `{0,2}` 和 `{1,3}`。每层通过率都为 1/2，所有独立层样本的奖励向量仍为 IID Bernoulli；reward-only coverage 无法解释 0.5313 的方差比。改善来自 score 与身份的结构。

最后一行则使用 \(P_\theta(A,B)=\frac12\mathrm{Bernoulli}_\theta(B)\)，奖励为 B，score 为 \(B-1/2\)，按无关标签 A 分层。它没有降低函数层面的采样噪声，却保留跨块重组的限制，所以比完整 IID 对照更差。

这两个例子支持“需要检查组更新影响结构”的数学主张。它们不支持“自然语言模型里一定有类似幅度”。

---

## 11. 首轮实验：直接判断算法价值

### 11.1 与 v4 的工作分工

v4 的反馈矩阵实验可保留为机制与适用边界。本补充的主实证改成相关 rollout，是因为目标已经升级为带计算收益的采样—估计联合方法。

已有文件仍明确尚无真实模型实验结果。Codex 应先检查项目是否已有可复用银行及 sampler，再决定实现缺口；不能假定所有资产都不存在，也不能把此前 CPU 检查标成已验证模型引擎。

### 11.2 默认配置

| 项目 | 预设 |
|---|---|
| 主模型 | `Qwen/Qwen2.5-1.5B-Instruct`，固定 revision/tokenizer/chat template |
| 首轮任务 | v3 已列的 SVAMP；固定数据拆分及数值答案 verifier |
| 参数空间 | 所有分支同一 rank-8 LoRA 配置；对该参数空间的原始边际 score 做审计 |
| 目标组大小 | K=4 |
| 分层宽度 | m=2，使用 v3 的独立算术分层 |
| 主采样块数量 | B=4，即 N=8；相邻独立宏批可合成 B=8,N=16 的预定次级检查 |
| 最大生成长度 | 192 新 token；EOS 和截断沿用 v3 |
| 解码 | temperature=1；不做 top-p/top-k 支持截断；dropout 关闭 |

这并不是对这些模型当前能力或运行速度的经验断言。任务是否学得动、解析是否可靠、完整链路是否符合预算，都需要独立 Dev 检查。[R9]

LoRA 只限定本次可训练参数空间，不改变权重推导。初始 LoRA 的部分梯度可能因标准初始化为零，所有分支必须相同；完整确认需要额外检查共同的、独立 IID 历史产生的一个中期 checkpoint，避免只在特殊初始化处有效。不能将 LoRA 结果自动扩大为全参数方差结论。

### 11.3 两个生成银行，多个廉价估计器

每个固定 prompt 都生成 IID 与独立分层两种银行。两种银行各含完整独立宏批；同一银行的 token、reward 和 score 供多个权重方案重用。

| 估计器 | 数据 | 用途 |
|---|---|---|
| IID-split | IID | 普通不重叠 K 组参照 |
| IID-cross | 同一 IID 银行 | 分离经典重组的收益 |
| IID-all | 同一 IID 银行 | 必须超过的完整 IID K 子组平均 |
| Strat-raw | 分层银行，每 K 条按预定原生成顺序成组 | 观察未经修正的耦合效应；它不一定保持目标 |
| Strat-cross | 同一分层银行 | 主算法候选 |

当 m<K 时，Strat-raw 按预先定义的相邻块拼成 K 条回答；不按结果挑组。

**不按 virtual groups 计生成 token 或样本量。**每个估计器的训练权重可以重算，但梯度回放成本必须计入预算；不能将五个 backward 算成一个。

### 11.4 首轮规模与费用

默认：16 个确认 prompt；每题、每个 sampler 8 个独立 N=8 宏批，共 2,048 条回答。另设 128 条 Dev 回答和 256 条独立读出/目标参考回答，合计上限 2,432 条初始回答，约 466,944 个最大 generated tokens，另有受预算限制的小步检查。

| 阶段 | 5090 上限 | Pro 6000 上限 |
|---|---:|---:|
| 模型 sampler/score 与吞吐合同 | 1.5h | 0.75h |
| Dev 与两种回答银行 | 3.0h | 1.50h |
| 权重、LoRA 梯度与方差审计 | 3.0h | 1.50h |
| 固定几何小步、独立读出 | 1.5h | 0.75h |
| 重跑余量 | 1.0h | 0.50h |
| 总计 | 10.0h | 5.00h |

所有模型/数据下载和 CPU 执行器准备应在租用 GPU 前完成。无实测吞吐时不能保证跑完默认规模；按固定顺序停止并报告覆盖。首轮失败不能把剩余预算自动投入另一个任务寻找阳性。

### 11.5 统计对象

首要判断是同一个 K、同一个 N 下，Strat-cross 相对 IID-all 的方差和成本。对预先固定的参数空间或读出 L，定义

\[
V_a=\mathbb E_x\operatorname{tr}\operatorname{Cov}(L\widehat G_a\mid x).
\]

用同题两个独立宏批的差可以无偏估计条件方差：

\[
\widehat v_{x,a}=\frac12\|L(\widehat G_{x,a}^{(1)}-\widehat G_{x,a}^{(2)})\|^2.
\tag{26}
\]

多个重复可组成预定不重叠配对或使用标准样本方差。同一 IID 银行上的 IID-cross/IID-all 使用配对比较。prompt 是跨题推断的聚类单位，宏批是同题的独立重复；virtual groups 和 token 都不是独立单位。

可以流式累积 LoRA 梯度向量和平方范数，不永久保存每条回答全参数梯度。若显存/时间只允许固定投影，结论限定于该投影。首轮 16 个 prompt 的区间可能很宽；bootstrap 是有限样本近似，不能宣称 distribution-free 证书。

报告均值差、方差比、实际端到端成本比以及两者乘积。均值保持检查应使用预定等效性容差和足够精度；“未显著不同”不能代替等效。算法推导正确时有限样本均值仍不必相等。

### 11.6 GO / KILL / INCONCLUSIVE

建议在 Dev 后冻结以下项目门槛；百分比是资源决策标准，非科学常数。

**GO 到完整确认：**数学/数值合同通过；相对 IID-all 的所测方差比上置信限小于 0.85，且方差与实际成本的乘积支持净改善。Dev 只允许在 B=4 与 B=8 两个预设结构中选择一次，确认前冻结主结构；确认后不得用另一个结构替换失败的主要终点。两个结构的结果都要完整报告。目标均值没有超出容差的可复现偏离，小步结果没有与预定模型系统性矛盾。

**KILL 当前算法版本：**精度足够时只赢 IID-split，无法超过 IID-all；收益完全被采样引擎/评分/回放成本吃掉；自然分层没有可分辨收益；或者所有实用收益都来自改变 K、筛掉失败组、改变长度权重等未声明因素。

**INCONCLUSIVE：**参考或方差分母接近零、CI 过宽、任务不学、吞吐不足。只根据实际方差和资源给出一次定量扩样决定，不换排序、seed、数据子集寻找阳性。

首轮同题门槛通过后，完整阶段还须通过式 (24) 对应的增加 prompt 对照。不能仅凭同题方差图宣称整体训练效率已提高。

---

## 12. 进一步抬高上限：允许自适应选择，但保持均值合同

固定 K，可以在多种 \((B,m)\) 结构中选择，\(m=1\) 对应 IID-all。设选择依赖 prompt 和过去训练历史 \(\mathcal H_t\)，在本轮新样本生成前完成。若每个候选在条件历史下都满足式 (5)，则

\[
\mathbb E[\widehat G_t\mid\mathcal H_t,x_t]=G_K(\theta_t,x_t).
\tag{27}
\]

这是条件期望的直接推论，但对算法有实际意义：方差模型即使估计不准，预先选择结构也不会像错误的条件信用回归那样直接引入均值偏差。

可以用独立 Dev 或过去 checkpoint 的校准估计式 (14) 和式 (24) 的相关量，决定是否使用 m=2，或采用 IID-all/增加 prompt 的方案。选择必须在当前组生成前冻结；看到当前奖励后改变块数、停止采样或挑组，会改变本合同。

不同 prompt 使用不同 N 时，仍按预定 prompt 权重平均每题估计器。若按总回答数统一归一化，就会把 N(x) 变成额外的 prompt 权重。

### 12.1 更激进的可研究项：按更新影响选择分层

式 (18) 提示：分层应尽量分开 \(\psi\) 的条件均值。可以研究只依赖当前前缀和过去校准、且不改变边际概率的 token CDF 排序。这属于后续方法候选，当前没有实现或收益证据。

必须同时满足：排序在生成当前 token 前确定；每个 token 原始概率不变；每层使用独立随机性；不能依据生成后的正确性将回答重新筛选入层；不能把不可因果获取的完整回答 \(\psi\) 排序当可执行 sampler。

如果 \(\hat\psi\) 的中心化 L2 误差不超过 \(\delta\)，条件期望的收缩性给出

\[
\sqrt{\operatorname{Var}(\mathbb E[\psi\mid J])}
\ge
\left(\sqrt{\operatorname{Var}(\mathbb E[\hat\psi\mid J])}-\delta\right)_+.
\tag{28}
\]

这是代理分层预测的误差边界，不是实际 \(\delta\) 很小的证明；向量版本用对应 L2 范数。基于影响函数的抽样设计在统计学中已有先例，不能把这一原则重新申报原创。[R10]

本轮不启动该 critic/排序实验。主版本若不能在预设 sampler 上越过强 IID 对照，应先报告失败原因，不自动升级成新的参数搜索。

### 12.2 verifier 分支的位置

外生配置也可独立分层，从而使扩展样本 Z 满足式 (8)；或在每块内部共享配置，仅使用式 (5) 的均值保证。两种情况必须区别。

若完整反馈矩阵已经免费或低成本可得，先对所有配置做精确积分会是更强的参照。随机配置调度不能在同样已知矩阵下凭空提供额外信息。因此 v4 在这里主要承担反馈关联的边界验证；不能靠刻意保留可积分的 judge 噪声制造新方法收益。

---

## 13. 完整实验预算与发表门槛

首轮通过后，再投入下表剩余阶段。它替代旧方案中的新增 GPU 路径，不与 v3/v4 的完整矩阵相加。

| 阶段 | 5090 h | Pro 6000 h |
|---|---:|---:|
| 首轮决定性实验 | 10 | 5 |
| 独立确认：方差、均值、影响函数预测、中期 checkpoint | 30 | 15 |
| 主任务在线学习，4 arms × 3 seeds | 100 | 50 |
| 第二场景确认或缩短在线复现 | 30 | 15 |
| 必要消融、成本与 prompt 分配对照 | 20 | 10 |
| 重跑余量 | 10 | 5 |
| 总上限 | 200 | 100 |

推荐主在线 arms：IID-all、Strat-cross、Strat-raw，以及同总预算增加 prompt 的 IID K 组方案。IID-split 和 IID-cross 保留冻结梯度及短在线消融；如其在线成本可负担，可替换冗余的 Strat-raw 长运行，变更需预先记录。

第二场景优先复用 v4 的 MBPP+ 与 `Qwen/Qwen2.5-Coder-1.5B-Instruct`，使用预定完整测试集通过与否作为二值主奖励，连续/网格反馈作为单独编译一致性分支。与主模型同属 Qwen 家族，不称跨架构证据。[U3, §10；R9]

30h / 15h 的第二阶段是否足够完成所需种子和学习长度，必须依实际吞吐与主实验学习速度决定。总上限内做不到，就缩小论文的实证主张，不能把不足的训练长度包装成完整复现。

### 13.1 以 solid accept 为目标，需要补齐什么

论文应最终能回答：

- **正确性：**为何相关生成仍精确保持指定 K 组的平均更新；实际数值引擎是否匹配证明。
- **必要性：**已有的跨组 baseline、配对重要性修正、完整 IID 子组平均分别解决了什么，为什么本设置仍需要提出的构造。
- **收益条件：**哪些组更新交互项决定额外收益；至少有一个可独立预测的正例与一个自然或受控的失败边界。
- **实用性：**同 GPU 时间/生成与评分预算下，结果超过强 IID 参照，并且不是仅将预算从较多 prompt 挪到较多同题回答造成的表象。

可以将“相同学习水平下实际 GPU 用时减少约 15%–20%，或同预算有稳定且有意义的指标改善”作为完整投入门槛，按 Dev 测得的任务噪声冻结。没有普遍保证这个阈值可达到，也不能从 CPU 的 0.5313 方差比推断训练加速幅度。

若结果只达到“构造正确 + 赢普通 split + 不赢完整 IID”，论文仍会面临“经典统计重组的直接应用”的核心质疑。若算法、强基线净收益、预测边界和真实学习都成立，才值得把主要定位提升到多样本 RL 的算法设计。

---

## 14. Codex 实施合同

### 14.1 先做的 CPU 工作

1. 阅读本补充 §3–9，与现有 v3/v4 的 score、归一化和数据结构核对。
2. 运行 `verify_v5.py`，输出新环境的 JSON；不要复制本轮结果当作本地运行记录。
3. 为 `cross_binary_weights`、`cross_grid_weights`、`iid_all_weights` 加项目测试，先以完全枚举为参照。
4. 检查实际采样器是否为 independent stratification，是否有不可追踪的共享随机数或有限精度 release 捷径。
5. 冻结 manifest、模型版本、数据拆分、seed、奖励和预算计量，再进入 GPU 阶段。

### 14.2 每条回答至少记录

```text
checkpoint_hash, adapter_hash, tokenizer_hash, prompt_id, bank_id
macro_id, block_id, stratum_id, private_rng_id, block_rng_id
K_target, B_blocks, m_width, sampler, token_order_rule
input_ids, response_ids, eos_position, truncated, active_mask
base_token_logp_at_theta0, raw_reward, verifier_version
arithmetic_interval_state_before_release, release_found, release_token
cross_adv_mean, cross_adv_positive, cross_adv_negative
iid_complete_adv, original_group_id, original_group_adv
```

\(B,m,K\) 是三个不同参数，不可把 `group_size` 同时用于三者。

### 14.3 loss 与数据合同

- 每个 macro 恰有 \(N=Bm\) 条实际回答；无结果筛组、补采、去重或提前停止。
- 每个虚拟组恰有 K 个**不同 block**。跨 prompt 不做虚拟组；prompt 分布混合会改变目标。
- 所有权重在 CPU/FP64 或经对照确认的精度计算，detach 后送入模型。
- 先验证未裁剪 on-policy loss；有 clipping 时用正负权重编译，并与小银行完全枚举的 loss/gradient 对齐。
- 逐回答长度归一化、随机 token 总数归一化、KL、梯度裁剪需独立标记。它们可以作为后续声明过的 kernel，但不能默默混入本文主合同。
- KL 项若为逐回答可加，按实际样本算一次；不能因虚拟重用次数放大。
- 模型生成与 teacher-forced logp 核对；默认 `temperature=1`，`dropout=0`。

### 14.4 必须输出的报告

```text
contract_report.md                 # 哪些数学/数值条件实际满足
prior_art_delta.md                 # 与本文引用的直接近邻逐条比较
frozen_manifest.json
sampling_coverage.json             # 完成题目、回答、截断、异常及预算
mean_and_variance_report.md        # target K 固定；含 IID-all
cost_report.json                   # 真实 GPU/CPU wall time、tokens、评分次数
pilot_decision.md                  # GO / KILL / INCONCLUSIVE 和证据
```

若仅得到固定银行恒等式、没有足够独立宏批，不得输出“总体方差下降”。若只得到固定 checkpoint 的投影方差，不得输出“训练加速”。预算耗尽时提交已完成结果与覆盖限制，不继续展开新模型/任务。

---

## 15. 贡献与证据登记

| 内容 | 当前地位 |
|---|---|
| 条件信用表征、score 正交、RLOO cross baseline | 经典工具与 v3 已有内容 |
| 产品经验测度、U-statistic、分层方差公式 | 已有统计学基础；本项目不申报原创 |
| 式 (4)–(5) 的本项目构造 | 本轮明确化的算法候选；一般形式有直接先例 |
| 式 (14) 的强 IID 对照净收益、式 (17) 的 GRPO 影响函数、式 (24) 的 prompt 分配边界 | 本轮自含推导与有限核验；发表新颖性取决于具体与既有结果的差异，不能以未搜到同标题证明 |
| 二值/网格 DP、正负 clipping 权重编译 | 本轮可运行参考实现；未进入实际 LLM harness |
| 自然 LLM 中的收益条件、校准精度、方差改善与成本收益 | 未验证 |
| 自适应结构、按影响函数分层 | 有条件的后续研究项，未实现 |
| ICLR solid accept | 研究目标，不是当前结论 |

---

## 参考来源与阅读范围

### 用户材料

[U1] 用户上传《粘贴的 markdown (1)。md(20260907-033740)》。完整阅读，重点核查一般条件信用、残差控制变量、独立 block、count frontier 与全局目标建议。

[U2] 《Dependent Rollouts v3：条件信用选择、停止局部性与归一化的识别边界》。使用 §1–9 的理论与 §10 的算术条件采样合同；不将已有结果重复申报为 v5 增量。

[U3] 《Dependent Feedback v4：验证器关联、组归一化与条件信用的理论审计》。使用 §0–3、§9–11 的目标、反馈边界与实验限制；v5 的新增实证分支替代新增 GPU 矩阵，不叠加预算。

### 本轮核对的公开原始文献

[R1] QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling. arXiv:2607.01179v1. 核对采样构造、§2.4.2 配对修正/支持限制、训练中 mean-centering 定义。https://arxiv.org/html/2607.01179v1

[R2] Dimitriev and Zhou. CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator. NeurIPS 2021, arXiv:2110.14002. 以及 Dong, Mnih, Tucker. Coupled Gradient Estimators for Discrete Latent Variables. NeurIPS 2021, arXiv:2106.08056v2. 核对耦合离散梯度与校正设定。https://arxiv.org/html/2110.14002 ; https://arxiv.org/html/2106.08056v2

[R3] Kuntz, Crucinio, Johansen. Product-form estimators: exploiting independence to scale up Monte Carlo. arXiv:2102.11575v3; Statistics and Computing. 核对 §1–2 产品经验测度、广义 U-statistic、方差/成本，以及 Appendix F 的 mixture 扩展。https://arxiv.org/html/2102.11575v3

[R4] Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic. arXiv:2603.01162v3. 核对固定 prompt 的 IID output–reward 假设及 U-statistic 设定。https://arxiv.org/html/2603.01162v3

[R5] On Advantage Estimates for Max@K Policy Gradients. arXiv:2606.06080v1. 核对 all-subsets 与 canonical 信号的关系及归一化限制。其 Max@K 目标不等于本文 normalized GRPO K 组目标。https://arxiv.org/html/2606.06080v1

[R6] SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation. arXiv:2608.09271v1. 核对二值目标、三个奖励水平的非保守反例及 clipping 适用边界。https://arxiv.org/html/2608.09271v1

[R7] Yasushi Taga. Generalized U statistics for stratified random samples. 1973. DOI:10.1080/03461238.1973.10414969. 本轮仅取得原始出版记录/索引摘要，正文访问失败。不能据此声称已经排除与本文一般统计部分的直接等价。https://www.tandfonline.com/doi/abs/10.1080/03461238.1973.10414969

[R8] Davis and Recht. What is the objective of reasoning with reinforcement learning? arXiv:2510.13651v1. v3/v4 已核对的 IID 二值权重基础；本轮不重新将其有限组公式归为原创。https://arxiv.org/html/2510.13651v1

[R9] Qwen2.5-1.5B-Instruct 与 Qwen2.5-Coder-1.5B-Instruct 官方模型卡。配置以运行时冻结的 revision 为准。https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct ; https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

[R10] Chen et al. Optimal sampling for design-based estimators of regression models. 2022. 本轮正文访问受限，仅核对索引中的标题和影响函数抽样设计研究对象；不将一般“按影响函数设计采样”归为本项目发明。https://pmc.ncbi.nlm.nih.gov/articles/PMC8918008/

本轮未审查所有相关统计学文献的完整证明，也没有发现可以保证论文优先权的证据。正式写作时应按上述具体重合点引用，不能把“首次完整统一”作为无法反驳的宽泛主张。

## 总体判断

这份 v4 已经是一份质量较高的**理论审计、假设合同与首轮实验方案**。我逐式检查后，没有发现主二值推导中的致命代数错误：共享与独立评分的平均更新公式、协方差分解、二值 \(K=2/3\) 退化、RLOO 平均不变性、固定候选银行的 Poisson-binomial 积分，以及连续奖励 \(K=2\) 反例，在文档列明的假设下都基本成立。

但按当前形态直接作为 ICLR 论文，我预计多数审稿人会落在 **weak reject 附近**。主要原因有四个：

1. 核心二值权重公式与现有 GRPO 目标分析高度相邻；
2. 当前 RLOO 唯一性定理限制过强，难以独立承担论文中心；
3. 代码测试 pilot 与最有价值的实际场景没有完全对齐；
4. 目前没有任何真实模型梯度、局部更新或在线训练结果。文档也明确承认，现有证据仅包括数学推导、文献核对和有限 CPU 检查。 

我对当前版本的主观评分如下，分数反映论文准备度，不代表录用概率：

| 维度            |   当前水平 | 判断                                    |
| ------------- | -----: | ------------------------------------- |
| 问题重要性         |   8/10 | 真实影响 GRPO、rubric RL、LLM judge 和随机验证流程 |
| 数学准确性         | 8.5/10 | 主结论基本正确，假设写得较严谨                       |
| 理论完整性         |   6/10 | 二值部分较完整，一般奖励和全局刻画不足                   |
| 当前 novelty    |   5/10 | 有清楚增量，理论中心仍偏窄                         |
| 实验协议          | 7.5/10 | 固定银行、双数据银行、小步验证设计很好                   |
| 当前实证证据        |   1/10 | 尚无真实模型证据                              |
| ICLR solid 程度 |   4/10 | 可以发展成论文，目前仍是优秀研究计划                    |

---

# 一、数学准确性：主公式成立，但论文中必须始终保留限定词

## 1. 二值共享与独立评分公式是正确的

在 actor 回答 IID、验证器配置向量与全部回答独立、奖励二值、配置分布固定、原始 on-policy score、无 clipping/KL 的条件下，

$$
G_{\mathrm S}
=
\mathbb E_U[w_{K,\varepsilon}(p_U)g_U],
\qquad
G_{\mathrm I}
=
w_{K,\varepsilon}(\bar p)\bar g
$$

的推导成立。条件于共享配置 \(U=u\) 后，伙伴奖励计数服从二项分布，自身奖励与伙伴计数独立，从而得到标量权重 \(w(p_u)\)。独立配置下，扩展样本 \((Y_i,U_i,\xi_i)\) IID，因此使用总体通过率 \(\bar p\)。文档对这部分的推导是闭合的。

不过，这个数学工具本身已经很接近 Davis–Recht 的条件线性权重和变换目标分析。他们已经证明，二值奖励下不同 advantage 权重会诱导不同的 \(h(p)\) 目标，并给出了 Bernstein 表示。([arXiv][1])

因此可以申报的增量应当是：

> 当单条回答的完整反馈通道保持相同，只改变组内验证器随机性的联合分布时，标准化 advantage 的平均条件信用仍可能发生变化。

不要把 \(w(p)\)、Bernstein 权重或“GRPO 优化变换后的成功率”列为主要原创贡献。文档自身已经意识到这一点。

## 2. 协方差分解正确，但“协方差非零”不等于行为方向变化

式

$$
G_{\mathrm S}
=
\bar w\bar g+\operatorname{Cov}_U(w(p_U),g_U)
$$

以及相对于 \(G_{\mathrm I}\) 的缩放残差是正确的。文档也正确指出：

* 协方差非零可能仍然共线；
* 反向共线不能被普通 cosine 识别；
* 各 prompt 内都只是正缩放，跨 prompt 聚合后仍会改变总体方向。

这部分逻辑严谨。

论文中应把“信用改变”分成三个可测层次：

$$
\begin{aligned}
&\text{组内候选相对信用变化},\\
&\text{prompt 间权重变化},\\
&\text{固定功能空间或目标空间中的方向变化}.
\end{aligned}
$$

目前文档已经提出这些区分，但主 theorem 和实验终点还没有完全围绕它们组织起来。

## 3. RLOO 不变性正确，但只是不变的平均更新

在当前分支中，回答 IID 且验证配置外生，因此伙伴奖励与自身 score 独立，RLOO 的 cross-baseline 项平均为零：

$$
G_R=\mathbb E[\bar q(Y)S(Y)]=\bar g.
$$

二值 \(K=2/3\) 时，z-score advantage 又逐样本等于 RLOO 的固定倍数，所以其固定银行期望只由逐候选边际决定。这些结论正确。

但必须在所有摘要、图注和 theorem 名称中写成：

> **mean-update invariance under exogenous verifier coupling**

不能简写成“RLOO 对关联不敏感”。以下对象仍可能变化：

* 单步梯度方差；
* 梯度的高阶矩；
* clipping 触发概率；
* Adam 状态；
* 有限步训练轨迹；
* 共享随机数产生的方差削减或结构化噪声。

这正是后续实验需要拆开的另一半问题。

## 4. 当前唯一性定理没有错误，但覆盖范围太窄

现有定理要求：

* 奖励二值；
* advantage 置换等变；
* 每条 advantage 只依赖自身奖励和成功计数；
* 逐组零和；
* advantage 表固定；
* 对所有有限策略和外生配置都要求 S/I 平均更新相等。

在这个类内推出 advantage 必须是常数倍 RLOO，证明是成立的。

问题在于，“只依赖自身奖励和成功计数”已经把函数类压缩得非常厉害。审稿人很可能认为：

1. 先将 IID 更新写成一个 Bernstein 多项式；
2. 普遍不变性迫使该多项式常数；
3. 零和条件递推得到 RLOO。

这是一个清楚的 lemma，却不太像整篇论文的主定理。

## 5. 连续奖励部分没有代数错误，但理论层次明显低于二值部分

连续 rubric 部分目前主要提供：

* 固定候选银行中的协方差恒等式；
* epsilon 对尺度消去的破坏；
* 连续奖励在 \(K=2\) 即可发生差异；
* 有限离散奖励的 \((T,Q)\) 动态规划。

这些内容是正确而实用的。

但如果论文最终把 Rubric Dropout 或 LLM judge 作为主要应用，那么最重要的实证场景是连续或多值奖励，而最完整的理论仍停留在二值奖励。这会造成明显的不平衡：

* theorem 研究二值 Bernoulli；
* headline 强调随机 rubric；
* 主实验使用多 criterion 连续分数；
* 理论只能在固定银行上精确积分，缺少总体结构结论。

这个缺口需要在 GPU 实验之前补掉。

---

# 二、novelty 审查：方向仍有空间，但“首次研究验证器关联”已经不能成立

## 1. 最重要的遗漏是 *An Imperfect Verifier is Good Enough*

这篇 2026 年工作已经把每组代码回答与测试结果写成 \(G\times T\) 二值矩阵，并比较了四类噪声：

* 每个回答、每个测试独立翻转；
* 每个回答整行翻转；
* 整组共享某个测试列翻转；
* 整组共享整个矩阵翻转。

从其构造可以直接推出：

* sample × test 与 group × test 对任意单个回答具有相同的测试结果分布，跨回答耦合不同；
* sample × rollout 与 group × rollout 对任意单个回答也具有相同的整行翻转分布，跨回答耦合不同。

所以它已经进行了**同单样本噪声规律、不同组内关联结构**的在线 GRPO 比较，只是没有提供你这里的固定银行平均梯度理论。([arXiv][2])

该工作还报告，在它的对称翻转设置中，group-level 噪声与 sample-level 噪声整体表现接近，group-level 略好；主要模型包括 Qwen3 8B、GLM4 9B，并有 Llama 3.1 8B 和 Qwen3 4B 消融。([arXiv][2])

这不会做掉你的论文，但需要重新定位：

| 不能再写的表述        | 可以写的表述                         |
| -------------- | ------------------------------ |
| 首次研究验证器噪声的组内关联 | 首次刻画固定逐回答反馈通道下，组内耦合如何改变标准化平均信用 |
| 首次比较共享与独立验证噪声  | 首次提供精确固定银行干预和平均更新预测            |
| 共享噪声必然更坏       | 关联效应由配置条件信用与归一化权重共同决定          |
| 现有工作只研究独立噪声    | 现有工作已有多种关联噪声实证，但缺少边际等价下的理论识别   |

更好的做法是把该论文变成一个外部验证对象：使用它的四种噪声结构，检查你的理论能否解释“为什么对称翻转只有较小差异，而构造的条件噪声可以出现方向反转”。

## 2. Rubric Dropout 是最值得正面切入的最近邻

Rubric Dropout 在每个 GRPO group 内共享同一个 rubric mask，并明确认为每个 rollout 使用不同 mask 会使组内比较失去意义。它进一步声称，mask 平均后 advantage 仅发生全局缩放。([arXiv][3])

但它的附录同时承认，期望保持分析发生在除以组标准差之前，因为组标准差本身也依赖 mask。([arXiv][3])

这正是你当前理论的最佳切口：

> Rubric Dropout 对标准化前 centered reward 的分析成立；标准化后的平均 policy update 还包含 mask 与逆标准差之间的耦合项。

这是一个具体、及时、可以实证验证的理论缺口。它比“随机单测试可能改变 GRPO”更有论文价值，因为：

* 已有实际方法和正向训练结果；
* 共享 mask 是真实设计决定；
* 所有 criterion verdict 通常一次 judge 调用即可获得；
* 可以在同一候选、同一 judge verdict 上精确重放 shared/independent mask；
* 可以直接检验已发表机制解释中的遗漏项。

## 3. 泛泛的“组构成决定 advantage 幅度”也已经拥挤

2026 年 9 月 3 日发布的 *Spurious Advantage Hidden in GRPO* 已经强调，二值 GRPO advantage 的幅度完全由组内正负样本计数决定，并提出 composition-free 的 SignBalance。([arXiv][4])

因此这些主张不足以支撑 novelty：

* 稀有正样本得到更大权重；
* GRPO 标准差导致组构成重加权；
* reward count 影响学习强度；
* 单看 reward sign 不足以理解更新。

你的论文必须进一步回答：

> 在单样本反馈规律完全固定时，哪些联合关联会改变行为选择；哪些 advantage 形式能够排除这种变化？

## 4. 方差理论和噪声校正也已有直接邻居

GRPO 已经被刻画为 U-statistic，并有有限样本 MSE、渐近分布和 group-size 分析。([arXiv][5])

Noise-corrected GRPO 也已经在 Bernoulli 标签噪声模型下提出无偏校正，并在数学和代码任务上验证。([arXiv][6])

所以你的 LP、固定银行积分和校准修正需要定位为：

* 因果识别工具；
* 精确机制审计；
* 对现有 shared rubric/noisy verifier 流程的分析；
* 可能的方法推论。

不要把它们包装成“首次解决 noisy GRPO”。

## 5. 当前各项贡献的 novelty 强度

| 当前内容                               | novelty 评价 | 建议                    |
| ---------------------------------- | ---------- | --------------------- |
| 二值 \(w(p)\) 与变换目标                  | 低          | 作为预备 lemma            |
| \(G_S\) 与 \(G_I\) 的条件配置公式          | 中          | 保留，但不能独立担当主定理         |
| 构造方向反转                             | 中低         | 作为可能性和边界反例            |
| 二值 \(K=2/3\) 退化                    | 低          | 已在 v3，作为 sanity check |
| 受限函数类中的 RLOO 唯一性                   | 中          | 扩展到一般奖励与一般 advantage  |
| 固定候选、同边际、精确重放                      | 中高         | 实验方法贡献                |
| Rubric Dropout 标准化后均值审计            | 高潜力        | 最适合作为主要实际问题           |
| 一般 coupling-invariant advantage 刻画 | 高潜力        | 最适合作为理论中心             |

---

# 三、最值得补的理论：一般奖励下的关联不变 advantage 完整刻画

当前论文最需要的不是再增加一个反例，而是把 RLOO 唯一性升级成一个真正统一的结构定理。

## 1. 建议的主定理

先考虑有限奖励空间 \(\mathcal R\)，固定组大小 \(K\)。令

$$
A:\mathcal R^K\to\mathbb R^K
$$

满足：

1. 对样本置换等变；
2. 对每个奖励向量逐点零和：

   $$
   \sum_i A_i(r_{1:K})=0;
   $$
3. 对所有具有相同一维边际分布的联合律 \(Q,Q'\)，都有

   $$
   \mathbb E_Q[A_i(R_{1:K})]
   =
   \mathbb E_{Q'}[A_i(R_{1:K})].
   $$

那么应当可以证明，存在单变量函数 \(f:\mathcal R\to\mathbb R\)，使得

$$
\boxed{
A_i(r_{1:K})
=
f(r_i)
-
\frac{1}{K-1}\sum_{j\ne i}f(r_j).
}
$$

反方向显然成立，因为右侧期望只由各坐标的一维边际决定。

这个结论比当前二值 count-based theorem 强很多：

* 允许二值、多值、连续离散化 rubric；
* 不预先假设 advantage 只依赖成功计数；
* 直接刻画“只看边际就足够”的全部对称零和 advantage；
* 二值奖励时，任意 \(f\) 只有两个函数值，自动退化成 scaled RLOO；
* 多值奖励时得到 transformed RLOO。

进一步加入“对所有策略与奖励通道都保持原始平均奖励梯度”：

$$
\mathbb E[f(R)S]
=
c\,\mathbb E[RS],
$$

则 \(f\) 必须是仿射函数：

$$
f(r)=cr+b.
$$

于是可以得到更强的推论：

$$
\boxed{
\text{scaled RLOO 是同时满足关联不变和原始奖励目标保持的唯一对称零和形式。}
}
$$

这才足以成为论文的理论中心。

## 2. 可行的证明路线

有限奖励空间下，可以用纯线性代数完成：

1. 对任意两个坐标和任意四个取值，构造保持一维边际的 \(2\times2\) probability mass swap；
2. 期望不变迫使所有混合矩形差为零：

   $$
   A_i(a,c,\ldots)+A_i(b,d,\ldots)
   -
   A_i(a,d,\ldots)-A_i(b,c,\ldots)=0;
   $$
3. 混合差为零推出 \(A_i\) 对各坐标可加分解；
4. 置换等变使所有伙伴坐标共享同一个单变量函数；
5. 逐点零和固定自身项与伙伴项之间的系数；
6. 得到 transformed RLOO。

底层“固定边际下期望不变意味着可加分解”很可能在 Fréchet 类或多边际耦合文献中已有类似结论。论文应把 novelty 放在 advantage estimator 的完整刻画、GRPO 推论和实证后果上，并对纯数学部分专项查重。

## 3. 加入一个 interaction decomposition

对一般 advantage，可以写成

$$
A_i=A_i^{\mathrm{marg}}+A_i^{\mathrm{int}},
$$

其中：

* \(A_i^{\mathrm{marg}}\) 是各坐标单变量函数之和，其期望由边际决定；
* \(A_i^{\mathrm{int}}\) 是联合交互项，其期望随 coupling 改变。

这样可以统一解释：

* RLOO：\(A^{\mathrm{int}}=0\)；
* 二值 z-score，\(K=2/3\)：交互项退化为零；
* 二值 z-score，\(K\ge4\)：成功计数对应的非线性产生交互；
* 连续 z-score，\(K=2\)：随机分母已经可以产生交互；
* 固定银行 LP：计算给定边际下交互项能达到的 Fréchet 包络。

建议定义一个可报告的关联敏感度：

$$
D_i(\nu_1,\ldots,\nu_K)
=
\sup_{Q,Q'\in\mathcal C(\nu_1,\ldots,\nu_K)}
\left|
\mathbb E_Q A_i-\mathbb E_{Q'} A_i
\right|.
$$

再定义功能空间中的版本：

$$
D_L
=
\sup_{Q,Q'}
\left\|
L\left(
G_Q-G_{Q'}
\right)
\right\|.
$$

附录 A 的 LP 已经接近这个对象，只需把它从“可选 CPU 检查”升级为理论定义和有限银行计算方法。

## 4. 增加一般奖励的大组极限定理

当前连续奖励缺少总体公式。可以定义：

$$
\mu_u=\mathbb E[R\mid U=u],
\qquad
\sigma_u^2=\operatorname{Var}(R\mid U=u),
\qquad
g_u=\mathbb E[R\,S(Y)\mid U=u].
$$

在 \(\varepsilon>0\)、有限二阶矩和适当一致可积条件下，随着 \(K\to\infty\)：

$$
\boxed{
G_{\mathrm S}^{(K)}
\longrightarrow
\mathbb E_U
\left[
\frac{g_U}{\sigma_U+\varepsilon}
\right],
}
$$

而独立配置满足

$$
\boxed{
G_{\mathrm I}^{(K)}
\longrightarrow
\frac{\bar g}{\sigma_{\mathrm{mix}}+\varepsilon},
}
$$

其中 \(\sigma_{\mathrm{mix}}^2\) 是混合反馈通道下单样本奖励的总体方差。

这个结论有三个价值：

1. 表明 coupling sensitivity 并不只是小组有限样本现象；
2. 对连续 rubric 直接给出可解释的逆条件标准差重加权；
3. 二值情形自然恢复 \(\sigma_u=\sqrt{p_u(1-p_u)}\) 的大组权重。

它的证明主要由条件大数定律和

$$
\mathbb E[(R-\mu_u)S\mid U=u]
=
\mathbb E[RS\mid U=u]
$$

组成，难度不高，但论文叙事价值很高。

---

# 四、反例的作用需要收缩

当前方向反转反例是正确的，也成功说明平均 TPR/FPR 相同不足以决定平均更新。

但它不能成为论文 headline，原因是：

* 在类别条件噪声模型里，反转要求至少部分配置具有负 Youden 指数；
* 自然代码测试中，一个通过完整测试集的正确程序会通过每一个子测试；
* 当前 uniform-single-test pilot 不具备旧反例中“某些配置系统性拒绝正确答案”的结构；
* 真实实验更可能看到非标量重加权或学习速度差异，而非完整方向反转。

建议将反例定位为：

> 一维反馈边际不能识别平均更新的最强存在性证明。

主要实证问题改成：

> 自然反馈流程中的 coupling defect 有多大，是否能够预测局部功能变化和在线训练差异？

---

# 五、实验计划审查

## 1. 当前 pilot 的优点

当前方案有几项设计非常好：

* 固定同一候选银行；
* 保存每个回答对所有配置的 verdict；
* 对 S/I/RLOO 精确积分；
* confirmation bank 与目标梯度 bank 分离；
* 使用独立目标斜率；
* 检查 \(\eta\) 和 \(\eta/2\)；
* 区分正缩放、反向共线和正交残差；
* 冻结 revision、hash、解析和超时规则；
* 明确 GO、STOP 和 INCONCLUSIVE。

这些设计已经明显强于普通“跑几条训练曲线看最终 benchmark”的实验。 

## 2. uniform-single-test 适合机制检查，不适合承担主实证

当前代码配置是每题均匀抽一个测试，奖励为该单测试是否通过。它确实能形成天然的反馈矩阵，但有三个问题：

### 第一，实际自然性有限

常见代码 RL 奖励通常是：

* 全部测试通过；
* 通过测试比例；
* 公共测试加隐藏测试；
* 编译、运行、安全等多个子奖励组合。

“每步只随机选择一个测试作为唯一奖励”更像理论探针。它可以证明机制存在，却很难独立支持广泛实践结论。

### 第二，与最接近的已有代码实验发生重合

*An Imperfect Verifier is Good Enough* 已经使用 MBPP、unit-test matrix、sample/group 关联结构和在线 GRPO。你的代码实验必须明确增加以下内容，才不会显得是重复：

* 固定同一候选的精确反事实；
* 保持单回答完整奖励分布，而不仅匹配噪声率；
* 实际平均梯度与目标斜率；
* 理论预测局部更新；
* 一般非对称配置；
* RLOO/mean-only 阴性对照。

### 第三，它不会自然命中最强反转机制

文档已经诚实指出，suite-correct 程序必然通过每一个单测试。

因此 pilot 的成功门槛应当是：

* 检出稳定的非标量信用差异；
* 差异能够预测 held-out suite 或功能读出的局部变化；
* 差异超过基线功能变化的预先冻结比例；
* 不要求出现方向反转。

## 3. 32 个 prompt 足够做 pilot，不能支撑主论文结论

当前 C、D 各只有 32 个独立题目。每题增加回答数量可以降低条件于 prompt 的 rollout 方差，却无法充分降低 prompt population 方差。

更合理的路径是：

1. 用 32 题估计 prompt-level effect variance；
2. 根据目标最小效应确定确认集大小；
3. 正式结果使用独立扩大的 prompt 集；
4. 所有区间在 prompt 层聚类；
5. 不把每条回答当成独立样本。

实际所需题数取决于效应大小；若 coupling effect 只有基线功能变化的几个百分点，32 题通常很难给出稳定结论。

## 4. 完整 \(H_{dc}\) 矩阵不适合放进 10 小时 pilot 的硬合同

文档建议保存

$$
H_{dc}=h_d^\top M\Delta_c
$$

来传播 C/D 两边的不确定性。统计逻辑是对的，但全参数梯度下实现成本可能很高：

* 1.5B 参数的单个 FP32 梯度约为数 GB；
* 64 个 prompt 级梯度无法轻易常驻显存；
* 逐对重放 1024 个内积会挤占主要 GPU 预算；
* 大量 CPU 写盘也可能成为瓶颈。

可以直接使用

$$
\widehat T=\bar h^\top M\bar\Delta
$$

并利用独立双样本结构。其方差可分解为

$$
\begin{aligned}
\operatorname{Var}(\widehat T)
={}&
\frac{1}{n_D}
\operatorname{Var}\!\left(h^\top M\mu_\Delta\right)
+
\frac{1}{n_C}
\operatorname{Var}\!\left(\mu_h^\top M\Delta\right)\\
&+
\frac{1}{n_Dn_C}
\mathbb E
\left[
\big((h-\mu_h)^\top M(\Delta-\mu_\Delta)\big)^2
\right].
\end{aligned}
$$

前两项可以通过一个对侧聚合向量得到 prompt 级标量；最后一项可用预先固定的少量 C–D 配对、随机投影或独立子样本估计。完整 \(H\) 可以放到确认阶段。

首轮应优先保证：

* 固定功能读出；
* 聚合目标斜率；
* \(\eta/\eta/2\) 局部预测；
* reward marginal 和 coupling contract。

## 5. 参数空间 cosine 不应成为主要结果

即使使用同一个参数化，参数梯度内积仍受：

* 参数尺度；
* 层宽；
* LoRA/full parameter 子空间；
* 预条件映射 \(M\)；
* normalization 参数化

影响。

因此主要结果应按优先级排列为：

1. 独立目标的一阶斜率；
2. 冻结功能读出的变化；
3. 实际小步后的功能变化；
4. 正缩放残差；
5. 参数 cosine 作为诊断。

当前文档已经接近这个顺序，应在论文结构中贯彻。

---

# 六、主实验应转向 Rubric Dropout 的因果审计

这是当前 ROI 最高的实证主线。

## 1. 固定银行阶段

对每个 prompt 生成一组回答，让同一个 judge 一次返回所有 criterion verdict：

$$
B_{ik}\in\{0,1\},
$$

然后在相同 \(B\) 上重放：

* group-shared mask；
* per-response independent mask；
* full rubric；
* z-score advantage；
* RLOO 或 mean-only advantage；
* 条件期望 advantage。

Rubric Dropout 本身已经说明，一个 judge 调用会返回全部 criterion verdict，因此重放不同 mask 通常不增加 judge 调用。([arXiv][3])

## 2. 最关键的 factorial

至少需要下面的机制矩阵：

| mask coupling            |      z-score | RLOO / mean-only |
| ------------------------ | -----------: | ---------------: |
| group-shared             |       主方法原设计 |           均值不变参照 |
| per-response independent | coupling 反事实 |         均值不变阴性对照 |

再加入 full-rubric no-dropout 基线。

这个设计能够回答三个独立问题：

### A. 标准化后的平均信用是否改变

比较：

$$
\text{shared z-score}
\quad\text{vs}\quad
\text{independent z-score}.
$$

### B. 训练收益是否来自平均方向变化

比较 z-score 与 RLOO。

若 RLOO 下 shared/independent 的平均更新相同，而 z-score 下不同，说明差异来自归一化产生的平均目标变化。

### C. 训练收益是否仍有纯方差机制

若 shared RLOO 和 independent RLOO 平均相同，却在在线训练中表现不同，则剩余差异主要来自更新方差、高阶矩和优化轨迹。

这能直接分辨 Rubric Dropout 提出的“方差正则化”解释与标准化均值改变。

## 3. 在线阶段不必把全部固定银行 arms 都跑满

考虑预算，可以采用：

### 主模型，三到四个核心 arms，至少三种子

1. full rubric + 原始标准化；
2. shared dropout + 原始标准化；
3. independent dropout + 原始标准化；
4. shared dropout + RLOO。

independent dropout + RLOO 可以先在固定银行和局部步阶段验证，在线只做较少种子或确认性消融。

### 第二模型或第二任务

选择一个较小确认矩阵：

* full rubric；
* shared dropout z-score；
* independent dropout z-score。

第二模型的价值是确认 coupling effect 不依赖单个 checkpoint 的梯度几何。

## 4. 评价必须独立于训练 rubric

Rubric 场景至少需要：

* 训练 proxy judge；
* 更强或跨模型家族的审计 judge；
* OOD evaluation prompts；
* full-rubric 分数；
* per-criterion proxy–audit disagreement；
* 过度声称、单 criterion exploit 或类似行为指标。

Rubric Dropout 已经采用了 proxy/gold 双 judge 和 OOD 评价，因此你的实验可以直接沿用这一评价原则，同时新增 coupling 与 advantage 的因果拆分。([arXiv][3])

---

# 七、论文的最佳理论与实证主线

当前 v4 的研究对象可以重新组织为：

## 核心问题

给定相同的逐回答反馈通道

$$
\mathcal L(R_i\mid Y_i=y),
$$

组内反馈的联合耦合是否会改变 group-normalized policy update？

## 理论贡献

1. 一般条件信用表示；
2. 二值有限组精确公式；
3. 一般奖励下 coupling-invariant advantage 的完整刻画；
4. scaled RLOO 的原始奖励目标唯一性；
5. z-score advantage 的 interaction decomposition；
6. 二值 \(K=2/3\) 与连续 \(K=2\) 的最小边界；
7. 连续奖励的大组极限；
8. 固定边际下的 coupling defect 和 LP 包络。

## 识别贡献

在同一个候选和反馈矩阵上：

* 保持逐回答完整反馈分布；
* 只改变联合耦合；
* 精确积分验证器随机性；
* 测量功能方向和独立目标斜率；
* 验证小步预测。

## 实证贡献

1. 代码测试矩阵作为确定性机制检查；
2. Rubric Dropout 作为主要自然应用；
3. 固定银行效应预测局部更新；
4. 局部预测进一步解释在线训练差异；
5. 第二模型或第二评分机制复现。

## 实用结论

最终结论不应预设 shared 或 independent 哪个更好。可以落在：

* 想严格保持平均反馈目标时，使用 RLOO/mean-only；
* 想引入多目标重加权时，必须显式报告 coupling 所诱导的目标；
* shared randomness 可能同时改变均值和方差；
* 训练收益需要通过 factorial 区分两者。

---

# 八、达到 ICLR 级 solid 论文仍缺什么

| 必需内容        | 当前状态 | 达标要求                                                        |
| ----------- | ---- | ----------------------------------------------------------- |
| 一个足够强的理论中心  | 部分具备 | 一般奖励下完整关联不变刻画                                               |
| 与最近工作的准确边界  | 不完整  | 补入 *An Imperfect Verifier*、Spurious Advantage、U-statistic 等 |
| 自然反馈中的非标量效应 | 无    | rubric 或真实 verifier 固定银行证据                                  |
| 理论预测真实局部后果  | 无    | 独立目标斜率与 \(\eta/\eta/2\) 一致                                  |
| 在线训练后果      | 无    | 多种子 shared/independent/RLOO 对照                              |
| 外部有效性       | 无    | 第二模型、第二领域或第二反馈机制                                            |
| 均值与方差分离     | 仅有计划 | z-score × RLOO 的 factorial                                  |
| 实际方法含义      | 尚不明确 | 给出何时使用 shared、independent、RLOO 的条件                          |
| 可复现合同       | 较完整  | 执行并公开 manifest、矩阵、代码和 claim ledger                          |

最低可接受的 solid package，我建议定为：

1. **一般 coupling-invariant advantage 定理；**
2. **连续奖励大组极限；**
3. **代码固定银行机制验证；**
4. **真实 rubric mask 固定银行验证；**
5. **固定银行方向能够预测独立局部更新；**
6. **一个主模型上的多种子在线训练；**
7. **第二 checkpoint 或第二任务上的确认结果；**
8. **对 Rubric Dropout 机制解释给出清楚结论。**

只有数学反例、DP、CPU 恒等式和一条小模型训练曲线，论文仍会显得像理论观察加 pilot。文档自己对这一点的判断是准确的。

---

# 九、建议的执行优先级

## P0：任何 GPU 之前完成

1. 补齐最近邻文献和 claim ledger；
2. 证明一般奖励的可加分解定理；
3. 推出 scaled RLOO 的目标保持唯一性；
4. 增加连续奖励大组极限；
5. 用 CPU 枚举验证新定理在有限奖励空间中的所有小 \(K\) 情形；
6. 将论文标题和摘要改为“相同边际不足以决定标准化信用”。

## P1：10 小时 pilot

1. 保留代码 test matrix；
2. 删除完整 \(H_{dc}\) 作为硬性要求；
3. 首先测固定功能读出和聚合目标斜率；
4. 同时准备一个小型真实 rubric verdict bank；
5. 对 shared/independent/RLOO 做精确积分；
6. 只有出现稳定的非标量功能差异并通过 \(\eta/\eta/2\) 验证，才释放在线训练预算。

## P2：完整论文实验

1. 主模型运行 full/shared/independent/RLOO 核心 arms；
2. headline 至少三种子；
3. 使用独立 proxy/audit judge；
4. 第二模型或第二任务确认；
5. 将已有 noisy verifier 工作的 paired noise modes 作为外部阴性或弱效应对照；
6. 报告均值方向、方差、局部后果和在线结果之间的对应关系。

---

## 最终研究判断

当前方案没有明显的核心数学错误，研究问题也有现实价值。现有二值公式和构造反例不足以单独形成 ICLR 级理论中心；代码单测试 pilot 也不足以独立形成 ICLR 级实证中心。

最有希望的升级路线是：

$$
\boxed{
\text{一般关联不变 advantage 定理}
+
\text{Rubric Dropout 标准化后因果审计}
+
\text{固定银行对真实训练后果的预测}
}
$$

完成这三部分后，论文会从“严谨的 verifier-coupling 观察”提升为一篇结构完整的机制论文：它给出一般刻画，指出现有方法分析中的具体缺口，并在真实训练流程中验证该缺口是否改变学习。当前文档已经打好了执行协议和边界意识，接下来继续增加人工反例的收益很低，应把主要精力投入一般定理与 rubric 实证。

[1]: https://arxiv.org/html/2510.13651v1 "What is the objective of reasoning with reinforcement learning?"
[2]: https://arxiv.org/html/2604.07666v1 "An Imperfect Verifier is Good Enough: Learning with Noisy Rewards"
[3]: https://arxiv.org/html/2608.11669 "Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL"
[4]: https://arxiv.org/html/2609.04063v1 "Spurious Advantage Hidden in GRPO"
[5]: https://arxiv.org/html/2603.01162v3 "Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic"
[6]: https://arxiv.org/html/2510.18924v3 "Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients"

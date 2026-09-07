# Dependent Feedback v4：验证器关联、组归一化与条件信用的理论审计

**理论复核 / 与 Dependent Rollouts v3 的关系 / Codex 实验合同**  
**日期：2026-09-06**  
**研究状态：没有预训练语言模型实验结果。本文实际完成了文献核对、数学推导和 CPU 有限模型检查。**

> 本文审计上一轮“验证器组内随机性关联改变 GRPO 平均方向”的方案，并以用户附件《Dependent Rollouts v3：条件信用选择、停止局部性与归一化的识别边界》为对照。本文增加验证器这一干预对象，重新安排最低成本实验；不把附件中已经建立的结论再次申报为新贡献。
>
> “证明”指在列明假设下给出的自含推导；不表示机器形式化验证或同行评审。CPU 检查验证有限实例与实现恒等式，不证明真实 LLM 的效应量、训练收益或论文录用可能性。
>
> 第一阶段硬预算：**10 小时 RTX 5090，或 5 小时 Pro 6000**。完整实验硬上限：**200 / 100 小时**。两类硬件分别计量；表中的分配不是吞吐预测，也不假设性能恰好相差两倍。本轮没有运行 GPU、下载模型或调用付费 judge。

---

## 0. 审计结论与研究决定

### 0.1 数学主公式和旧反例成立，但上一轮的研究定位过于乐观

上一轮以下两个式子，在“actor 回答 IID、验证器配置与所有回答独立、二值反馈、固定配置分布、on-policy 原始序列 score、无 clipping/KL”的设定下成立：

\[
G_{\mathrm I}=w_{K,\varepsilon}(\bar p)\nabla\bar p,
\qquad
G_{\mathrm S}=\mathbb E_U[w_{K,\varepsilon}(p_U)\nabla p_U].
\]

旧反例的方向反转也能复现。加入与附件一致的 `epsilon=1e-6` 后仍成立。

但是，**这两个公式中的 IID 二值归一化权重，是 Davis–Recht 等已有分析的直接应用；K=2/3 的退化边界已经出现在附件 v3 中。**单独把它们组织成“新理论中心”，贡献不够。[R1, R2, U1]

更准确的研究定位是：

> **逐回答反馈分布相同，是否足以保证两个验证流程在标准化策略更新中可以互换？如果不能，如何在固定回答上精确隔离这项差异，识别它对实际目标的作用？**

可争取的增量主要来自：明确的验证器联合分布问题、适用与安全边界、真实评分机制中的严格干预，以及能够独立复现的学习后果。当前尚没有实证支持。

### 0.2 逐项修正

下表同时检查上一轮的明确表述与实施时容易出现的外推；后者不是上一轮已经证明或明确宣称的结论。

| 审计对象 | 审计结果 | 本文采用的处理 |
|---|---|---|
| 共享和独立评分的二值平均梯度公式 | 在受限设定下正确 | 补齐条件化证明、epsilon、端点和 score 定义 |
| TPR/FPR 一样仍可能反向学习 | 存在性正确 | 区分逐回答边际一致与仅聚合 TPR/FPR 一致；补齐不会反转的条件 |
| K=2/3 退化边界的原创性 | 附件已有更强结论 | 明确沿用附件 v3，且其逐样本结论更强 |
| “关联不变性唯一要求 RLOO” | 原文证明与量词不完整 | 限定优势函数类，给出完整必要性、充分性与递推证明 |
| 将相关回答下的 RLOO 偏差移用于本分支 | 条件不成立 | IID actor 加外生评分关联下，RLOO 的平均更新保持不变 |
| “独立评分更正确” | 没有普遍依据 | 独立评分只是一个明确参照；真实目标需要另行指定 |
| 参数 cosine 变化足以说明学习损害 | 不成立 | 使用独立目标斜率、固定功能读出、正缩放残差与小步检验 |
| 固定矩阵重采样可直接证明总体预测 | 不成立 | 固定银行的精确反事实、总体均值估计、样本外预测分别报告 |
| 先用 3B judge 建立大量评分配置 | 性价比和判定性不足 | 首轮改用代码测试产生反馈矩阵，避免额外 judge GPU 成本 |
| 将 K=2/3 阴性对照推广到所有奖励 | 不成立 | 仅适用于两个固定奖励取值；连续奖励 K=2 已可不同 |
| 二值结果直接解释 Rubric Dropout | 不成立 | 连续 rubric 使用独立的固定银行分析与离散网格 DP |
| 相同答案/奖励分布意味着相同优化轨迹 | 不成立 | 分开原始均值、方差、Adam、clipping、有限步和训练过程 |
| 当前方向是否已足够支持 solid accept | 尚无实证，不能判断 | 改为有止损条件的研究分支，真实模型首轮通过后再评估 |

### 0.3 具体决定

先推进**验证器关联分支的廉价机制实验**。它可以使用普通 IID 生成，避免首轮实现算术分层 sampler、release mask 和专门的采样引擎。

这与 v3 属于同一研究家族。数学上不能把两者视为互不相关的新方向；实验上，验证器分支更容易固定候选、精确匹配奖励边际并消除评分随机性。

不同时启动 v3 的完整分层训练矩阵与本分支的完整训练矩阵。先用本分支判断：自然反馈是否具有足够强、可解释的平均信用效应。

---

## 1. 与附件 v3 的关系：哪些可以继承，哪些不能搬用

### 1.1 一个统一对象，两个不同的干预位置

附件 v3 研究：

\[
(Y_1,\ldots,Y_K)\sim Q,\qquad Y_i\sim\pi_\theta,
\qquad R_i=r(Y_i).
\]

回答之间相关，奖励是回答的固定函数。其 RLOO 偏差来自伙伴奖励对自身回答的条件依赖。[U1, §1–2]

本分支研究：

\[
Y_1,\ldots,Y_K\stackrel{\mathrm{iid}}\sim\pi_\theta,
\qquad R_i=r(Y_i,U_i,\xi_i),
\]

其中配置向量 \(\boldsymbol U\) 与整个回答向量独立，\(\xi_i\) 为各自私有的评分随机性；改变的是 \(U_i\) 之间的关联。

把扩展对象写成 \(Z_i=(Y_i,U_i,\xi_i)\)，奖励可以重新看成固定函数，score 仍为 \(S(Y_i)\)。这样可以放入 v3 的条件信用框架。

**这项嵌入说明：一般的条件期望表征已经覆盖新分支。**新分支的实际价值是干预更直接、RLOO 可作为更强的阴性对照、实际验证流程具有明确的配置选择。

### 1.2 可继承的内容

- 固定 checkpoint 的原始边际 score；不能把 surrogate 梯度等同于联合 sampler 的梯度。
- 区分计数重加权、正缩放、功能方向、独立目标改善与训练收益。
- 二值 K=2/3 的逐样本缩放，K=4 的计数条件分解。
- 同一批回答上的线性权重干预；固定线性优化器映射；独立数据与不确定性传播。
- v3 的高阶反例可作为统一理论的边界例子。它已经比“奖励相关导致均值变化”更强。[U1, §5.4]

### 1.3 不能直接搬用的内容

**停止局部性。**本分支没有算术层标签，也没有相应 release 时刻；不能声称效应只发生在某个前缀。终端优势作用于整条回答的 score。

**RLOO 的分层偏差。**本分支中回答保持 IID 且验证器随机性外生，伙伴奖励不依赖自身回答，因此这项偏差为零。

**完整奖励向量分布相同的反例。**v3 改变回答身份与计数之间的高阶关系。本分支的共享/独立评分通常会改变完整奖励向量分布，不能声称两项干预拥有同样强的匹配条件。

**完整轨迹的行为保护。**即使某些 token loss 的差异为零，共享参数更新仍可能改变这些 token 的行为；本分支更没有这种保护保证。

### 1.4 本轮对附件实际复核了什么

本轮重跑附件附录 A 的完整 CPU 脚本，得到与附件一致的最大等式误差 `1.5543122344752192e-15`。重点交叉检查了条件信用表征、K=2/3/4 边界、高阶反例、固定池重组和固定几何的小步解释。

本轮没有把附件引用的全部论文重新逐篇审查；只重新核对与本分支直接相关的最近邻。附件的全部文献边界不应因此被标记为“再次全面验证”。

---

## 2. 数学合同：先固定 estimand

### 2.1 策略和 score

固定提示词 \(x\)、checkpoint \(\theta_0\)、tokenizer、生成上限 \(H\) 和解码分布。省略 \(x\)：

\[
S(Y)=\nabla_\theta\log\pi_\theta(Y)\big|_{\theta_0},
\qquad \mathbb E[S]=0.
\tag{1}
\]

要求公共支持、可微性和可交换期望与微分。理论使用原始祖先采样、`temperature=1`，不使用改变支持的 top-k/top-p。

EOS 计入 score；padding 不计入。硬截断保留真实截断轨迹，不补虚构 EOS。解析失败、全同奖励组和超时按照预先冻结的规则处理，不筛掉再补样。

### 2.2 验证器

设配置 \(U\sim\mu\)，私有随机数 \(\xi\sim\nu\)，且与 actor 生成独立。先研究二值奖励：

\[
r(y,u,\xi)\in\{0,1\},\qquad
q_u(y)=\mathbb E_\xi[r(y,u,\xi)].
\]

定义

\[
p_u(\theta)=\mathbb E_{Y\sim\pi_\theta}q_u(Y),
\quad
\bar q(y)=\mathbb E_Uq_U(y),
\quad
\bar p(\theta)=\mathbb E_U p_U(\theta).
\tag{2}
\]

\(q_u\)、\(\mu\)、\(\nu\) 均不随 \(\theta\) 改变，至少在当前求导问题中固定。因此

\[
g_u:=\nabla p_u=\mathbb E[q_u(Y)S(Y)],
\qquad \bar g:=\nabla\bar p=\mathbb E_Ug_U.
\tag{3}
\]

`p_u` 是当前策略在配置 u 下的总体通过率；`q_u(y)` 是固定回答的通过概率。后文有限银行中的 `q_i` 是另一种对象，不能混用。

### 2.3 两种安排

**S：共享。**先抽 \(U\sim\mu\)，所有回答用同一个 U；私有评分随机性 \(\xi_i\) 独立。

**I：独立。**各回答独立抽 \(U_i\sim\mu\)，私有评分随机性也独立。

两者对每个固定回答具有同一完整奖励分布：

\[
\mathcal L(R_i\mid Y_i=y)=\mathrm{Bernoulli}(\bar q(y)).
\tag{4}
\]

任意其他配置关联也必须满足 **\(\boldsymbol U\perp\boldsymbol Y\)**。仅要求每个 \(U_i\) 对自身 \(Y_i\) 独立还不够：如果它依赖别人的回答，cross baseline 仍可能相关。

不覆盖以下实现：judge 一次读取整组回答并比较评分；按已生成的回答选择 rubric；配置由奖励后筛选决定；actor 可以看见 U 并据此生成。这些是不同的联合律。

### 2.4 优势和 loss

令 \(N=\sum_iR_i\)，采用

\[
A_i^R=R_i-\frac1{K-1}\sum_{j\ne i}R_j,
\]

\[
A_i^Z=\frac{R_i-N/K}{\sqrt{(N/K)(1-N/K)}+\varepsilon},
\qquad \varepsilon=10^{-6}.
\tag{5}
\]

全同奖励组的优势显式设为零，组本身仍计入平均。`std` 使用 `ddof=0`，epsilon 位于根号外。

\[
\widehat G=\frac1K\sum_iA_iS(Y_i),
\qquad
\mathcal L=-\frac1K\sum_i\operatorname{stopgrad}(A_i)
\sum_{t\le T_i}\log\pi_\theta(y_{it}\mid y_{i,<t}).
\tag{6}
\]

理论针对这一未裁剪的 on-policy score surrogate。按每条回答长度除、按随机有效 token 总数除、动态 clipping、Adam 状态更新、KL 都需要另报。

---

## 3. 完整复核：二值标准化的有限组公式

### 3.1 先求固定配置下的条件期望

令 \(M=\sum_{j\ne i}R_j\)。固定 u 时，回答和私有评分独立，因此

\[
M\sim\mathrm{Binomial}(K-1,p_u),
\]

并且 M 与自身 \((Y_i,\xi_i)\) 独立。

定义非负系数

\[
b_m^+=\frac{K-1-m}{\sqrt{(m+1)(K-1-m)}+K\varepsilon},
\]

\[
b_m^-=\frac{m}{\sqrt{m(K-m)}+K\varepsilon}.
\tag{7}
\]

\(\varepsilon=0\) 时遇到 `0/0` 的端点按同分组约定取零。

当自身奖励为 1 时，\(A_i^Z=b_M^+\)；为 0 时，\(A_i^Z=-b_M^-\)。所以

\[
\mathbb E[A_i^ZS_i\mid U=u,M=m]
=b_m^+\mathbb E[R_iS_i\mid u]
-b_m^-\mathbb E[(1-R_i)S_i\mid u].
\]

因为 \(\mathbb E[S_i\mid u]=0\)，后一个期望是 \(-g_u\)，故上式等于

\[
(b_m^++b_m^-)g_u.
\]

令

\[
\phi_m=b_m^++b_m^-,
\]

\[
w_{K,\varepsilon}(p)=\sum_{m=0}^{K-1}
{K-1\choose m}p^m(1-p)^{K-1-m}\phi_m.
\tag{8}
\]

由交换性，组平均期望不额外产生一个 K 因子，得到

\[
\boxed{G_u=w_{K,\varepsilon}(p_u)g_u.}
\tag{9}
\]

这是已有 IID 条件线性优势分析在本记号下的具体形式。[R1, R2]

### 3.2 对配置积分

共享安排先对 U 条件化，直接得到

\[
\boxed{G_{\mathrm S}=\mathbb E_U[w(p_U)g_U].}
\tag{10}
\]

独立安排中的 \((Y_i,U_i,\xi_i)\) 本身 IID，单个奖励通过率为 \(\bar p\)，所以

\[
\boxed{G_{\mathrm I}=w(\bar p)\bar g.}
\tag{11}
\]

旧答案中的公式在 \(\varepsilon=0\) 下就是式 (10)–(11)。本轮没有发现其代数错误。

### 3.3 可以写出目标，但不能扩大适用范围

令 \(h'(p)=w(p)\)。在固定 \(\mu\)、固定验证器和上述全局可微设定下，

\[
G_{\mathrm S}=\nabla\mathbb E_U h(p_U),
\qquad
G_{\mathrm I}=\nabla h(\mathbb E_U p_U).
\tag{12}
\]

这两个目标不同，但不能仅由 Jensen 不等式推断哪个更有益。h 的曲率、配置梯度与评价目标都必须纳入。

如果每一步按当前模型重选 \(\mu\)，式 (10)–(11) 可以作为“当前配置被冻结”的局部均值；式 (12) 的全局固定目标解释可能失效。

二值结论不能自动扩展到三个及以上奖励取值；本文后续对连续奖励只使用可靠的固定银行等式。

---

## 4. 差异什么时候出现，什么时候只是缩放

### 4.1 协方差分解

令 \(\bar w=\mathbb E_Uw(p_U)\)，则

\[
\boxed{
G_{\mathrm S}=\bar w\bar g+
\mathbb E_U[(w(p_U)-\bar w)(g_U-\bar g)].
}
\tag{13}
\]

因而

\[
G_{\mathrm S}-\frac{\bar w}{w(\bar p)}G_{\mathrm I}
=\operatorname{Cov}_U(w(p_U),g_U).
\tag{14}
\]

右侧是标量与向量的协方差。它非零仍可能只改变标量倍数，也可能产生反向共线；不能把“协方差非零”直接解释为方向旋转。

固定功能读出 F、固定线性更新映射 M，设

\[
L=D F(\theta_0)M.
\]

当 \(L\bar g\ne0\) 时，正交方向变化是

\[
\Pi_{L\bar g}^{\perp}LG_{\mathrm S}
=\mathbb E[(w(p_U)-\bar w)\Pi_{L\bar g}^{\perp}Lg_U].
\tag{15}
\]

同时必须报告沿 \(L\bar g\) 的系数，以免遗漏反向共线。

### 4.2 足以让方向差异消失的条件

| 条件 | 可以推出的结论 |
|---|---|
| 所有 \(p_u\) 相同 | S 与 I 的总体平均更新相同 |
| 所有 \(w(p_u)\) 相同 | S 与 I 是同 prompt 下的正比例关系 |
| \(p_u\) 只取 a、1−a | 利用 w 的对称性，仍为正比例；通过率方差可以很大 |
| 全部 \(g_u=a_u v\)，且 \(a_u\ge0\) | 两种更新都沿 v 的非负方向 |
| reward 配置改变但 \(g_u\) 在被测 score/功能空间中不可见 | 这些读出可能没有差异 |
| K=2 或 3，固定二值奖励 | S 与 I 的平均优势在每个固定候选银行上就完全相同 |

这些是充分条件，不能反向推断其必要性。

### 4.3 差异上界

对任意固定线性 L，

\[
\|\Pi_{L\bar g}^{\perp}LG_{\mathrm S}\|
\le \sqrt{\operatorname{Var}(w(p_U))}
\sqrt{\mathbb E\|\Pi_{L\bar g}^{\perp}Lg_U\|^2}.
\tag{16}
\]

这是 Cauchy–Schwarz 的直接推论，只提供上界。通过率异质性很大并不保证方向效应很大。

### 4.4 跨 prompt 重加权必须单独报告

每题都只有正缩放 \(G_{\mathrm S}(x)=a(x)G_{\mathrm I}(x)\)，仍可能改变聚合梯度。

因此需要区分：

1. 同题内出现不同的行为信用；
2. 各题仍共线，但题目权重改变；
3. 两者共同存在。

第一阶段只有每题两个组，通常不足以精确识别每题的总体方向。此时可以报告随机 prompt 分布下的聚合均值变化，不能直接声称已证明同题内非标量选择。

---

## 5. RLOO 与不变性边界

### 5.1 在本分支中，RLOO 平均更新不依赖评分关联

对 \(j\ne i\)，因为 \(Y_i\) 与 \((Y_j,\boldsymbol U,\xi_j)\) 独立，

\[
\mathbb E[R_jS(Y_i)]=0.
\]

于是对任意外生配置关联，

\[
\boxed{G_R=\mathbb E[\bar q(Y)S(Y)]=\bar g.}
\tag{17}
\]

这不与 v3 冲突：v3 的回答之间存在相关性，刚才的独立性不成立。

更强的固定银行事实是：对固定候选及任意奖励联合分布，只要每个候选的奖励均值 q_i 相同，

\[
\mathbb E[A_i^R]=q_i-\frac1{K-1}\sum_{j\ne i}q_j.
\tag{18}
\]

式 (18) 不需要 actor IID，因为候选已经固定；但从固定银行升到 \(\bar g\) 需要式 (17) 的总体假设。

### 5.2 K=2/3 是逐样本退化，沿用 v3

对任何二值奖励向量：

\[
A_i^Z=\frac{K-1}{\sqrt{N(K-N)}+K\varepsilon}A_i^R.
\tag{19}
\]

所以

\[
K=2:\quad A^Z=\frac1{1+2\varepsilon}A^R,
\]

\[
K=3:\quad A^Z=\frac2{\sqrt2+3\varepsilon}A^R.
\tag{20}
\]

由式 (18)，这两个 K 的评分关联不变性在**固定候选的期望优势**层面就成立，不只是总体梯度相同。

K=4 才首次允许不同混合计数拥有不同标准差。附件 v3 的 H₂ 分解保持有效，但其代数验证不算新的实证发现。[U1, §5]

### 5.3 完整的受限唯一性结论

考虑如下优势函数类：

- 奖励为二值；
- 对样本置换等变；
- 每条优势只依赖自身奖励和组成功数；
- 每组优势和为零；
- 全成功或全失败组优势为零。

本命题中，这张优势表在固定 K 后保持不变，不根据策略概率、评分配置 U、回答身份或当前数据重新设定。

设组内有 n 个成功时，成功回答优势为 \(\alpha_n\)，失败回答为 \(\beta_n\)。零和条件为

\[
n\alpha_n+(K-n)\beta_n=0,
\quad \alpha_K=\beta_0=0.
\tag{21}
\]

IID 平均更新系数的 Bernstein 系数为

\[
\phi_m=\alpha_{m+1}-\beta_m.
\tag{22}
\]

**命题。**在此函数类内，要求对所有有限策略、所有外生隐藏评分配置，S 与 I 的平均更新始终相同，当且仅当优势是常数倍 RLOO：

\[
A_i=c\left(R_i-\frac1{K-1}\sum_{j\ne i}R_j\right).
\tag{23}
\]

c 可以为零或负数；若另要求保持奖励上升方向，需要 c>0。

**必要性证明。**取 Bernoulli actor，成功概率为 p。配置 1 的奖励等于 actor 的二值输出；配置 0 永远返回零。以概率 λ 选择配置 1。

共享均值为 \(\lambda w(p)\nabla p\)，独立均值为 \(\lambda w(\lambda p)\nabla p\)。普遍不变性要求

\[
w(p)=w(\lambda p),\quad 0<p<1,\ 0<\lambda<1.
\]

所以 w 在 (0,1) 上为常数 c，由连续性扩展至端点。Bernstein 多项式基线性独立，所以式 (22) 中所有 \(\phi_m=c\)。结合式 (21)，从 \(\beta_0=0\) 递推得到

\[
\alpha_n=c\frac{K-n}{K-1},\qquad
\beta_n=-c\frac n{K-1},
\]

即式 (23)。

**充分性证明。**由式 (17)，常数倍 RLOO 对任意允许的配置关联都有均值 \(c\bar g\)。□

这是对经典 IID 权重表示与线性均值结构的组合推论。它补齐上一轮的证明，但不应单独被包装为全新的优化算法或关于所有 RL estimator 的不可能定理。

---

## 6. 反向更新：旧反例复核与更清楚的边界

### 6.1 原四回答反例

策略为四类 softmax：

\[
\pi=(0.005,0.005,0.495,0.495),\qquad
r^*=(1,1,0,0),\qquad J^*=0.01.
\]

配置表为：

| 概率 | 回答 1 | 回答 2 | 回答 3 | 回答 4 |
|---:|---:|---:|---:|---:|
| 0.17 | 0 | 0 | 0 | 0 |
| 0.23 | 0 | 0 | 1 | 1 |
| 0.30 | 1 | 1 | 1 | 0 |
| 0.30 | 1 | 1 | 0 | 1 |

逐回答边际奖励为 `(0.60, 0.60, 0.53, 0.53)`，因此 TPR=0.60、FPR=0.53，二者在 S/I 下完全一致。

K=8 的实际复核结果：

| 约定 | \(G_S/\nabla J^*\) | \(G_I/\nabla J^*\) |
|---|---:|---:|
| epsilon=0 | −0.043322127117424954 | +0.13006674006863228 |
| epsilon=1e−6 | −0.04332152301781931 | +0.1300664614394021 |

这里的除号表示两个向量严格共线时的标量系数。脚本另外枚举了全部 \(4^8=65,536\) 个 actor 组，验证共享期望。

在 epsilon=1e−6 时，真实目标的一阶斜率分别约为

\[
(\nabla J^*)^\top G_S=-4.24594\times10^{-6},
\qquad
(\nabla J^*)^\top G_I=1.27478\times10^{-5}.
\]

**它只证明这个固定点附近的局部方向。**不能直接声称整条训练轨迹坍缩，也不能把很小的绝对斜率当成实用量级证据。

### 6.2 可化成更简单的二类、条件噪声反例

令 actor 只有“正确/错误”两类，正确率为 J。三个配置：

| 配置概率 | 正确回答通过率 t_u | 错误回答通过率 f_u |
|---:|---:|---:|
| 0.17 | 0 | 0 |
| 0.23 | 0 | 1 |
| 0.60 | 1 | 1/2 |

最后一个配置对错误回答使用独立私有 Bernoulli 随机数。则仍有平均 TPR=0.60、FPR=0.53，并且

\[
G_S=\left[0.30w((1+J)/2)-0.23w(1-J)\right]\nabla J,
\]

\[
G_I=0.07w(0.53+0.07J)\nabla J.
\tag{24}
\]

在 J=0.01、K=8 时得到相同系数。这说明反转不依赖四类 softmax 的特殊参数几何。

它仍使用一个明显反相关的配置，以及较弱的平均验证器。不能隐藏这些限制。

### 6.3 条件噪声模型中的完整符号判据

进一步假设：每个配置内，同一真值类别的所有回答具有相同通过率。令

\[
a_u=t_u-f_u,\qquad p_u=f_u+a_uJ.
\]

则

\[
\boxed{G_S=c_S\nabla J,
\quad c_S=\mathbb E_U[w(f_U+a_UJ)a_U],}
\]

\[
\boxed{G_I=c_I\nabla J,
\quad c_I=w(\bar f+\bar aJ)\bar a.}
\tag{25}
\]

这才是该模型中共享配置的精确符号条件。平均 Youden 指数 \(\bar a>0\) 只保证 I 的系数为正，不能单独保证 S。

若**每一个配置**都满足 \(a_u\ge0\)，则 \(c_S\ge0\)。因此，在此条件噪声模型内，反转需要某些配置本身反相关。普通“共享一个好验证器”不自动满足旧反例机制。

更一般地，对任意评价目标 T，如果所有 \((\nabla T)^\top g_u\ge0\)，则共享更新在 T 方向上的斜率非负，因为 w 始终为正。

### 6.4 一个实用的排除证书

令

\[
m_K=\min_m\phi_m,\qquad M_K=\max_m\phi_m.
\]

Bernstein 基非负且和为一，因此

\[
0<m_K\le w(p)\le M_K.
\]

又因 \(a_u\in[-1,1]\)，写 \(a=a_+-a_-\)，可得

\[
c_S\ge m_K\mathbb E a_+-M_K\mathbb E a_-
\ge\frac{m_K+M_K}{2}\bar a-\frac{M_K-m_K}{2}.
\tag{26}
\]

所以

\[
\boxed{\bar a>\frac{M_K-m_K}{M_K+m_K}\quad\Longrightarrow\quad c_S>0.}
\tag{27}
\]

这是**充分安全条件，通常不尖锐**。epsilon=1e−6 时，本轮计算：

| K | 足以保证正方向的平均 Youden 下界 |
|---:|---:|
| 4 | 0.0467456833 |
| 8 | 0.1970778165 |
| 16 | 0.3459765951 |

这些阈值仅适用于式 (25) 的条件噪声模型。真实 LLM judge 可能在同一真值类别内偏好不同回答，不能把上述数字直接当作其安全认证。

### 6.5 一个必须纳入 CPU 阴性的例子

若以概率 η 共享地将所有奖励取补，以概率 1−η 保持真值，则

\[
G_S=(1-2\eta)w(J)\nabla J,
\]

因为 \(w(J)=w(1-J)\)。当 η<1/2 时不会反转。

因此，“让整组共享一次标签翻转”通常无法复现旧反例。不能拿一个并不满足机制条件的合成噪声实验来否定或支持主结论。

### 6.6 TPR/FPR 的另一处边界

在一般回答空间中，即使聚合 TPR>FPR，\(\bar q(y)\) 仍可能在正确类别内、错误类别内依赖回答身份。此时

\[
\nabla\bar p\not\equiv(\mathrm{TPR}-\mathrm{FPR})\nabla J^*.
\]

因此，独立评分也不自动沿真实正确率上升。式 (25) 的类别内常数条件不可省略。

本项目要比较的是**同一个逐回答反馈通道**的两种分组安排。真实目标是否改善，需要另测，不能仅以 TPR/FPR 推断。

---

## 7. 固定候选银行：精确积分评分关联

### 7.1 数据对象

对固定的一组候选 \(y_1,\ldots,y_K\)，建立矩阵

\[
B_{iu}=r(y_i,u),\qquad u=1,\ldots,C,
\]

配置概率为 \(\mu_u\)。候选 score \(S_i\) 固定。

共享评分的期望优势精确为

\[
\bar A_i^S=\sum_u\mu_u A_i(B_{1u},\ldots,B_{Ku}).
\tag{28}
\]

对二值反馈，令

\[
q_i=\sum_u\mu_uB_{iu}.
\]

独立评分时，伙伴计数是 Poisson-binomial，而不是参数 \(K^{-1}\sum_iq_i\) 的 binomial。设其 PMF 为

\[
\rho_{i,m}=[z^m]\prod_{j\ne i}[(1-q_j)+q_jz].
\]

则

\[
\boxed{\bar A_i^I=q_i\sum_m\rho_{i,m}b_m^+
-(1-q_i)\sum_m\rho_{i,m}b_m^-.}
\tag{29}
\]

简单动态卷积总成本为 \(O(K^3)\)，K=8/16 都很便宜，不必枚举 \(2^K\)。附录保留小 K 枚举作为独立实现检查。

固定银行的精确反事实为

\[
\boxed{\Delta G_B=\frac1K\sum_i(\bar A_i^S-\bar A_i^I)S_i.}
\tag{30}
\]

只需改变同一批 token 的 loss 权重，不需要重新生成，也不需要为每个评分配置进行一次反向传播。

### 7.2 两个不可混淆的量

一般有

\[
\mathbb E[A(R)]\ne A(\mathbb E[R]).
\]

所以不能先把每个回答的评分取均值 q_i，再标准化 q 向量，冒充式 (29)。

同样，式 (29) 使用每个固定回答的 q_i；不能把它们替换为一个 prompt 通过率，也不能用同一组观察成功比例替代总体 \(p_u\) 后声称实现了式 (10) 的精确预测。

### 7.3 逐回答边际全部相同，仍可产生信用

取四个等概率配置，矩阵行为候选、列为配置：

```text
1 0 0 1
0 1 0 1
0 0 1 1
0 0 1 1
```

每一行均值都为 1/2。K=4、epsilon=1e−6 时：

```text
shared expected advantage:
[+0.03867496793, +0.03867496793, -0.03867496793, -0.03867496793]

independent expected advantage:
[0, 0, 0, 0]
```

RLOO 的期望优势也为零。这是一个简洁的固定银行识别例子，表明组标准化可以从配置关联中产生额外信用。

若把四个候选视为均匀四类 softmax actor 的完整支持，再对 actor 抽样积分，独立梯度仍为零，共享梯度约为

```text
[+0.00090644456, +0.00090644456, -0.00090644456, -0.00090644456]
```

固定银行数值和总体数值不同，不能互换。

### 7.4 积分降低的是哪部分方差

对原始随机梯度 \(\widehat G\)，以候选银行 B 为条件：

\[
\operatorname{Cov}(\widehat G)
=\operatorname{Cov}(\mathbb E[\widehat G\mid B])
+\mathbb E[\operatorname{Cov}(\widehat G\mid B)].
\tag{31}
\]

使用式 (28)–(29) 消除了给定候选时的评分配置随机性，保留原算法平均更新。候选采样和 prompt 采样的方差仍存在。

S 与 I 各自积分后，跨候选银行的方差不一定相同。不能称这已经实现了完全相同噪声的均值干预。

这是经典条件期望降方差，不独立申报为新算法。

### 7.5 随机 judge 的缓存限定

代码测试可以把 B 做成确定性矩阵。LLM judge 如果存在私有推理随机性，单次缓存得到的是一个实现后的随机银行。

给定该银行，重排配置的计算仍精确；但它只对这个实现后的评分通道精确。向 live judge 推广，需要说明哪些随机性被固定，哪些通过独立银行重复积分。

逐回答单独评分；judge 输入不得包含伙伴回答。仅复用相同 seed 不足以证明实验实现了本文想研究的配置通道。

---

## 8. 连续 rubric：不能继续使用二值公式

### 8.1 直接与 Rubric Dropout 对照

该工作采用组内共享 rubric mask。其附录给出了共同奖励分母的消去，以及标准化前期望与方差的公式，同时明确指出组标准差依赖 mask。[R3]

需要区分：

1. 共同的正标量奖励分母可以在无 epsilon 的标准化中消去；
2. 对随机 mask 先取期望，再除随机标准差，不能交换；
3. 该论文观察到的训练收益不能被我们的反例否定；其机制解释也不能作为归一化后均值保持的证明。

独立 mask 的价值需要按目标评估。首先声明想优化的是 full rubric、平均子 rubric，还是另一种效用。

### 8.2 一个可靠的固定银行协方差公式

设连续奖励列为 \(R(u)\)，定义

\[
b_i(u)=R_i(u)-\frac1K\sum_jR_j(u),\quad
D(u)=\operatorname{std}(R(u))+\varepsilon,
\quad V(u)=D(u)^{-1}.
\]

在 epsilon>0 时所有量有限；同分列的 b 为零。对候选积分有

\[
\boxed{\mathbb E_UA_i^Z
=\mathbb E[V]\mathbb E[b_i]+\operatorname{Cov}(V,b_i).}
\tag{32}
\]

这是固定银行等式，对实数奖励成立。它精确指出为什么标准化前的均值结论不足以推出标准化后的均值结论。

epsilon=0 时，同分列对优势贡献为零，但逆标准差本身未定义；必须单独处理这些列，不能直接把式 (32) 原样使用。

### 8.3 共同尺度消去的 epsilon 边界

若同一组奖励为 \(a(u)r_i+b(u)\)，且 \(a(u)>0\)，则 epsilon=0 时标准化优势不变。

固定 epsilon>0 时：

\[
\frac{a(r_i-\bar r)}{a\sigma+\varepsilon}
=\frac{r_i-\bar r}{\sigma+\varepsilon/a}.
\]

所以尺度不变性一般不再精确。本项目固定奖励范围和 epsilon；不能混用不同奖励尺度下的数值。

### 8.4 连续奖励在 K=2 就能变化

两种等概率配置给出奖励 `(1,0)` 和 `(0,0.2)`。逐回答奖励分布在 S/I 下相同。

epsilon=0 时，共享安排的期望优势是 `(0,0)`；独立安排为 `(0.25,-0.25)`。

这说明 K=2/3 阴性对照只能用于固定二值奖励。对连续 rubric 观察到 K=2 的差异，不是实现 bug。

### 8.5 离散网格的精确 DP

当每次均匀保留 m 个等权二值 criterion，奖励为

\[
R_i=V_i/m,\qquad V_i\in\{0,\ldots,m\},
\]

独立 mask 下各回答的 \(V_i\) 独立，且分布可由已评分的 criterion 向量计算。

对伙伴维护两个整数统计：

\[
T=\sum_{j\ne i}V_j,\qquad Q=\sum_{j\ne i}V_j^2.
\]

给定自身 v：

\[
A_i=\frac{Kv-(v+T)}
{\sqrt{K(v^2+Q)-(v+T)^2}+Km\varepsilon}.
\tag{33}
\]

若分子根号对应的整数组方差为零，优势设零。动态规划对 `(T,Q)` 积分即可，无需枚举所有 mask 组合。

附录实现并对小例子逐项枚举复核。不能把任意连续奖励先粗糙舍入成网格，再称“精确积分原始算法”。非等权 rubric 使用已知有限支持枚举或带 Monte Carlo 误差的积分，并明确计算对象。

---

## 9. 优化方向：只提出目标明确的修正

### 9.1 最可靠参照：RLOO

如果目标是平均验证器奖励 \(\bar p\)，RLOO 在本分支下具有式 (17) 的平均更新。它应作为基本参照。

这不是新方法，也不保证最优方差、最优训练速度或真实正确率。按 prompt 采用独立固定正尺度还会引入 prompt 权重，需要另报。

### 9.2 对独立评分优势取条件期望

当反馈矩阵本来就可得到时，使用式 (29) 或式 (33) 替代独立抽样得到的优势，可以保持 I 的平均更新并积分评分配置噪声。

它在代码测试、一次判定全部 criterion 后再做 mask 的流程中尤其容易实现。实际成本必须计入获取整张反馈矩阵；如果原流程只验证一个测试，就不能把全部测试或额外 judge 调用视作免费。

这个实现可用于机制审计和稳定性对照。当前不将其命名为新的 GRPO 方法；它是否拥有独立方法贡献，要由文献与实证决定。

### 9.3 二值场景中的 oracle 校正

如果准确知道当前 checkpoint 的 \(p_u\) 和 \(\bar p\)，共享组的优势乘以

\[
c_u=\frac{w(\bar p)}{w(p_u)}
\tag{34}
\]

可得到

\[
\mathbb E_U[c_uG_u]=w(\bar p)\bar g=G_I.
\]

乘以 \(1/w(p_u)\) 则恢复 \(\bar g\)。权重 `detach`，配置和奖励通道仍固定。

它是由已知权重得到的校正，不自动具有方法 novelty。最大的实际困难是逐 prompt、逐配置的通过率校准。

**禁止用当前组的成功率直接代替 p_u 后声称无偏。**估计量与当前梯度相关，且逆权重非线性。

### 9.4 校准误差的明确界

由 Bernstein 导数公式可取

\[
L_K=(K-1)\max_m|\phi_{m+1}-\phi_m|
\]

作为 w 的 Lipschitz 上界。若独立校准满足

\[
|\hat p_u-p_u|\le\delta_u,
\quad |\widehat{\bar p}-\bar p|\le\delta_0,
\]

则

\[
|\hat c_u-c_u|
\le\frac{L_K}{m_K}\delta_0+
\frac{M_KL_K}{m_K^2}\delta_u.
\tag{35}
\]

由此，校正均值误差不超过

\[
\sum_u\mu_u w(p_u)\|g_u\|
\left[\frac{L_K}{m_K}\delta_0+
\frac{M_KL_K}{m_K^2}\delta_u\right].
\tag{36}
\]

该界需要独立校准与有效误差区间。只给 p 的点估计不能声称得到保证。首轮不执行大规模逐题校准；先判断自然矩阵是否有足够效应。

### 9.5 不采用的捷径

不根据确认集的损害最大化选择 seed、mask 或评分配置；不把共享/独立之间的混合比例扫描包装成新优化器；不通过保留出现混合奖励的组来“增强信号”；不以独立评分作为真实目标的自动替身。

---

## 10. 最低成本决定性实验：用代码测试替换首轮 LLM judge

### 10.1 首轮要回答的实际问题

数学上的存在性已经确认。首轮要判断：

> 在真实模型生成的回答、自然测试反馈和预先冻结的配置分布下，评分关联是否产生可分辨的平均更新变化，以及该变化是否影响独立评价目标？

只发现 reward counts 或 norm 改变，不能满足继续做完整论文的门槛。

### 10.2 默认模型和数据

默认 actor 为 `Qwen/Qwen2.5-Coder-1.5B-Instruct`，固定模型和 tokenizer revision。官方模型卡给出 1.54B 参数。[R8]

代码任务使用 MBPP+ 的固定子集与测试输入。EvalPlus 提供扩展测试，用于加强代码结果的可验证性。[R9]

本次更换附件 v3 的 Countdown/SVAMP 首轮，是因为干预对象变为验证器配置，代码测试天然提供同一回答在不同检查下的反馈。附件没有已经完成的真实模型银行，因此不存在本轮必须弃用的真实实验数据。

下载依赖、模型与数据、构建隔离执行器、检查参考实现，应在租用 GPU 之前完成。不得把等待下载计作“免费 GPU 时间”。

### 10.3 测试配置的定义

主配置：对每题预先冻结的一组测试输入，**均匀选择一个测试**，奖励为该测试通过与否。

测试选择只依赖题目和固定 hash，不依赖候选输出，不挑选最容易引起方向反转的测试。

对每个候选运行全部预定测试，在 CPU 上保存二值 verdict 向量。配置集合和概率一旦冻结，S/I 的逐回答奖励分布严格一致。

二级稳健性配置可以在主实验通过后使用“一个固定大小测试子集全部通过”；不能在首轮同时扫描多个子集大小。

### 10.4 三个评价目标必须分开

**平均检查通过率 \(J_{\mathrm{avg}}\)。**即主评分配置的平均奖励。这是理论中 \(\bar p\) 对应的目标。

**预定完整测试集全部通过率 \(J_{\mathrm{suite}}\)。**作为更严格的操作性目标。通过有限测试集不等于对所有可能输入语义正确。

**独立题目上的同类指标。**用于实际泛化和局部更新效果；不能只在计算训练方向的同一候选池上验证收益。

主报告同时给出前两个目标。以随机单测试训练，优化平均测试通过率，与“整个程序完全正确”的目标存在差距；这个差距不能归因给评分关联。

在正确实现下，通过整个预定测试集的程序必然通过其中每一个测试；因此这个实验不会复刻旧反例中“部分配置拒绝全部正确答案”的结构。它用于判断自然反馈中的信用重分配与改善速度，不能把是否出现完整方向反转设为必须命中的结果。

### 10.5 数据银行与默认规模

| 银行 | 默认规模 | 用途 |
|---|---:|---|
| Dev | 16 题，每题 16 回答 | 解析、吞吐、任务可学性、固定读出与步长；不用于正式效果声明 |
| C | 32 题，每题 2 个独立 K=8 组，共 512 回答 | 固定反馈矩阵，精确 S/I/RLOO 权重与 prompt 级估计 |
| D | 32 道独立题，每题 16 回答，共 512 回答 | 独立评价目标梯度、参考及局部更新评估 |

初始总计 1,280 回答，最大新 token 数 384，生成硬上限 491,520 token；另外为小步更新后的评估预留少量生成预算。

题目分割和顺序由公开记录的 hash 冻结。可以预先排除测试执行器不支持的接口或不可确定性环境；不能根据 C 中某种评分安排是否更好进行排除。

吞吐不满足预算时，按固定顺序停止并报告已完成部分。不能缩样后沿用原统计能力声明。

### 10.6 首轮只保留这些 arms

在同一候选组上计算：

- \(\bar A^S\)：共享配置的精确期望优势；
- \(\bar A^I\)：独立配置的精确期望优势；
- \(\bar A^R\)：RLOO 的精确期望优势；
- 预定平均检查/完整测试目标的 score 梯度，用于解释。

K=2/3 仅复用已有候选的固定子组做权重和少量梯度一致性检查，不生成新的 K 网格。K=4 的 H₂ 恒等式保留 CPU 检查；K=16 延后。

不为每种配置重新生成回答。不为每个配置做一次完整 backward。不重新训练 judge。

### 10.7 首轮预算分配

| 工作 | RTX 5090 上限 | Pro 6000 上限 |
|---|---:|---:|
| 采样/评分一致性、显存与吞吐核验 | 0.75 h | 0.40 h |
| Dev/C/D 候选生成 | 2.50 h | 1.20 h |
| 固定银行梯度及独立目标读出 | 4.00 h | 2.00 h |
| 固定几何小步及评估 | 1.75 h | 0.90 h |
| 重跑余量 | 1.00 h | 0.50 h |
| **总计** | **10.00 h** | **5.00 h** |

代码执行与矩阵积分是 CPU 工作，但也应记录墙钟成本。阶段预算是支出上限，不是对硬件速度的承诺。

### 10.8 第二场景的安排

只有代码场景通过后，再加入一次判定多个 criterion 的 rubric 场景。可以先使用程序可验证的指令约束检验机制；IFEval 提供了可程序检查的指令类别。[R10]

程序约束场景不能冒充语义型 LLM judge 证据，也不能把自行组合的约束集称为原始 IFEval benchmark。真正的 rubric 扩展应记录 judge、criterion 原始 verdict、mask 规则与连续奖励定义，使用第 8 节的积分。

不让第二场景成为首轮失败后的替代阳性搜寻。若更换场景，要作为新假设登记并重新评估预算。

---

## 11. 统计与学习后果：怎样避免自己骗自己

### 11.1 首轮主要 estimand

对 C 中 prompt x 的组，计算

\[
\widehat\Delta_x=\frac1{B_x}\sum_b
\frac1K\sum_i(\bar A^S_{xbi}-\bar A^I_{xbi})S_{xbi}.
\]

再按预定 prompt 权重平均。它对给定银行的评分关联积分精确，但依然有 actor 和 prompt 采样误差。

从独立 D 银行得到目标梯度估计 \(\hat g_D\)，报告

\[
\hat T=\hat g_D^\top M\widehat\Delta.
\tag{37}
\]

C 与 D 独立很重要；使用同一批回答估计两个向量的内积会引入相关估计误差。bootstrap 应分别重采样 C 和 D 的 prompt，并传播两边的不确定性。

若 \(\hat g_D\) 本身不可分辨，不能把其归一化方向当作稳定真值。

**双银行内积的可实现统计量。**将每个 C prompt 的差值记为 \(\Delta_c\)，每个 D prompt 的目标梯度记为 \(h_d\)，保存小矩阵

\[
H_{dc}=h_d^\top M\Delta_c.
\]

式 (37) 是 H 的预定加权平均。分别对 D 行和 C 列做 prompt 级 bootstrap，即可传播两边误差，且不会把同一 prompt 的回答当成独立题目。H 可通过参数分块、CPU 临时梯度或受预算约束的重放计算；不要为了得到一个置信区间永久保存每条回答的全参数 score。

若只保存 \(\hat g_D^\top M\Delta_c\)，随后只重采样 c，就得到的是给定估计方向 \(\hat g_D\) 的条件区间，不能将其标记为已传播 D 不确定性的真实目标区间。预算不支持完整 H 时，优先报告 Dev 冻结的功能读出，并将目标斜率标为未充分确认。该成本必须计入第 10.7 节。


### 11.2 功能读出

开发阶段冻结少量可微读出，例如固定独立题目中正确与错误参考回答的 log-likelihood 差。报告

\[
DF(\theta_0)MG_S,\quad DF(\theta_0)MG_I.
\]

这些读出反映所选行为，不完整代表能力。不得按 C 中差异最大的位置选择 token 或方向。

两种解释必须分开：固定读出的一阶方向可被精确测量；它是否代表真实任务收益由式 (37) 和独立生成评估确认。

### 11.3 正缩放处理

在固定功能空间中比较

\[
\min_{a\ge0}\|LG_S-aLG_I\|.
\tag{38}
\]

同时报告有符号的平行系数。只给 cosine 或正交 norm 会遗漏反向共线。

对 norm 零点不能用普通 percentile bootstrap “区间不含零”判断。优先使用开发集冻结的有符号方向或均值向量置信域。

首轮不强行从每题两个组估计高维逐题旋转。可以识别聚合变化；需要同题内结论时，在完整阶段对少量预定题进行独立扩样。

### 11.4 小步更新的合同

主分支使用 SGD，或事先冻结、两臂共同使用的线性 M。每次都从同一个 \(\theta_0\) 开始：

\[
\theta_S^+=\theta_0+\eta M\hat G_S,
\quad
\theta_I^+=\theta_0+\eta M\hat G_I.
\]

在 Dev 冻结 η，然后在 D 检查 η 与 η/2。固定可微读出 F 的预测为

\[
F(\theta_S^+)-F(\theta_I^+)
=\eta DF(\theta_0)M(\hat G_S-\hat G_I)+O(\eta^2).
\tag{39}
\]

没有经过验证的 Hessian 上界时，不能把这个展开变成严格数值保证。步长减半、残差规律和独立评估只能支持所测邻域中的结论。

等范数、等 KL 对照作为补充。它们改变步长，不能取代共同 η 的比较；小 KL 也不自动证明光滑性。

Adam、梯度 clipping、奖励后筛组可能让 \(\mathbb E[f(\hat G)]\ne f(\mathbb E\hat G)\)。完整在线结果不能代替均值机制检查。

### 11.5 如何进一步隔离均值与方差

完整阶段可沿用 v3 的共同噪声干预：用独立银行冻结 \(\hat\Delta_A\)，再从新银行取相同的原始更新 X_b，对比

\[
X_b\quad\text{与}\quad X_b+\hat\Delta_A.
\]

给定 \(\hat\Delta_A\)，两臂的原始噪声协方差相同。这只检验预测均值差的后果，不等价于复现 S 的整个更新分布。

该分支通过首轮后才安排；不在首轮增加多条训练曲线。

---

## 12. 成功、失败与止损

### 12.1 GO：释放下一阶段预算

必须同时满足：

1. 采样、score、奖励边际、DP 与 K=2/3/RLOO 合同通过。
2. 自然代码反馈下出现可分辨的平均信用差异；至少一个预定功能方向或目标斜率有稳定量级。
3. 差异不能仅靠单一全局正尺度解释；并明确剩余证据是 prompt 重加权还是同题行为差异。
4. η、η/2 的局部结果与测得的方向基本相容，且不是少数解析/执行异常造成。

沿用 v3 的**相对已分辨参考功能变化约 10%**作为继续投入的参考门槛，或使用 Dev 预先冻结的绝对量阈值。它不是科学定律。分母不可分辨时禁止报告不稳定百分比。

方向反转不是必须成功标准；否则会迫使实验搜寻极弱验证器。实际可重复的目标改善速度变化也有价值，但应与论文 headline 相符。

### 12.2 STOP：停止该实证主线

- 合同正确且统计精度足够，关键效应的上置信界低于实践门槛。
- 所有效果只有 norm 或全局正缩放，未带来预定目标或行为选择差异。
- 效应仅存在于专门构造的弱 verifier/极低正确率样本，自然设置没有相应证据。
- 真实结果依赖按候选选择配置、删除全同奖励组、按结果选题或使用不匹配的解码概率。
- 新检索发现等价的验证器联合分布结论和实证识别已被完整覆盖。

### 12.3 INCONCLUSIVE：不能把宽区间当成阴性

如果参考近零、吞吐不足、组数不足或执行器失败，结果是不确定。

首轮预算到达即停止。下一步只允许根据实际方差给出一次定量扩样建议；不默认获准继续，也不通过换任务、换 judge 或扫 mask 寻找阳性。

粗略功效计划可使用独立 prompt 级标准差 \(\hat\sigma\) 与预定有符号效应 δ：

\[
n\approx\frac{(1.96+0.84)^2\hat\sigma^2}{\delta^2}.
\]

这是近似规划，不替代正式区间；C/D 双重估计误差、聚类和多重比较要另计。

---

## 13. 完整预算与论文证据顺序

| 阶段 | RTX 5090 | Pro 6000 |
|---|---:|---:|
| 首轮决定性实验 | 10 h | 5 h |
| 独立确认、少量逐题校准与方向拆分 | 25 h | 12 h |
| 固定几何、共同噪声、小步后果 | 20 h | 10 h |
| 主模型在线三臂 × 三种子 | 100 h | 50 h |
| 第二模型家族/第二评分机制复现 | 25 h | 13 h |
| 重跑与误差核验 | 20 h | 10 h |
| **硬上限** | **200 h** | **100 h** |

在线三臂默认是：原生共享配置标准化、原生独立配置标准化、共享配置 RLOO。配置分布、模型、生成预算、训练 prompts 和验证开销匹配。

条件期望版本用于前面的机制和小步实验；若要把它作为独立方法主张，必须增加能分开“均值改变”和“积分降方差”的相应对照，并从已有额度中重新分配，不能悄悄增加预算。

先测 IID/RLOO 开发训练是否确实学得动，再复制种子。冻结 token 数、updates、batch 和 checkpoint 评估节点，以实测吞吐约束工作量，不承诺任何固定 GPU 小时必然能完成某个规模。

完整论文需要自然反馈证据、独立目标后果、第二种评分机制或模型复现。只拥有数学反例、DP 实现和一条短训练曲线，不足以支持“solid accept”的判断。

---

## 14. 给 Codex 的执行合同

### 14.1 文件与职责

```text
configs/pilot.yaml
src/reward_coupling/advantages.py       # 严格定义 RLOO / std / epsilon
src/reward_coupling/expectation.py      # 二值与网格 DP
src/reward_coupling/sample.py           # 普通 IID actor，无特殊 dependent sampler
src/reward_coupling/execute_tests.py    # 隔离代码执行与逐测试 verdict
src/reward_coupling/gradient_audit.py   # 同候选多权重、固定 theta0
src/reward_coupling/local_steps.py      # 同一初态的 eta / eta/2
src/reward_coupling/statistics.py       # prompt 级配对估计，C/D 双银行误差
scripts/run_pilot.py
reports/contract_report.json
reports/pilot_result.json
reports/budget_ledger.json
reports/claim_ledger.md
```

上面是**待 Codex 实现的文件合同**。本轮实际提供的是附录 CPU 验证器及重跑结果，没有提供已跑通的 LLM 训练系统。

### 14.2 必须保存的数据

```text
checkpoint_revision, checkpoint_hash, tokenizer_revision, template_hash
prompt_id, split, prompt_hash, group_id, slot_id, actor_rng_stream
generation_config, forward_dtype, score_dtype, attention_backend
response_ids, response_length, eos_index_or_null, truncated
sampling_token_logp, scoring_token_logp, active_mask
candidate_code_hash, parser_status
verifier_version, test_manifest_hash, context_ids, context_probabilities
test_verdicts, execution_status, timeouts, exceptions
expected_shared_advantage, expected_independent_advantage, expected_rloo_advantage
suite_result, average_test_result
trainable_parameter_manifest, gradient_reduction, epsilon, ddof
elapsed_gpu_seconds, elapsed_cpu_seconds, generated_tokens, scored_tokens
```

不同完整 actor 组使用独立随机流。固定银行在 S/I 之间完全复用。

### 14.3 默认配置示意

```yaml
experiment: verifier_coupling_pilot
seed: 20260906
model_id: Qwen/Qwen2.5-Coder-1.5B-Instruct
model_revision: MUST_PIN_BEFORE_RUN
tokenizer_revision: MUST_PIN_BEFORE_RUN
dataset: mbpp_plus
dataset_revision: MUST_PIN_BEFORE_RUN
verifier: uniform_single_test
context_selection: prompt_hash_before_actor_generation
reward_values: [0, 1]
actor_sampling: iid
temperature: 1.0
top_p: 1.0
top_k: 0
repetition_penalty: 1.0
max_new_tokens: 384
group_size: 8
num_dev_prompts: 16
num_confirmation_prompts: 32
num_evaluation_prompts: 32
responses_per_dev_prompt: 16
responses_per_confirmation_prompt: 16
responses_per_evaluation_prompt: 16
std_ddof: 0
std_epsilon: 0.000001
epsilon_position: outside_sqrt
constant_reward_group: keep_and_zero_advantage
loss_reduction: sum_tokens_then_mean_responses
clip_policy_ratio: false
kl_coefficient: 0.0
optimizer_map: identity
primary_gradient_scope: all_actor_parameters
allow_silent_lora_fallback: false
allow_reward_based_filtering: false
allow_config_search_on_confirmation: false
budget_gpu_hours_5090: 10.0
budget_gpu_hours_pro6000: 5.0
```

`MUST_PIN_BEFORE_RUN` 是必须填写的 manifest 字段，未填写时程序拒绝运行。不同库对 `top_k=0` 的约定需核查；以“原始完整支持祖先分布”为合同，而不是盲信某个配置字段的默认行为。

### 14.4 数值与显存

主机制审计优先使用可容纳的 FP32 权重和梯度、microbatch=1、必要的 activation checkpointing。它通常比混入低精度有限差分更容易解释；实际显存与吞吐在第一阶段测量。

不需要同时保存每条回答的全参数梯度。按组累积所需加权梯度，分块与固定读出做内积，并在 CPU 保存必要的聚合量。

采样和 teacher-forced scoring 必须核对 token logp。不同精度、不同 kernel、KV cache 与完整前向会产生差异；先用相同概率实现建立主合同，再记录数值误差。不能把 BF16 生成、FP32 重评分自动视为精确 on-policy。

若采用固定 LoRA 子空间，必须单独登记、冻结初始化与 trainable manifest；结论只涉及该参数化，不能继续标记为全参数梯度结论。不能为节省显存悄悄更换 estimand。

不默认使用有限差分测方向：低精度权重的小扰动可能被舍入消除。若采用 JVP 或有限差分，需要独立的误差与核支持核验，不能用它们替代缺失的精度合同。

### 14.5 代码执行安全与数据完整性

模型生成的代码必须放在隔离容器/进程中运行，限制 CPU、内存、执行时间和文件系统权限，关闭网络。不得直接在持有凭据、项目文件或宿主权限的研究进程中执行。

区分候选程序失败与评估基础设施失败。语法错误、候选超时按冻结规则记失败；执行器崩溃或资源错误需要登记并暂停相关题目处理，不能静默记零或选择性重跑有利样本。

固定输入顺序、参考输出、浮点容差及可接受返回类型。每题参考实现先通过整个测试合同。

逐测试 verdict 必须对应独立选择该测试时的执行语义。若候选代码具有全局状态，顺序执行全部测试可能改变后续输出；应使用每个测试的新实例/新进程或可验证的状态重置，并记录方案。否则“整张矩阵的一列”等同于“独立抽到该测试”这一边际匹配前提不成立。对于由系统负载引起的超时波动，另做预定重复核验并记录不确定 verdict，不能将基础设施随机失败混成确定性评分。

### 14.6 执行顺序

```text
CPU-0: 重跑本文 CPU 验证器和附件 v3 检查。
CPU-1: 固定 revisions、prompts、tests、splits、reward law 和预算 ledger。
CPU-2: 构建执行器，用参考程序和人工故障程序验证 test verdict。
GPU-0: 检查 logp、EOS、reduction、精度、显存和吞吐。
GPU-1: 生成 Dev；冻结读出、步长和统计阈值。
GPU-2: 生成 C/D，不看效果改变配置。
CPU-3: 建矩阵，计算 S/I/RLOO 精确权重，完成阴性与 DP 对照。
GPU-3: 计算固定银行梯度、独立目标斜率和预定功能读出。
GPU-4: 仅在仍有额度且合同通过时，执行 eta、eta/2 分支。
REPORT: PASS / STOP / INCONCLUSIVE；逐项对应证据与剩余预算。
```

上面的 GPU 阶段是后续计划。本轮没有执行。

### 14.7 不能写进实验报告的结论

- “数学保证本方法会提升 LLM”。
- “RLOO 在一切 dependent setting 中无偏”。
- “逐回答 TPR/FPR 一样，所以全部评分统计一样”。
- “二值 K=2/3 的结论也适用于连续 rubric”。
- “矩阵内恒等式吻合，所以完成了独立总体预测”。
- “参数 cosine 改变，所以真实正确率下降”。
- “同分组无梯度，所以可以删除再补样”。
- “CPU 脚本通过，所以 BF16 模型采样与梯度实现也正确”。
- “没显著差异，所以证明不存在效应”。
- “共享随机配置必然有害，独立随机配置必然更好”。

---

## 15. 最近邻文献与剩余原创空间

文献核验截至 2026-09-06。以下只判断所核验文本中的假设和结论；没有证明检索穷尽，也不排除尚未公开的并行研究。

| 文献 | 已覆盖内容 | 本项目不能再主张 | 本分支仍需检验的增量 |
|---|---|---|---|
| Davis–Recht [R1] | IID 二值优势的条件线性权重、有限组变换、Bernstein 表示 | 首次推出 w(p)、首次发现归一化改变目标 | 外生验证配置的关联安排与边际等价性边界 |
| SoftmaxGRPO [R2] | 有限组 surrogate、奖励取值相关的目标边界 | 简单换一种 advantage 就构成理论突破 | 对同一评分通道的联合分布进行严格干预 |
| Rubric Dropout [R3] | 共享 mask、共同分母消去、标准化前期望/方差；实际收益 | 首次提出组内共享评分配置 | 标准化后均值的可测差异及其是否解释实际后果 |
| Rate or Fate / RLVεR [R4] | 条件噪声、TPR/FPR 与方向；逐程序 Bernoulli wrapper | 首次研究噪声验证器导致反向学习 | 相同逐回答通道下，共享配置改变有限组方向；明确额外假设 |
| Noise-corrected GRPO [R5] | 奖励噪声校正和无偏梯度目标 | 首次提出 GRPO 奖励去偏 | 配置关联及其与现有校正适用条件的关系 |
| QuasiMoTTo [R6] | 相关回答采样及 RLOO baseline 问题 | 首次发现依赖影响 group-relative 学习 | 回答保持 IID 时，验证器依赖单独产生怎样的标准化效应 |
| Constrained GRPO [R7] | 奖励标量化与共同分母造成目标耦合 | 泛泛的“先归一化还是先聚合不交换” | 固定逐回答分布的配置关联可互换性与自然机制证据 |
| 用户附件 v3 [U1] | 条件信用、二值最小 K 边界、高阶身份反例、停止局部性和因果合同 | 将这些结果再次包装为新项目原创 | 提供一个更便宜、可精确匹配的实际反馈干预场景 |

### 15.1 当前可以诚实写出的贡献

**理论澄清。**给出同边际验证通道的关联可互换性条件、局部反向更新反例、受限安全判据，并明确这些结果由哪些已有工具推出。

**识别协议。**固定候选和逐回答反馈分布，通过精确条件积分测量关联造成的信用变化，连接固定功能和独立目标。

**实证贡献：当前没有。**只有自然机制、独立题目、小步后果与合理预算的训练结果通过后，才能把这一项补入论文。

### 15.2 本轮的研究判断

上一轮“只剩这个方向且具有足够清晰 novelty”的判断应收紧。它与附件 v3 高度相关，核心均值公式也与现有 IID 理论紧密相连。

仍值得做首轮，是因为严格匹配的反馈矩阵干预成本低，能够迅速判断实际相关性；不是因为已经证明它有足够录用价值。

若最终只有旧反例和若干归一化等式，应停止把它当作独立顶会论文。若真实反馈中的差异足够强、可以预先解释并影响独立目标，则更适合与 v3 统一成一篇围绕**分组反馈如何选择信用**的论文，保留单一主线。

---

## 16. 本轮实际执行结果

环境：Python 3.13.5，NumPy 2.3.5；可选 LP 使用 SciPy 1.17.0。

| 实际执行项目 | 结果 |
|---|---:|
| 旧四回答反例重跑，formula vs enumeration | 最大误差 6.63358257213531e−15 |
| 附件 v3 完整 CPU 脚本重跑 | 最大等式误差 1.5543122344752192e−15 |
| 本文新增检查的最大等式误差 | 约 1.2923e−13 |
| 二值 K=2/3，任意固定矩阵边际匹配 | 通过 |
| RLOO 固定矩阵关联不变性 | 通过 |
| Poisson-binomial DP 对小 K 完全枚举 | 通过 |
| 网格奖励 `(sum, sumsq)` DP 对枚举 | 通过 |
| epsilon=1e−6 下旧反例反转 | 通过 |
| 多个非对称随机策略/配置的总体公式 | 通过 |
| 条件噪声安全下界与对称翻转阴性 | 通过有限实例检查 |
| 连续奖励 K=2 反例 | 通过 |
| 常数 w 对应 RLOO 的递推 | 通过有限 K 检查 |
| 连续奖励固定银行协方差等式 | 通过 |
| 可选固定银行 LP，K=3 区间退化 | 通过数值检查 |

误差容差为 `3e-11`。其中的随机有限例子不替代一般证明；LP 输出是浮点优化结果，不是符号证书。

**尚未执行：**真实模型生成、test runner 与模型集成、采样和评分 logp 核验、GPU 吞吐、实际梯度与功能量、校准通过率、独立目标斜率、训练收益。

---

## 参考来源

来源标识用于区分外部文献、附件原有结论和本轮推导。本文的数学证明与新 CPU 结果见正文和附录；不把网页描述当作实验证据。

**[U1] 用户附件。**《Dependent Rollouts v3：条件信用选择、停止局部性与归一化的识别边界》，2026-09-06，文件 `dependent_rollout_research_dossier_v3(1).md`。重点对照 §1–2、§5、§7–10、附录 A。它明确没有预训练模型实验证据。

**[R1] Damek Davis, Benjamin Recht. What is the objective of reasoning with reinforcement learning?** arXiv:2510.13651v1。核对 §4 的自身回答与伙伴计数独立步骤，以及有限组权重表示。  
`https://arxiv.org/html/2510.13651v1`

**[R2] SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation.** arXiv:2608.09271v1。核对二值有限组目标与一般标量奖励适用边界。  
`https://arxiv.org/html/2608.09271v1`

**[R3] Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL.** arXiv:2608.11669v1，2026-08-12。核对 §3.2–3.3 与 Appendix A：共享 mask、归一化分母消去、标准化前分析及其 caveat。  
`https://arxiv.org/html/2608.11669v1`

**[R4] Ali Rad et al. Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards.** arXiv:2601.04411v1。核对条件噪声、方向分析与 Appendix M 的逐程序 Bernoulli wrapper。不将本轮反例写成推翻其全部定理。  
`https://arxiv.org/html/2601.04411v1`

**[R5] Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients.** arXiv:2510.18924v3。确认奖励噪声校正已有直接先行工作；本文未重新证明其全部校正结果。  
`https://arxiv.org/html/2510.18924v3`

**[R6] QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling.** arXiv:2607.01179v1。相关生成与 baseline 问题的最近邻；与本分支的 IID actor 区分。  
`https://arxiv.org/html/2607.01179v1`

**[R7] Constrained Group Relative Policy Optimization.** arXiv:2602.05863v4。核对 scalarization 与共享分母耦合；该工作自身也区分稳定性与精确保留原始 Lagrangian 目标。  
`https://arxiv.org/html/2602.05863v4`

**[R8] Qwen 官方模型卡。** Qwen2.5-Coder-1.5B-Instruct。仅作为模型身份和规模来源；实验必须 pin revision。  
`https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct`

**[R9] EvalPlus 官方仓库。** 提供扩展代码测试与评估实现；实验应冻结数据和执行器版本。  
`https://github.com/evalplus/evalplus`

**[R10] Jeffrey Zhou et al. Instruction-Following Evaluation for Large Language Models.** arXiv:2311.07911。程序可验证指令约束的原始来源。  
`https://arxiv.org/abs/2311.07911`

---

## 附录 A：可选的固定银行关联上下界

此项只用于 CPU 检查或排除极小效应，不是主方法，也不是新的概率优化理论。

对固定二值奖励边际 q_i、固定有符号 score 投影 z_i，令 r 遍历 \(\{0,1\}^K\)，

\[
c(r)=\frac1K\sum_iA_i(r)z_i.
\]

所有保持边际的奖励联合律 P 满足

\[
P(r)\ge0,\quad \sum_rP(r)=1,\quad
\sum_rP(r)r_i=q_i.
\]

最小/最大可能投影为线性规划

\[
\min/\max_P\sum_rP(r)c(r).
\]

K=8 仅有 256 个变量。K=2/3 或 RLOO 的 cost 是边际决定的线性函数，区间应退化。

这个上下界允许针对当前候选选择联合律，因此一般比“单个外生配置安排对所有候选通用”的可行域更宽。它是逐银行的宽松包络，不代表存在可部署的统一 sampler。不能用该 LP 在确认集设计最有害配置，再把结果称为自然现象。

---

## 附录 B：本轮运行的完整 CPU 验证器

将以下代码保存为 `verify_reward_coupling_v4.py`：

```bash
python verify_reward_coupling_v4.py > verify_reward_coupling_v4_results.json
```

只依赖 NumPy；SciPy 仅用于附录 A 的可选 LP，缺失时明确记录跳过。没有模型、网络和 GPU 依赖。


```python
#!/usr/bin/env python3
"""Finite CPU audit of verifier coupling. No pretrained model / GPU experiment.

Python 3.10+, NumPy. SciPy is optional and used only for a finite-bank LP.
Rewards: population std (ddof=0), epsilon outside sqrt, ties retained as zeros.
Run: python verify_reward_coupling_v4.py > verify_reward_coupling_v4_results.json
"""
from __future__ import annotations
import itertools
import json
import math
import platform
from typing import Sequence
import numpy as np

REPORT: dict[str, object] = {}
RNG = np.random.default_rng(20260906)
EPS = 1e-6


def check(name: str, a, b, tol: float = 3e-11) -> None:
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not np.isfinite(err) or err > tol:
        raise AssertionError((name, err, tol))
    REPORT[name] = err


def advantages(rewards: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize the last axis; reward values may be continuous."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim < 1 or r.shape[-1] < 2 or not np.all(np.isfinite(r)):
        raise ValueError('Need finite rewards and group size >= 2')
    if eps < 0 or not math.isfinite(eps):
        raise ValueError('eps must be finite and nonnegative')
    centered = r-r.mean(axis=-1, keepdims=True)
    sd = np.sqrt(np.mean(centered**2, axis=-1, keepdims=True))
    nonconstant = np.ptp(r, axis=-1, keepdims=True) > 0
    return np.divide(centered, sd+eps, out=np.zeros_like(r), where=nonconstant & (sd > 0))


def rloo(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    k = r.shape[-1]
    return (k*r-r.sum(axis=-1, keepdims=True))/(k-1)


def coeffs(k: int, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(k, int) or k < 2 or eps < 0:
        raise ValueError('Invalid k or eps')
    m = np.arange(k, dtype=float)
    dp = np.sqrt((m+1)*(k-1-m))+k*eps
    dm = np.sqrt(m*(k-m))+k*eps
    plus = np.divide(k-1-m, dp, out=np.zeros(k), where=dp > 0)
    minus = np.divide(m, dm, out=np.zeros(k), where=dm > 0)
    return plus, minus


def weight(p: float, k: int, eps: float = EPS) -> float:
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must lie in [0,1]')
    plus, minus = coeffs(k, eps)
    return float(sum(math.comb(k-1, m)*p**m*(1-p)**(k-1-m)
                     *(plus[m]+minus[m]) for m in range(k)))


def independent_binary(q: Sequence[float], eps: float = EPS) -> np.ndarray:
    """Exact conditional expected advantages, O(K^3) simple DP.

    q_i is the pass probability of a fixed candidate, NOT a prompt pass rate.
    Supports deterministic q=0/1 without division by q or 1-q.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or len(q) < 2 or not np.all(np.isfinite(q)) or np.any((q<0)|(q>1)):
        raise ValueError('Invalid candidate marginals')
    plus, minus = coeffs(len(q), eps)
    out = np.zeros(len(q))
    for i in range(len(q)):
        pmf = np.array([1.])
        for j in range(len(q)):
            if j != i:
                pmf = np.convolve(pmf, [1-q[j], q[j]])
        out[i] = q[i]*(pmf@plus)-(1-q[i])*(pmf@minus)
    return out


def shared_matrix(matrix: np.ndarray, mu: Sequence[float] | None = None,
                  eps: float = EPS) -> np.ndarray:
    """matrix[i,u] is the frozen reward for candidate i, context u."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 1:
        raise ValueError('Expected [K, number_of_contexts]')
    w = np.ones(mat.shape[1])/mat.shape[1] if mu is None else np.asarray(mu, dtype=float)
    if w.shape != (mat.shape[1],) or np.any(w<0) or not np.all(np.isfinite(w)) or not np.isclose(w.sum(),1):
        raise ValueError('Invalid context distribution')
    return advantages(mat.T, eps).T@w


def independent_binary_enum(q, eps: float = EPS) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if len(q)>12:
        raise ValueError('Enumeration only for small CPU checks')
    bits = np.array(list(itertools.product([0.,1.], repeat=len(q))))
    mass = np.where(bits == 1, q, 1-q).prod(axis=1)
    return mass@advantages(bits, eps)


def independent_grid(pmfs: np.ndarray, scale: int, eps: float = EPS) -> np.ndarray:
    """Rewards v/scale, v=0..V. DP over partner sum and sum of squares.

    This integrates a prescribed discrete reward law exactly up to FP64 error;
    it does not round arbitrary continuous rewards into that law.
    """
    p = np.asarray(pmfs, dtype=float)
    if p.ndim != 2 or p.shape[0]<2 or scale<=0 or not isinstance(scale, int):
        raise ValueError('Invalid grid distribution')
    if np.any(p<0) or not np.all(np.isfinite(p)) or not np.allclose(p.sum(1),1):
        raise ValueError('Each row must be a probability distribution')
    k, nv = p.shape
    out = np.zeros(k)
    for i in range(k):
        state = {(0,0):1.}
        for j in range(k):
            if i == j:
                continue
            nxt: dict[tuple[int,int],float] = {}
            for (s,q), mass in state.items():
                for v in np.flatnonzero(p[j]):
                    key=(s+int(v),q+int(v*v))
                    nxt[key]=nxt.get(key,0.)+mass*p[j,v]
            state=nxt
        for v in np.flatnonzero(p[i]):
            for (s,q), mass in state.items():
                total=s+int(v); square=q+int(v*v)
                # Integer numerator prevents cancellation for exact ties.
                numerator=k*square-total*total
                if numerator < 0:
                    raise AssertionError('Negative exact variance')
                if numerator:
                    out[i]+=p[i,v]*mass*(k*v-total)/(math.sqrt(numerator)+k*scale*eps)
    return out


def independent_grid_enum(pmfs: np.ndarray, scale: int, eps: float = EPS) -> np.ndarray:
    k,nv=pmfs.shape
    if nv**k>200000:
        raise ValueError('Enumeration guard')
    out=np.zeros(k)
    for values in itertools.product(range(nv),repeat=k):
        v=np.array(values)
        mass=float(np.prod(pmfs[np.arange(k),v]))
        out+=mass*advantages(v/scale,eps)
    return out


def population(matrix, pi, mu, k: int, eps: float = EPS):
    """Deterministic context table, categorical softmax actor."""
    mat=np.asarray(matrix,float);pi=np.asarray(pi,float);mu=np.asarray(mu,float)
    jac=np.diag(pi)-np.outer(pi,pi)
    pu=pi@mat; gu=jac@mat; q=mat@mu
    gs=sum(mu[u]*weight(float(pu[u]),k,eps)*gu[:,u] for u in range(len(mu)))
    gi=weight(float(pi@q),k,eps)*(jac@q)
    return gs,gi,pu,gu


def enumerate_shared_population(matrix, pi, mu, k: int, eps: float = EPS):
    n=len(pi)
    if n**k>100000:
        raise ValueError('Enumeration guard')
    ys=np.array(list(itertools.product(range(n),repeat=k)))
    mass=np.prod(pi[ys],axis=1)
    score=np.eye(n)[ys]-pi
    ans=np.zeros(n)
    for u in range(len(mu)):
        a=advantages(matrix[ys,u],eps)
        ans+=mu[u]*np.einsum('b,bk,bkd->d',mass,a,score)/k
    return ans


def optional_lp(q, signed_scores, eps: float = EPS):
    """Pointwise frozen-bank Frechet envelope, not a deployable sampler."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return {'status':'SKIPPED: SciPy not installed'}
    q=np.asarray(q,float);z=np.asarray(signed_scores,float);k=len(q)
    if k>10 or z.shape!=q.shape:
        raise ValueError('LP is restricted to small fixed banks')
    bits=np.array(list(itertools.product([0.,1.],repeat=k)))
    cost=advantages(bits,eps)@z/k
    aeq=np.vstack([np.ones(len(bits)),bits.T]);beq=np.r_[1.,q]
    lo=linprog(cost,A_eq=aeq,b_eq=beq,bounds=(0,None),method='highs')
    hi=linprog(-cost,A_eq=aeq,b_eq=beq,bounds=(0,None),method='highs')
    if not lo.success or not hi.success:
        raise RuntimeError('LP solver failed')
    return {'min':float(lo.fun),'max':float(-hi.fun),
            'max_primal_residual':float(max(np.max(abs(aeq@lo.x-beq)),np.max(abs(aeq@hi.x-beq)))),
            'status':'Numerical LP optimum; FP64 solver, not symbolic certificate'}


def run() -> None:
    REPORT['environment']={'python':platform.python_version(),'numpy':np.__version__}
    for k in (2,3,7):
        for val in (.1,.2,1/3):
            check(f'constant_reward_K{k}_{val}_error',advantages(np.full(k,val),0.),0.)
    max_small=0.;max_dp=0.;max_rloo=0.;max_grid=0.
    for eps in (0.,EPS):
        for k in (2,3):
            gamma=(k-1)/(math.sqrt(k-1)+k*eps)
            for r in itertools.product([0.,1.],repeat=k):
                max_small=max(max_small,float(np.max(abs(advantages(np.array(r),eps)-gamma*rloo(np.array(r))))))
            for _ in range(12):
                mat=RNG.integers(0,2,size=(k,7));mu=RNG.dirichlet(np.ones(7))
                check(f'fixed_bank_K{k}_eps{eps}_{_}_error',shared_matrix(mat,mu,eps),independent_binary(mat@mu,eps))
        for k in range(2,9):
            for rep in range(4):
                q=RNG.random(k)
                if rep==0:q[:2]=[0.,1.]
                max_dp=max(max_dp,float(np.max(abs(independent_binary(q,eps)-independent_binary_enum(q,eps)))))
                mat=RNG.integers(0,2,size=(k,5));mu=RNG.dirichlet(np.ones(5))
                max_rloo=max(max_rloo,float(np.max(abs(rloo(mat.T).T@mu-rloo(mat@mu)))))
        for k,nv in [(2,3),(3,4),(4,3),(5,4)]:
            for _ in range(3):
                pmf=RNG.dirichlet(np.ones(nv),size=k)
                max_grid=max(max_grid,float(np.max(abs(independent_grid(pmf,nv-1,eps)-independent_grid_enum(pmf,nv-1,eps)))))
    check('binary_K2_K3_sample_identity_max_error',max_small,0.)
    check('binary_DP_vs_enumeration_max_error',max_dp,0.)
    check('RLOO_fixed_bank_coupling_invariance_max_error',max_rloo,0.)
    check('grid_DP_vs_enumeration_max_error',max_grid,0.)

    pi=np.array([.005,.005,.495,.495]);gold=np.array([1.,1.,0.,0.])
    mat=np.array([[0,0,1,1],[0,0,1,1],[0,1,1,0],[0,1,0,1]],float)
    mu=np.array([.17,.23,.30,.30]);jac=np.diag(pi)-np.outer(pi,pi);gt=jac@gold
    for eps in (0.,EPS):
        gs,gi,pu,gu=population(mat,pi,mu,8,eps)
        enum=enumerate_shared_population(mat,pi,mu,8,eps)
        check(f'old_counterexample_eps{eps}_formula_enum_error',gs,enum)
        cs=float(gs@gt/(gt@gt));ci=float(gi@gt/(gt@gt))
        assert cs<0<ci
        check(f'old_counterexample_eps{eps}_shared_collinearity_error',gs,cs*gt)
        check(f'old_counterexample_eps{eps}_independent_collinearity_error',gi,ci*gt)
        REPORT[f'counterexample_eps{eps}']={'shared_c':cs,'independent_c':ci,'J_true':float(pi@gold),
                'true_slope_shared':float(gt@gs),'true_slope_independent':float(gt@gi)}
        ws=np.array([weight(float(v),8,eps) for v in pu]);gb=gu@mu
        cov=(gu-gb[:,None])@(mu*(ws-mu@ws))
        check(f'population_covariance_eps{eps}_error',gs-(mu@ws)*gb,cov)
        # Two-action/private-noise version gives exactly the same scalar.
        c_class=.30*weight(.505,8,eps)-.23*weight(.99,8,eps)
        check(f'class_conditional_reduction_eps{eps}_error',cs,c_class)

    # Population formula independently tested on arbitrary nonsymmetric tables.
    err=0.
    for k in (2,3,4,5):
        for _ in range(7):
            p=RNG.dirichlet(np.ones(3));m=RNG.integers(0,2,size=(3,4)).astype(float)
            mu0=RNG.dirichlet(np.ones(4));gs,_,_,_=population(m,p,mu0,k)
            err=max(err,float(np.max(abs(gs-enumerate_shared_population(m,p,mu0,k)))))
    check('random_population_formula_enum_max_error',err,0.)

    # All per-candidate marginals equal 1/2 yet normalized shared credit differs.
    mat0=np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,1,1]],float)
    sa=shared_matrix(mat0);ia=independent_binary(mat0.mean(1))
    check('equal_marginal_independent_advantage_zero_error',ia,0.)
    assert np.linalg.norm(sa)>0.05
    REPORT['equal_marginals_fixed_bank']={'marginals':mat0.mean(1).tolist(),'shared_advantages':sa.tolist(),
                                       'independent_advantages':ia.tolist()}
    gs0,gi0,_,_=population(mat0,np.ones(4)/4,np.ones(4)/4,4)
    check('equal_marginal_population_independent_gradient_zero_error',gi0,0.)
    assert np.linalg.norm(gs0)>0
    REPORT['equal_marginal_population_shared_gradient']=gs0.tolist()

    # Positive affine common scaling cancels only when epsilon=0.
    r=np.array([0.,.2,.4,1.])
    check('affine_scale_eps0_error',advantages(7*r+2,0.),advantages(r,0.))
    REPORT['epsilon_affine_scale_discrepancy']=float(np.max(abs(advantages(7*r+2,.1)-advantages(r,.1))))
    assert REPORT['epsilon_affine_scale_discrepancy']>0.1

    # K=2 invariance does not extend to nonbinary rewards.
    continuous=np.array([[1.,0.],[0.,.2]])
    s=shared_matrix(continuous,eps=0.)
    pgrid=np.zeros((2,6));pgrid[0,[0,5]]=.5;pgrid[1,[0,1]]=.5
    i=independent_grid(pgrid,5,0.)
    check('continuous_K2_shared_zero_error',s,0.)
    check('continuous_K2_independent_advantage_error',i,[.25,-.25])
    REPORT['continuous_K2_counterexample']={'shared':s.tolist(),'independent':i.tolist()}

    # Uniform shared complement flips: no reversal at noise < 1/2.
    flip=0.2; ps=np.linspace(.01,.99,99)
    check('weight_complement_symmetry_error',
          [weight(float(p),8) for p in ps],[weight(float(1-p),8) for p in ps])
    cs=np.array([(1-2*flip)*weight(float(p),8) for p in ps])
    assert cs.min()>0
    REPORT['symmetric_shared_flip_min_positive_coefficient']=float(cs.min())

    # Rigorous (possibly loose) positivity certificate uses Bernstein coefficients.
    safety={}
    for k in (4,8,16):
        ap,am=coeffs(k);phi=ap+am;low=float(phi.min());high=float(phi.max())
        threshold=(high-low)/(high+low)
        safety[str(k)]={'min_Bernstein_coefficient':low,'max_Bernstein_coefficient':high,
                       'sufficient_mean_Youden_threshold':threshold}
        for _ in range(100):
            u=RNG.dirichlet(np.ones(6));t=RNG.random(6);f=RNG.random(6);a=t-f;j=float(RNG.random())
            mean_a=float(u@a)
            c=float(sum(u[v]*weight(float(f[v]+a[v]*j),k)*a[v] for v in range(6)))
            lower=(low+high)*mean_a/2-(high-low)/2
            if c<lower-2e-12:raise AssertionError('Safety lower bound failed')
    REPORT['class_conditional_safety_certificates']=safety

    # Uniqueness recurrence: constant Bernstein coefficient yields scaled RLOO.
    for k in range(2,13):
        c=1.7;apos=np.zeros(k+1);aneg=np.zeros(k+1)
        for m in range(k):
            apos[m+1]=c+aneg[m]
            if m+1<k:aneg[m+1]=-(m+1)*apos[m+1]/(k-m-1)
        check(f'constant_weight_RLOO_recurrence_K{k}_error',apos[1:],c*(k-np.arange(1,k+1))/(k-1))

    # Exact finite-bank normalization vs averaging decomposition.
    for _ in range(8):
        matc=RNG.random((4,6));mu0=RNG.dirichlet(np.ones(6))
        centered=matc-matc.mean(0,keepdims=True);sd=matc.std(0)
        v=1/(sd+EPS);eb=centered@mu0;ev=float(mu0@v)
        cov=((centered-eb[:,None])*(v-ev))@mu0
        check(f'finite_bank_centering_covariance_{_}_error',shared_matrix(matc,mu0),ev*eb+cov)

    REPORT['LP_K3']=optional_lp([.2,.5,.8],[1.,-.4,.2])
    REPORT['LP_K4_equal_marginals']=optional_lp([.5]*4,[1.,1.,-1.,-1.])
    if 'min' in REPORT['LP_K3']:
        check('LP_K3_collapsed_interval_error',REPORT['LP_K3']['min'],REPORT['LP_K3']['max'])
        assert REPORT['LP_K4_equal_marginals']['max']>0>REPORT['LP_K4_equal_marginals']['min']
    REPORT['max_asserted_equality_error']=max(float(v) for k,v in REPORT.items() if k.endswith('_error') and isinstance(v,(int,float)))
    REPORT['status']='PASS: finite CPU checks only. No LLM outputs, model downloads, GPU, or training.'


if __name__=='__main__':
    run()
    print(json.dumps(REPORT,indent=2,ensure_ascii=False))
```


## 附录 C：交付物与 Codex 首次执行指令

文档正文和附录 B 自包含；独立脚本与 JSON 用于复跑和逐项核查。`pilot.yaml` 是待填写 revision 的配置模板，尚不是可直接启动真实模型训练的系统。

```text
先阅读本文 §0–6 的假设与审计结论，再实现 §7、§10–14。
首先在 CPU 运行 verify_reward_coupling_v4.py、verify_dependent_v3.py
和 verify_previous_counterexample.py，对照各自 JSON；不据此声称 LLM 效应。

把本文的优势、固定反馈矩阵、独立银行统计和执行器合同实现成测试。
先固定 model/tokenizer/data revisions、测试清单与数据 hash。
预算选择 RTX5090 10h 或 Pro6000 5h，不同时各运行一份 pilot。
仅在资产就绪后启用 GPU 计时；所有失败重试和评估纳入预算。
遇到假设不满足、统计不充分或额度耗尽时，生成明确报告并停止。

不得按效果修改题目、反馈配置、reward 定义或主要统计终点。
交付代码、manifest、合同测试、逐项 claim ledger 和预算明细。
本文没有保证训练收益；不要为维护本文结论而筛选实验。
```

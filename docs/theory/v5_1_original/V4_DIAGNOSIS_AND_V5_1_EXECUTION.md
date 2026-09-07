# v4 实验诊断与 v5.1 实施修订

**日期：2026-09-07**  
**依据：用户提供的 `ANALYSIS.md`、v3、v4、v5 文档，以及本轮独立数学推导、原始文献核查和 CPU 有限枚举。**  
**状态：未加载预训练模型，未连接原服务器，未重跑 GPU；未获得原始 C/D 银行、方向张量、逐 token 日志及完整仓库。**

本文使用 `ANALYSIS.md` 中报告的数值诊断实验；这些数值没有在原服务器上独立复算。本轮新增的数值来自同包 `verify_v5_1.py` 的有限模型计算。两种证据不能互换。

## 0. 研究决定

1. v4 已有固定银行信用差异；尚无可靠的总体方向、独立任务收益或有限步收益结论。D 正点估计受未中心化 score 噪声严重影响；局部步长尚未进入验证过的线性区间。[F1 §1–4]
2. v4 只做一次有明确边界的修复：补回唯一未测的失败回答投影，恢复中心化诊断；在少量固定读出上定位数值与共同曲率。修复用于解释已有数据，不自动升级为新的确认实验。
3. **v5 默认独立分层构造应更换主估计器。**使用完整分层 U 估计器（下文记为 `Strat-full`）；原 `Strat-cross` 保留为一般相关块的合法实现与小规模对照。Codex 的重配对建议成立，其均值保持与方差支配证明见 §4。
4. v5 的主要比较为同 K、同 N、同参数空间下的 `Strat-full` 与 `IID-all`。新增理论不保证前者普遍胜出。层信息无效时，它可以严格更差。
5. 先修测量、计算和采样合同，再进行一个有统计可判定性的筛选。v4 的任务收益不必先做成，才允许检查 v5；二者关注的平均更新目标不同。
6. **不因从 v4 升为 v5.1 就重置预算。**先从原回执汇总已用 GPU 时间。尚可投入的小时数服从原项目总上限 200h RTX 5090 / 100h Pro 6000。两种硬件分别核算。新试点所需规模必须经过实测吞吐核算，10h/5h 是原首轮上限，不是每份文档自动新增的拨款。

本轮不能确认的内容：原模型具体 revision、实际参数布局、数值路径、所有 bank 的覆盖和是否存在额外日志。Codex 从本地 manifest 读取这些内容；不能根据本文示例猜测旧配置。

---

## 1. v4 到底卡在哪里

### 1.1 报告支持的状态

| 对象 | 报告中的事实 | 可以支持的判断 |
|---|---|---|
| 信用差 | Dev 6/32 组、C 2/8 组非零 | 固定自然银行存在稀疏机制信号；没有总体效应量证明 |
| 独立目标 | D 4 个 prompt、每题 8 条，共 31/32 通过全部测试 | 当前银行接近饱和，失败信息极少 |
| D 原始斜率 | +81.99273267；一个样本贡献 +85.11101439 | 点估计极敏感；不能作为稳定正收益 |
| 步长 | 1e-4 和 5e-5 的相对残差 1.02998、0.69919 | 两个步长均未验证局部线性近似 |
| 回放成本 | C 差分约 710 秒；C/D 投影约 1527 秒 | 该全参数 FP32、CPU 卸载配置挤压了有效重复规模 |
| 功能读出 | canonical-vs-None 序列 log-odds | 反映固定读出，不能直接等同于任务正确率 |
| 正缩放对照 | C 阶段尚未计算 | 不能由非零功能差宣布总体非标量效应 |

来源：[F1 §1–4]。这里没有证据否定 v4 的条件化恒等式；也没有充分证据支持其自然任务训练价值。

### 1.2 这不是单一的“模型太小 / η 太大”问题

同时存在四个瓶颈：

- **反馈结构**：大量回答对全部测试一致，评分关联没有可利用的差异。
- **目标测量**：原始 reward-score 积在近饱和奖励下带入大量零均值 score 噪声。
- **局部数值**：两臂共有的位移较大，差值线性化受到混合曲率影响。
- **计算配置**：昂贵的全参数回放迫使统计规模缩成 C8/D4。

缩小 η 只处理其中一部分；更换 baseline 也不会补出缺少的独立题目。上一版计划未把低方差目标估计器与真实回放吞吐设为硬前置条件，这一点应当修订。

### 1.3 自然评分矩阵的低成本结构检查

对固定回答 i 的各测试反馈记为 B_{iu}，其平均通过率 q_i。若每行 q_i 都是 0 或 1，则该银行中所有奖励都与配置无关，shared/independent 的优势完全一致。

因此，“存在部分测试通过、部分失败的回答”是该干预出现差异的必要条件之一，不是充分条件。许多样本仅全对或全错，会让机制信号自然稀疏。

先在 CPU 上输出每题 q_i 分布、部分正确回答占比、精确优势差及零差异原因，再决定哪些梯度计算有信息。可以省去严格为零的差分 backward，但零组仍计入总体平均；不能只保留有信号组后声称估计原总体。

---

## 2. D 目标：一条缺失投影如何恢复诊断

### 2.1 固定 C 后的正确估计对象

固定已经生成的 C 银行和方向

\[
d=\widehat G_{S,C}-\widehat G_{I,C},\qquad z(Y)=S(Y)^\top d.
\]

若生成分布和评分时的原始策略一致，且 EOS、截断、mask 和 reduction 正确，则

\[
\mathbb E[z\mid x,C]=0.
\]

对任何独立于当前回答、给定 prompt 可确定的 baseline b(x)，

\[
\mathbb E[(r-b)z\mid x,C]=\mathbb E[rz\mid x,C].
\tag{1}
\]

这只是 score 恒等式；不是通过减掉一个数人为改变研究目标。经典 baseline 文献已经讨论其最优选择与方差。[R1]

### 2.2 当前银行可以精确还原的两个量

报告给出一个未测的失败回答：

\[
z_f=S(\text{Mbpp/474:0:7})^\top d.
\]

它在原 `r*z` 计算中因为 r=0 被跳过，**未测量不等于零**。[F1 §2]

固定 baseline b=1，按 4 个 prompt 等权、每题 8 条平均，有

\[
\boxed{T_{b=1}=-z_f/32.}
\tag{2}
\]

报告中的 RLOO 重构式为

\[
\boxed{T_{\mathrm{RLOO}}=-0.7801849729962018-z_f/32.}
\tag{3}
\]

因此，补一次方向投影，就能同时恢复两种合法中心化诊断。式 (3) 为正当且仅当

\[
z_f<-24.96591913587846.
\]

这不预测 z_f 的值。本文没有原模型和 d，也没有测量它。

**实施要求：**使用旧 checkpoint、旧方向张量、同一参数顺序和同一原始序列 score；保留真实失败回答，不能重新生成一条替代。可用一次完整反传后与 d 做内积，或已验证的方向导数实现。模型和方向加载有真实成本，应与下一次必要开机合并。

`b=1` 与 RLOO 均可报告，但不能根据哪个给出正值再决定主估计器。它们是看到旧数据后增加的诊断，旧 D 不再构成这项选择的未见确认集。

### 2.3 为什么中心化仍救不了一条失败的信息量

对一个 prompt，设 r 为二值，p=E r，n 条回答 IID，z 的均值为零，g=E(rz)。RLOO 的精确方差是

\[
\boxed{
\operatorname{Var}(T_R)
=\frac{\operatorname{Var}((r-p)z)}n
+\frac{p(1-p)\mathbb Ez^2+g^2}{n(n-1)}.
}
\tag{4}
\]

**证明。** RLOO 等于二阶 U 统计量，核为

\[
h((r,z),(r',z'))=\tfrac12(r-r')(z-z').
\]

一阶投影为 \(((r-p)z-g)/2\)，二阶 canonical 项为

\[
-\tfrac12[(r-p)z'+(r'-p)z].
\]

前者给出式 (4) 第一项；后者二阶矩为 \([p(1-p)E z^2+g^2]/2\)，除以 \(\binom n2\) 得第二项。□

式 (4) 同时说明：RLOO 通过估计 baseline 引入有限样本代价；它不普遍支配每个独立固定 baseline。

对 b=1，设失败率 ε_f=1-p，失败条件下 μ_f=E[z|r=0]，则

\[
g=-\varepsilon_f\mu_f,
\quad
\operatorname{Var}(T_1)
=\frac{\varepsilon_fE[z^2\mid r=0]-\varepsilon_f^2\mu_f^2}{n}.
\tag{5}
\]

当 μ_f≠0 时，相对方差为

\[
\frac{\operatorname{Var}(T_1)}{g^2}
=\frac{E[z^2\mid r=0]}{n\varepsilon_f\mu_f^2}-\frac1n.
\]

有效信息受失败样本数以及失败 score 的波动控制。31 条成功不会等价于 31 条有效的失败方向证据。这里 ε_f 是总体概率，不能把观察到的 1/32 当作已知值代入并宣称获得了置信保证。

方向最优独立常数为

\[
b^*=E[rz^2]/E[z^2]
\]

（分母正时）。需要独立校准或有效的交叉拟合，不在 4 个已见题上寻优。首轮不必增加一个复杂 baseline 模型。

### 2.4 保存格式修正

所有未来投影行必须包含：

```text
response_id, reward, projection_status = measured | not_measured
projection_value = float | null
checkpoint_hash, parameter_space_hash, direction_hash
score_reduction, response_mask_hash, dtype_contract
```

零 reward 行可以按计算需求延迟反传，但必须保留 tokens 和明确缺失标记。不要因为原估计权重为零，就把未来可能需要的 score 当作已测零。

---

## 3. C/D 推断与局部步长：分别解决

### 3.1 条件问题与总体问题

给定现有 C 的 d，可以问：**沿这次实际得到的方向，目标导数是多少？**此时只传播 D 的误差。这个问题范围较小，但有意义。

若问：**两种评分流程的总体平均更新会怎样改变目标？**则 d 本身也随机，需要传播 C 和 D 两侧误差。

设独立均值估计 \(\hat g,\hat\delta\) 的均值为 g、δ，协方差为 V_g、V_δ（已是均值协方差），则

\[
\boxed{
\operatorname{Var}(\hat g^\top\hat\delta)
=\delta^\top V_g\delta+g^\top V_\delta g+
\operatorname{tr}(V_gV_\delta).
}
\tag{6}
\]

证明是分别展开两侧中心化误差，利用独立性消去交叉项。固定优化器映射 M 时相应为

\[
\delta^\top M^\top V_gM\delta+
 g^\top MV_\delta M^\top g+
\operatorname{tr}(M^\top V_gMV_\delta).
\]

4 个 D prompt、每题一组，不能充分识别题间异质性和题内采样噪声。Bootstrap 不能创造从未出现的失败模式，也不能把同组 token 或虚拟组当成独立重复。

下一次设计应保存 prompt 层和独立 macro-bank 层。题内方差使用重复 macro-bank；跨题目标使用 prompt 层。当前旧银行最多用于条件诊断和方差预估。

### 3.2 报告的 Taylor 式正确，但还没有完成定因

设

\[
\Delta=d_S-d_I,\quad m=(d_S+d_I)/2,
\quad D(h)=F(\theta+h d_S)-F(\theta+h d_I).
\]

F 为 C³，方向固定时

\[
D(h)=hDF(\theta)\Delta+h^2D^2F(\theta)[m,\Delta]+O(h^3).
\tag{7}
\]

二阶项含共有位移 m，不能仅由 \(\|\Delta\|^2\) 估计。在所走线段上 \(\|D^2F\|\le L\) 时，

\[
\|D(h)-hDF(\theta)\Delta\|
\le\frac{Lh^2}{2}(\|d_S\|+\|d_I\|)\|\Delta\|.
\]

报告中独立臂一阶读出位移可达约 -3.97，而两臂差约 -0.35。这足以警惕共同位移，但尚不足以排除数值或更新实现问题。[F1 §3]

### 3.3 用三个测量定位问题，不只继续缩小 η

**A. 原点的纯差方向中心差分**

\[
C(h)=\frac{F(\theta+h\Delta/2)-F(\theta-h\Delta/2)}h
=DF(\theta)\Delta+O(h^2).
\tag{8}
\]

它检验梯度、方向写入和数值回放是否一致；它不是两条真实训练臂的收益。

**B. 真实两臂的共同中点**

令 \(\theta_h=\theta+hm\)，则

\[
D(h)=hDF(\theta_h)\Delta+O(h^3\|\Delta\|^3).
\tag{9}
\]

若第三导数的算子范数上界为 M_3，余项上界为 \(M_3|h|^3\|\Delta\|^3/24\)。在预算允许时比较原点与中点的方向导数，可以识别共有位移引起的斜率变化。

**C. 正负步的奇偶部分**

\[
D_{\mathrm{odd}}(h)=\frac{D(h)-D(-h)}2=hDF(\theta)\Delta+O(h^3),
\]

\[
D_{\mathrm{even}}(h)=\frac{D(h)+D(-h)}2=h^2D^2F(\theta)[m,\Delta]+O(h^4)
\tag{10}
\]

最后一个 O(h⁴) 使用 C⁴ 光滑性。它把首要混合曲率与线性项分开，正负步均只作局部诊断。

**执行次序：**先重复 θ 原点读出确定数值底线；再做 (8) 的两档 h；只有 (8) 成立而真实两臂仍偏离，才加 (9) 或 (10)。h 不根据改善方向挑选。每次从同一 checkpoint 重置，不能累积加减浮点参数。原模型回放路径和读出计算精度保持一致。

更小 h 可能落入参数 ULP、长序列 log probability 抵消和不确定性底线。若预测差异比数值底线还小，结论是当前配置测不准，不能无限缩步直到出现漂亮比值。

### 3.4 需要输出的最小数值表

```text
h, F(theta), F(theta+h*dS), F(theta+h*dI)
J(theta)*Delta, J(theta+h*m)*Delta [only if measured]
absolute_difference, absolute_remainder, relative_remainder
repeated_forward_floor, exact_reset_status, nonzero_parameter_changes
```

所有主张限定于这些固定 F。任务正确率需要独立生成或有效 score 估计，不能拿 canonical-vs-None 自动替代。

---

## 4. v5 主估计器修订：完整分层 U 形式

### 4.1 假设与目标

固定 prompt、θ、采样规则和可复用奖励，写

\[
P=\frac1m\sum_{j=1}^{m}P_j.
\]

每层 j 有 B≥K 个样本

\[
Z_{bj}\sim P_j,
\]

**所有样本在给定冻结历史后相互独立**。这里独立性比“不同块独立”更强。层标签可以是潜变量，同一回答可以出现在多个层。score 始终为原始 P 对应策略的 score，不对条件 P_j 求导。

对平方可积、对称的 K 元向量核 H，目标

\[
G_K=E_{P^{\otimes K}}H.
\]

normalized GRPO 的目标就是其 IID K 更新；恢复它不等于恢复原始 expected reward 梯度。[F3 §3]

### 4.2 完整分层估计器

对整数向量 k=(k_1,...,k_m)，\(\sum_jk_j=K\)，令 U_k 表示：每层从 B 个实际样本中选 k_j 个互不重复样本，对全部层内子集组合平均 H。

定义

\[
\boxed{
T_{\mathrm{full}}=
\sum_{k_1+\cdots+k_m=K}
\frac{K!}{m^K\prod_jk_j!}\,U_k.
}
\tag{11}
\]

**均值证明。** P 的 K 重乘积等于层标签 IID 均匀抽取的混合。标签计数 k 的概率就是式 (11) 的权重。各层所取实际样本独立，U_k 的期望等于对应层组合下 H 的期望。对 k 求和得到 \(E T_{full}=G_K\)。□

### 4.3 Codex 的重配对解释成立

令 \(\mathcal S\) 为每层无序的 B 个完整样本集合。对每层独立随机置换 b 编号，再计算旧 `Strat-cross`。给定 \(\mathcal S\)，这些配对的均匀平均恰好是式 (11)：K 个虚拟槽位的层计数是多项式分布，同层索引均匀无放回，不同层的索引独立。

因此

\[
\boxed{T_{full}=E[T_{cross}\mid\mathcal S],}
\]

\[
\boxed{
\operatorname{Cov}(T_{cross})-\operatorname{Cov}(T_{full})
=E\operatorname{Cov}(T_{cross}\mid\mathcal S)\succeq0.
}
\tag{12}
\]

本式为 Rao–Blackwell 条件期望事实，不是新的统计工具。[R2]

**决定：**在 v5 默认独立分层下，旧块配对没有额外统计含义，主实现应改成式 (11)。一般相关块、共同 offset 或组依赖 judge 不能直接使用它；这些情况下旧 `Strat-cross` 的独立块构造仍有价值。

### 4.4 不能把适用边界省略

取 B=m=K=2，每块把一个 Bernoulli(1/2) 回答复制两次，块之间独立。原 score 为 r−1/2。

旧跨块估计器平均更新为 0.249999500001，符合目标。误用式 (11) 后平均值变成 0.18749962500075，产生 25% 相对偏差。

因此：**每层各自 IID，不代表各层相互独立。**需要跨层独立的生成流；共享前缀随机数、lattice shift、环境状态或随机 judge 都要检查。已经复制的回答不能靠重新编号恢复独立。

---

## 5. 与 IID-all 的精确比较

### 5.1 新增的改善项

原 v5 对 Strat-cross 的精确分解仍然正确，但新估计器必须加入被条件期望消除的方差：

\[
\boxed{
\Sigma_{I,all}-\Sigma_{S,full}
=(\Sigma_{I,all}-\Sigma_{S,cross})
 +(\Sigma_{S,cross}-\Sigma_{S,full}).
}
\tag{13}
\]

后项半正定。它提高了候选算法，却仍不保证超过 IID-all。

### 5.2 完整分层估计器的有限样本协方差

令多重指标 α=(α_1,...,α_m)，s=|α|∈{1,...,K}。定义

\[
h_\alpha(x)=
\left[\prod_{j=1}^m\prod_{\ell=1}^{\alpha_j}(\delta_{x_{j\ell}}-P_j)\right]
P^{\otimes(K-s)}H.
\tag{14}
\]

其含义是：对 α 对应的每个已选自变量，减去在所属层分布下的平均；其余变量对 P 积分。每个 h_α 在任意一个自变量对所属 P_j 积分时为零。

设

\[
c_\alpha=\frac{(K)_s}{m^s\prod_j\alpha_j!},
\quad (K)_s=K!/(K-s)!,
\quad \Sigma_\alpha=E[h_\alpha h_\alpha^\top].
\]

则

\[
T_{full}-G_K=\sum_{1\le|\alpha|\le K}c_\alpha U_\alpha(h_\alpha),
\]

\[
\boxed{
\operatorname{Cov}(T_{full})=
\sum_{1\le|\alpha|\le K}
\frac{c_\alpha^2}{\prod_j\binom B{\alpha_j}}\Sigma_\alpha.
}
\tag{15}
\]

**证明。** 对式 (11) 中每个核按所属层测度中心化展开。固定 s 个有标签自变量，其进入 K 个位置的选择计数给出 \((K)_s/\prod\alpha_j!\)，标签概率为 m^{-s}，其余位置积分为 P；得到 c_α。不同实际数据子集的 canonical 项只要有一个索引不匹配，对该独立自变量积分就为零。相同子集数量为 \(\prod_j\binom B{\alpha_j}\)，从而得到式 (15)。□

这是多样本 U-statistic 的标准正交化思路。本文给出本项目核的具体系数和可运行验证，不申报一般理论的原创权。[R2–R3]

### 5.3 无效分层为什么仍可能更差

若所有 P_j=P，令 Σ_s 为普通 IID Hoeffding 第 s 阶协方差。定义层计数概率

\[
M_s(\alpha)=\frac{s!}{m^s\prod_j\alpha_j!},
\quad
Q_s(\alpha)=\frac{\prod_j\binom B{\alpha_j}}{\binom{Bm}s}.
\]

前者是目标 IID 层标签的多项式分布，后者是从平衡实际银行均匀选 s 个不同样本时的多元超几何分布。式 (15) 化为

\[
\boxed{
\Sigma_{S,full}=
\sum_{s=1}^K
\frac{\binom Ks^2}{\binom Ns}
\big[1+\chi^2(M_s\|Q_s)\big]\Sigma_s,
\quad N=Bm.
}
\tag{16}
\]

而 IID-all 对应系数中没有 χ² 项。证明只需将 \(c_\alpha=\binom KsM_s(\alpha)\) 代入 (15)，并用

\[
\sum_\alpha M_s(\alpha)^2/Q_s(\alpha)=1+\chi^2(M_s\|Q_s).
\]

s=1 的额外代价为零；高阶代价一般为正。B=4,m=2 时，s=1,2,3,4 的 χ² 分别为 0、0.0208333、0.09375、0.3671875。

因此，奖励计数看起来“更均衡”不是充分胜负判据。固定标签平衡与恢复 IID 虚拟组计数之间存在高阶代价。若没有有用的分层信息，这个代价没有收益抵消。

### 5.4 本轮实际有限枚举

统一 K=4,B=4,m=2,N=8、epsilon=1e-6；score 来自明确 categorical logits 策略。表内是完整参数向量协方差 trace 的比值。

| 有限概率模型 | Strat-full / Strat-cross | Strat-full / IID-all |
|---|---:|---:|
| 两层成功率 0.2 / 0.8 | 0.872940 | 0.417683 |
| 完全无效层，成功率 0.5 / 0.5 | 0.869997 | 1.021660 |
| 两层成功率 0.1 / 0.6 | 0.959984 | 0.650303 |
| 四类别身份分层，层成功率都 0.5 | 0.973952 | 0.560363 |
| 四类别按奖励分层 | 1.000000 | 0.789500 |

两种四类别实例分别完整枚举 65,536 个实际银行。K=2 的 q=(0.2,0.8) 和 q=(0.5,0.5) 复核均得到新/旧方差比 0.75，吻合 Codex 报告；无效层的前者/强 IID 比为 1.125。

表中没有自然模型的通过率或效果预测。它展示了严格改善、相等及输给强 IID 的三种可能。

---

## 6. 不增加虚拟模型计算的精确权重

### 6.1 二值奖励：多项式计数加层内超几何

对固定银行，层 j 的成功数为 C_j。考虑自身来自层 j、奖励 r。移除自身后，层 ℓ 可用总体大小及成功数为

\[
N_\ell=B-\mathbf1_{\ell=j},
\quad C'_\ell=C_\ell-r\mathbf1_{\ell=j}.
\]

K−1 个伙伴的层计数 t 服从 Multinomial(K−1;1/m,...,1/m)。给定 t，层内成功数

\[
M_\ell\sim\operatorname{Hypergeom}(N_\ell,C'_\ell,t_\ell),
\]

条件下不同层独立。将 M_ℓ 卷积，得到伙伴总成功数 π_{j,r}(n)。于是

\[
\widetilde A_{j,r}=\sum_n\pi_{j,r}(n)a(r,n),
\qquad
\boxed{T_{full}=\frac1{Bm}\sum_{b,j}\widetilde A_{j,r_{bj}}S_{bj}.}
\tag{17}
\]

其中 a 沿用 v5 的二值 GRPO 优势。每个实际样本被虚拟组选中的概率为 K/(Bm)，组核平均含 1/K，所以基础系数是 1/(Bm)。

二值情况下只有至多 2m 种不同优势权重；m=2 时至多四种。不能因此把不同回答的 score 合并成同一个 score。

### 6.2 可直接实现的生成函数

对每个 ℓ 构造

\[
F_\ell(t,z)=\sum_{a=0}^{K-1}
\frac{t^a}{m^a a!}
\sum_h\Pr(M_\ell=h\mid t_\ell=a)z^h.
\]

则

\[
\pi_{j,r}(n)=(K-1)![t^{K-1}z^n]\prod_\ell F_\ell(t,z).
\tag{18}
\]

DP 状态是已选数量、成功数。`full_stratified_weights` 已实现，返回 B×m 权重。`brute_full_weights` 独立枚举 composition 和层内子集核验。支持 B=K，不要求 B 大于 K；必须 B≥K。

### 6.3 网格奖励与 clipping

网格奖励写为整数 v∈{0,...,L}，真实奖励 v/L。每层移除 focal 后，用

\[
\prod_i(1+t z^{v_i}w^{v_i^2})
\]

得到层内子集的 `(数量,奖励和,平方和)` 计数，除以层内组合数，再按式 (18) 的 \(m^{-a}/a!\) 因子跨层卷积。

标准差用精确整数判据 \(K\sum v_i^2-(\sum v_i)^2\)；用整数单位计算优势时 epsilon 也乘 L。全同奖励组显式为零。

同包 `full_stratified_grid_weights` 返回平均 A、A⁺、A⁻，不对真实连续奖励进行偷偷量化。状态数随 K 和网格大小增长，不能许诺任意连续 reward 都便宜。

对固定 ratio ρ，有

\[
E\min(\rho A,\operatorname{clip}(\rho)A)
=E[A^+]\min(\rho,\operatorname{clip}(\rho))+
 E[A^-]\max(\rho,\operatorname{clip}(\rho)).
\tag{19}
\]

这在当前固定银行上精确恢复虚拟组 clipped loss。一般网格奖励不能先平均 A 再直接 clipping；二值奖励下自身优势符号固定，均值权重足够。

本轮对网格 DP、完全枚举、clipped loss 及远离 kink 的 ratio 导数完成核验。该结果不把随机参数经过多轮训练后的选择效应变成无偏，也不把动态 Adam 或全局梯度裁剪变成线性映射。

---

## 7. 自然分层应该测什么

### 7.1 影响函数仍然是正确对象

沿用 v5 的

\[
\psi(y)=E[H(y,Y_2,...,Y_K)]-G_K.
\]

在固定 K,m、B 增大时，Strat-full 与 IID-all 的一阶协方差差仍为

\[
\frac{K^2}{N}\operatorname{Cov}_J(E_{P_J}\psi)+O(N^{-2}).
\tag{20}
\]

完整分层修复改变高阶系数，保留这一阶结构。B=4/8 时不能丢掉高阶项后自动判断胜负。

### 7.2 早期格式分层可能没有目标相关信息

二值 GRPO 的 u_r、t_r、p、g 定义沿用 v5 §7：

\[
\psi(y)=\frac{u_r}{K}S(y)+\frac{K-1}{K}t_rg-G_K.
\]

若所有层成功率相同 q_j=p，利用 \((1-p)u_0+pu_1=0\)，可以化为

\[
\boxed{
E_{P_j}\psi=
\frac{u_1-u_0}{K}
\left(E_{P_j}[(r-p)S]-g\right).
}
\tag{21}
\]

所以，层间 reward rate 相同并不排除收益；需要检查 reward-centered score 是否不同。反过来，如果分层主要改变开头格式，而各层 q_j 相同且 \(E_{P_j}[(r-p)S]=g\)，则一阶收益严格为零。

这是一条可检验的条件预测，不能从“首 token 是格式”直接判定自然模型满足它。

### 7.3 校准中避免平方噪声制造收益

m=2 时，层间一阶能量在读出 L 下为

\[
D_1^L=\tfrac14\|L(\mu_1-\mu_2)\|^2,
\quad \mu_j=E_{P_j}\psi.
\]

直接对小样本层均值差平方，必然混入估计方差。可用两份独立校准重复 A/B：

\[
\widehat D_1^L=\tfrac14
\langle L(\hat\mu_1^A-\hat\mu_2^A),
L(\hat\mu_1^B-\hat\mu_2^B)\rangle.
\tag{22}
\]

条件于准确且已冻结的 ψ 系数，它对相应平方能量无偏；有限结果可为负，不能截成零后宣称无偏。若 p、g 本身用 plug-in 估计，仍需独立校准并传播系数误差。不要在首轮堆叠巨大校准任务；先用重复 macro-bank 测清方法方差，再在完整阶段独立验证结构预测。

固定 coupling onset、token 排序和采样层数。在 Dev 上最多选择一个预先声明的配置并冻结；不能在确认银行上搜索对齐最好的分层。

---

## 8. 参数空间、精度和成本

### 8.1 LoRA 的测量合同

LoRA 可以降低回放成本，但会改变所测梯度空间。所有实验臂共享相同 target modules、rank、α、参数初始化、checkpoint 和优化器映射。

PEFT 默认初始化通常是 A 随机、B=0。[R5] 对 \(W=W_0+cBA\)，初始时

\[
\nabla_A\log\pi=cB^\top\nabla_W\log\pi=0.
\]

因此初始测量只看到部分切空间。不能把 A 的零梯度当作模型没学习，也不能让各臂随机选不同 A 后比较方差。可在初始状态测一轮，再在**共同 IID 短训练得到的同一 checkpoint**复核；不能各臂训练到不同位置后继续用“同一点方差”解释。

旧 v4 的全参数 d 不能直接投给新 LoRA 模型后称为同一方向。旧缺失投影沿旧合同补；新主实验另记参数空间。

### 8.2 浮点与 sampler

独立分层需要每层私有 fresh randomness，不用共同 shift；每条回答独立前缀，不复制一个生成前缀给全部样本。固定 reward 直接随回答重用，随机 judge 还要记录其独立性。

沿用 v3 条件区间引擎，在未 release 时用当前条件区间与 token CDF 的交集采样；score 使用原始 token 概率。CDF 累积、归一化和区间更新用足够精度，记录残余质量误差。仅因为区间窄不能宣布 release。

CPU 有理数例子成立，不等于 BF16/FP32 的生成引擎成立。采样与 teacher forcing 的 dropout、temperature、top-p/top-k、padding、EOS/截断、logp reduction 必须相同。对旧 full-FP32 与新混合精度路径分别核验；少量数值不一致需报告量级，不能用“同模型”代替概率检查。

遇到数值异常不能悄悄改为另一种 sampler、丢弃样本或补样，然后继续声称精确边际。修复或明确判为该实现合同未通过。

### 8.3 真正节省开销的位置

二值 Strat-full 与 IID-all 都将虚拟组变成实际回答权重；不重复生成、评分或为每个虚拟组反传。

但在冻结梯度审计中，多个不同 arms 仍需不同加权梯度/方向导数，不能声称它们同时免费。主重复只保存 `Strat-full` 和 `IID-all`，旧 cross 的核验在小子集完成。每个 macro-bank 保留足以计算预先冻结度量下方差的统计量，不长期存每条全参数 score。

用固定 LoRA 参数空间时，可逐 macro-bank 保存梯度向量并 CPU 累积均值/二阶范数；若空间仍大，用预先固定功能方向或固定随机投影。随机投影只能支持其已验证的度量精度，不能冒充整个协方差矩阵的 Loewner 顺序实证。

---

## 9. 修订 v5 的统计与投入标准

### 9.1 两种主要比较

同一 K、同一生成数 N、同一 prompt/checkpoint/参数空间，比较 `Strat-full` 与 `IID-all` 的原始梯度方差。核验目标保持使用理论、CPU 枚举和采样合同；两份不同自然银行的平均梯度不应被要求逐数相同。

完整阶段加入同总预算更多 prompt。例如 N=8、K=4，比较每题 8 回答与两题各 4 回答。对线性 token 成本近似、总回答预算 T，后者可能通过降低题间梯度方差获胜。

记题间方差 V_x=Var_x G_K(x)，题内方差分别 V_{S,N}、V_{I,K}。同题大量采样优于更多题的必要比较为

\[
N(V_x+E V_{S,N})<K(V_x+E V_{I,K}).
\tag{23}
\]

真实成本不线性时，用每个更新的实测成本替代 N、K。不能仅在选定 prompt 上降低方差就声称全训练效率提高。

### 9.2 首轮筛选不应要求小样本给出过窄的方差置信区间

旧方案每题 8 个独立重复，虽然足以看巨大效应，但很难可靠确认 15% 的方差收益。

仅作规划近似：若每题 scalar projection 近似高斯、16 题×每题 8 重复、两臂独立且结构同质，log 方差比的标准误约

\[
\sqrt{4/[16(8-1)]}\approx0.189,
\]

而 \(|\log0.85|\approx0.163\)。真实高维、异质和近零信号会改变功效，不能把这个近似当作实际 CI。它说明原先“首轮上置信限低于 0.85”的要求可能远超预算支持的精度。

**修订：**Dev 冻结一个有实践意义的效应阈值；首轮估计成本、方差比及其不确定性，明确分成支持继续、排除有用收益、信息不足。信息不足时只允许根据实测方差制定一次定量扩样，不能无限换配置寻找阳性。v5 尚未运行，可以在看其效果前更正此统计合同。

### 9.3 时间归一化判据

固定目标、相同独立更新平均条件下，设 q 为部署时 `Strat-full/IID-all` 的实测单次成本比，ρ 为相同度量下方差比。单位总时间的近似方差比是

\[
\rho_{time}=q\rho.
\tag{24}
\]

若 qρ≥1，方差收益被成本抵消。报告 CPU 预处理、生成、评分、回放、模型加载的成本，并区分一次性审计开销和每次训练都会发生的开销。

此判据仅涉及固定点梯度估计效率，不足以直接推出非线性优化的训练收益。后者需要多种子在线证据。

---

## 10. 给 Codex 的执行顺序与停止条件

### 阶段 A：CPU 与现有资产，先完成

- 从已封存回执汇总实际模型、参数空间、bank 完成数、方向文件和已耗预算，不根据本文件猜测。
- 重跑本包 CPU 脚本；独立比对 Strat-full composition 枚举与 DP。检查 IID-all 保持 K，而不是把 K 改成 N。
- 增加缺失投影状态；从旧银行定位 `Mbpp/474:0:7` 和旧 d，不填零。
- 计算现有 v4 评分矩阵的零差分结构，保留全部组的分母。
- 修改 v5 主权重实现为 Strat-full，检查生成时的跨层独立性；旧 cross 不删除，限定使用条件。

**通过标准：**均值/权重/正负分解/重配对恒等式误差小于预设 CPU 容差；相关块反例必须暴露误用偏差。不能只验证阳性实例。

### 阶段 B：一次有边界的 v4 修复

下一次确有必要加载旧模型时，补 z_f，输出式 (2)(3) 和旧原始估计。按 §3 只测必要的原点重复、纯差中心差分和较小步；如需定位混合曲率，再加中点或正负步。

**停止条件：**缺少旧 checkpoint 或 d 则明确不可重建；数值底线高于欲测变化则标记不可分辨。不得重新生成答案冒充旧银行，也不因估计转正就升级为新确认。一次修复结束后封存 v4 的证据边界。

### 阶段 C：v5 小成本预检查

主模型和任务暂沿用 v5 的候选（Qwen2.5-1.5B-Instruct、SVAMP、K=4,m=2,B=4、LoRA），具体 revision 从环境冻结。这些是配置，不是方法前提。

先只在独立 Dev 上检查：生成与评分一致、任务存在可用奖励变化、采样 strata 的真实独立流、release 数值、LoRA 非零梯度子空间和实测吞吐。若任务几乎全对/全错，预先定义并报告筛选后的目标题目分布；确认题不能按方法获益选择。

**停止条件：**概率合同不通过、真实成本不能支持最少独立重复、或目标度量的信号不足，则先解决该具体问题。不要同时扩大模型、变换任务、增加 K 并启动长训练。

### 阶段 D：两臂冻结统计

对同一组预先固定的 prompt，两种 sampler 各采相互独立的 macro-banks。每个 macro-bank 有 B×m 条实际回答；跨宏批随机流独立。

主臂为 `IID-all` 与 `Strat-full`。旧 cross 的 DP 在相同分层银行上计算，少量 backward 核对其支配预期；它不成为每个重复都跑的昂贵第三主臂。均值/方差计算中保留全同奖励组。

输出：题内方差比及不确定性、功能投影、梯度均值差的统计范围、严格相同 K/N/参数空间、实际 qρ 成本比、异常率。分别保留 prompt 和 macro-bank 层级，不用虚拟组数充当样本量。

**继续条件：**在正确合同下，有可分辨且足以抵消真实成本的收益；或者实测方差支持一次预算内的明确扩样。**停止条件：**有足够精度排除实践收益、成本吃掉改善、或只能赢旧弱估计器而无法赢强 IID。

### 阶段 E：方法贡献确认

只有 D 通过，才进行共同中期 checkpoint、同预算更多 prompt 和多种子在线训练。使用共同 checkpoint 检查 LoRA 初始化的偶然几何；online 中记录 K 固定、ratio/clipping/reduction 和优化器差异。

当前没有依据给出 solid accept 概率。能支持方法论文的证据应当包括：相关采样带来的信息结构、可计算的目标保持权重、强 IID 参照上的净收益、真实预算下的训练改善。

---

## 11. 先行工作与贡献边界

| 对象 | 本轮判断 |
|---|---|
| 乘积经验测度、虚拟重组、Rao–Blackwell 化 | 经典统计方法；Product-form estimators 已明确联系 generalized U-statistics。[R2] |
| 完整分层 U 形式与分层方差理论 | 有直接的 1973 分层 U-statistic 文献记录；本轮未取得全文，不能宣布排除了等价定理。[R3] |
| 相关采样使 RLOO baseline 有偏 | QuasiMoTTo 已明确讨论；不能作为本项目新发现。[R4] |
| 精确二值/网格/正负 clipping 权重 | 本项目已给可执行实现和验证；是否足够独立 novelty 仍需专门比较，不能凭实现成功断言 |
| 与 IID-all 的失败边界、自然 ψ 分层预测、时间收益 | 可用于组织论文的实际问题；理论公式本身可能是经典工具的专门化，价值需要真实证据 |

**论文主张不能建立在“新版公式比上一版更多”上。**这次修改修复了默认算法的统计浪费，也提供了更强的否定对照；它提高了实验的可信度和候选方法的效率，尚没有产生真实模型成功结果。

---

## 12. 本轮实际检查与交付文件

运行：

```bash
python verify_v5_1.py
```

Python≥3.10，依赖 NumPy。只使用 CPU。脚本输出 `verify_v5_1_results.json`。

实际完成：

- K=2 Codex 重配对例子复核；K=4 的 Bernoulli 和四类别策略完整枚举。
- 重配对平均与完整层内 U 形式一致。
- DP 权重与 composition/子集完全枚举一致。
- 均值保持、协方差式 (15)、条件方差式 (12) 核验。
- 无效分层 χ² 边界和非法跨层依赖反例。
- RLOO 与固定 baseline 方差式核验。
- 网格奖励 DP、正负权重、clipped loss 及非 kink 导数一致。
- 双银行内积方差和多项式函数的共同曲率恒等式。

全部等式检查最大绝对误差 **1.4682699500667695e−14**。CPU assertion 容差为 3e−11。没有机器形式化证明；有限例子不代替一般数学证明，不支持自然 LLM 效应量。

没有测量 z_f，没有查看原仓库，没有运行 GPU。两种四类别例子各枚举 65,536 个可能银行；这些不是实际生成的 LLM 回答。

## 参考材料与核验范围

**[F1]** 用户提供 `ANALYSIS.md`，132 行。数值来源以该报告为准：§2 对应行19–48；§3 行50–67；§4 行69–74；重配对提案 §6 行84–112。

**[F2]** `dependent_feedback_v4_theory_audit_and_codex_plan.md`。本轮继承其固定目标、epsilon、bank 与原始 score 约定；对实验测量作上述修订。

**[F3]** `dependent_group_v5_upper_bound_supplement.md`。重点复核 §3–8 的目标、独立分层块、方差分解、影响函数与权重；默认主估计器按本文修订。

**[F4]** `dependent_rollout_research_dossier_v3(1).md`。继承独立算术分层、原始边际 score 和 fresh-randomness 条件区间引擎。

**[R1]** Greensmith, Bartlett, Baxter. *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning*. JMLR 5, 2004。核验期刊原始页面；本文 baseline 公式另给自含推导。  
`https://www.jmlr.org/papers/v5/greensmith04a.html`

**[R2]** Kuntz, Crucinio, Johansen. *Product-form estimators: exploiting independence to scale up Monte Carlo*. arXiv:2102.11575v3，2021-11-01。核验 HTML 的文献关系、无偏性、方差、计算开销讨论。  
`https://arxiv.org/html/2102.11575v3`

**[R3]** Taga, Yasushi; Wakimoto, Kazumasa; Yanagawa, Takashi. *Generalized U statistics for stratified random samples*. 1973，DOI 10.1080/03461238.1973.10414969。核验出版记录；本轮获取全文失败，未进行逐定理等价性比较。  
`https://www.tandfonline.com/doi/abs/10.1080/03461238.1973.10414969`

**[R4]** *QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling*. arXiv:2607.01179v1，2026。核验 HTML 中独立分层、lattice、相关 RLOO baseline 与修正讨论。  
`https://arxiv.org/html/2607.01179v1`

**[R5]** Hugging Face PEFT 官方 LoRA 文档，访问日期 2026-09-07。核验默认 A 随机、B 零初始化；实际安装版本必须由 Codex 锁定。  
`https://huggingface.co/docs/peft/developer_guides/lora`

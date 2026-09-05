# What Dependent Rollouts Teach a Policy
## A theory-first research dossier for a possible ICLR 2027 paper

**Working subtitle:** *From coverage to contrast selection in group-relative reinforcement learning*  
**Literature audit date:** September 4, 2026  
**Decision:** **CONDITIONAL GO — authorize a bounded pilot, not an unconditional paper commitment.**

> **Central claim to investigate.** A rollout sampler determines not only which answers are found, but which successful and unsuccessful answers are contrasted during learning. The resulting update can contain prompt reweighting, reward-direction preconditioning, and additional score-dependent selection. These effects are different from coverage and estimator variance, and should be separated experimentally.

**Evidence status.** The finite examples, algebraic identities, and spectral bound below were checked by CPU enumeration or numerical linear algebra. The document contains proofs, not just proposed theorem statements. No pretrained-language-model experiment has been run for this dossier. Statements about practical effect sizes, throughput, and downstream gains are hypotheses or planning assumptions, not results. Novelty assessments are scoped to the primary sources inspected; they are not a priority guarantee.

---

## 1. Executive assessment: what survives, and what must change

This direction can support a substantive paper, but the earlier framing is insufficient by itself.

The paper should **not** be sold as “dependent rollouts bias RLOO,” “pairwise gradients need pairwise corrections,” “better coverage need not mean better training,” or “cross-group baselines remove dependence-induced baseline bias.” Those claims are already known or immediate consequences of existing theory. QuasiMoTTo explicitly discusses dependence-induced baseline bias and hypothesizes that it explains differences between its samplers. CARMS supplies pairwise importance corrections; PAIR addresses inclusion-weighted pair estimation. [R1–R4]

The stronger program has three connected parts:

1. **Characterization:** identify precisely when dependence acts as a reward-independent positive preconditioner, and when it introduces a component that cannot be determined from the original gradient alone.
2. **Sampler-specific predictions:** derive the actual operators for stratified and lattice sampling, including a quantitative safety boundary for stratification and a prediction of its expected update from independently sampled trajectories.
3. **Causal identification:** change the assignment of successful and unsuccessful trajectories to group positions while preserving the complete success-count profile and the stored trajectories. Test whether this changes the mean learning direction and predicts held-out learning consequences.

The most important improvement over the earlier plan is **count-preserving rewiring**. A within-group versus independent-baseline comparison changes several things at once. Rewiring provides an additional intervention that holds every binary reward-only group statistic fixed. It therefore distinguishes “more informative-looking groups” from “different examples receiving the learning signal.”

A second improvement is an **IID-only prediction for stratified sampling**. Its expected RLOO update can be calculated from the score–reward covariance between sampling strata, without observing any stratified-policy-gradient estimate. For binary normalized GRPO, a related finite calculation uses a Poisson-binomial distribution. This is a stronger empirical target than reconstructing the gradient from the same data used to measure it.

A third improvement is a correction to the interpretation of the geometry. A non-collinear update is not automatically harmful, and it is not automatically evidence against preconditioning. A claim about reward-independent preconditioning requires multiple reward directions or a structural characterization. A claim about harm requires an objective-specific, optimizer-specific, independently evaluated consequence.

**Recommended form of the paper:** theory + predictive causal analysis. A new optimizer or sampler is unnecessary. Cross-group baselines should primarily be an identification instrument, not a method invented to complete the story.

---

## 2. Exact estimand and assumptions

### 2.1 Begin with the on-policy score-point problem

For a fixed prompt $x$, let

$$
p_\theta(y)=p_\theta(y\mid x),\qquad
s_\theta(y)=\nabla_\theta\log p_\theta(y),\qquad
J(\theta)=\mathbb E_{p_\theta}[r(Y)].
$$

The verifier reward $r$ is parameter-independent. Initially use a finite output space, positive probabilities, and a differentiable policy. A bounded generation horizon gives a finite, although enormous, output space for an LLM. A response reaching the cap remains an outcome; do not silently discard it.

Generate a group

$$
(Y_1,\ldots,Y_K)\sim Q_\theta,\qquad Y_i\sim p_\theta.
$$

Assume exchangeability. Randomly permuting the members of a marginal-preserving group produces the required symmetrization without changing a permutation-invariant estimator.

The clean estimator is sequence-level RLOO:

$$
\widehat g_Q
=\frac1K\sum_{i=1}^K
\left(r_i-\frac1{K-1}\sum_{j\ne i}r_j\right)s_i.
\tag{1}
$$

The score is the **sum**, not the length-average, of response-token log-probability gradients. The intended gradient is

$$g=\nabla J=\mathbb E_p[s(Y)r(Y)].$$

Everything is evaluated at the generating checkpoint. Rewards and advantages are detached when differentiating the sampled surrogate. The joint sampler may depend on $\theta$, but the estimator in (1) uses marginal scores, not the score of $Q_\theta$.

### 2.2 What is excluded from the primary theorem

Exclude PPO clipping, multiple off-policy epochs, outcome-dependent rollout selection, trajectory-length normalization, and random reward-standard-deviation normalization initially. These are distinct sources of changed estimands. Current implementations differ along these axes, so the experimental loss must be specified mathematically rather than called merely “GRPO.” [R3, R17]

Section 7 gives an exact extension for binary group-standardized advantages. It does **not** establish that arbitrary clipped, multi-epoch, token-normalized implementations have the same expected update.

Do not fold a parameter-dependent KL reward into $r$ and silently apply the parameter-independent-reward theorem. Either omit KL in the primary diagnostic or add a separately specified KL-gradient term equally to all arms.

### 2.3 Prompts share parameters

For a prompt distribution $\mu$, all results can be written in the joint space

$$P(dx,dy)=\mu(dx)p_\theta(dy\mid x).$$

Use conditionally centered rewards

$$r_0(x,y)=r(x,y)-\mathbb E[r\mid x]$$

and functions satisfying $\mathbb E[f\mid x]=0$. The partner operator acts within each prompt, while the score subspace is formed by the **shared** model parameters across prompts.

This distinction matters. A positive scalar change separately at every prompt,

$$g_Q(x)=c_xg(x),\qquad c_x\ge0,$$

can rotate or even oppose the aggregate update

$$\mathbb E_x[c_xg(x)]$$

when prompt gradients conflict. It is prompt reweighting, not necessarily a harmless global learning-rate change.

---

## 3. A precise separation of the scientific objects

| Object | Definition or measurement | What it does not establish |
|---|---|---|
| Marginal policy | Every $Y_i\sim p_\theta$ | Unbiasedness of a group-baseline estimator |
| Coverage | $\Pr_Q(\max_i r_i=1)$ for binary rewards | Direction or variance of a policy-gradient update |
| Reward-count profile | Distribution of $N=\sum_i r_i$ | Which successful or failed trajectories receive large advantages |
| Mean update | $g_Q=\mathbb E_Q[\widehat g_Q]$ | Quality of a finite noisy training step |
| Estimator covariance | $\operatorname{Cov}_Q(\widehat g_Q)$ | Alignment of its mean with the objective |
| Positive rescaling | $g_Q=cg$, $c>0$, under a declared metric/parameterization | Same finite-step behavior under all optimizers |
| Reward-independent preconditioning | One fixed operator maps the original gradient to the coupled update for an entire reward class | Something inferable from one acute pair of vectors |
| Reversal | $g^TMg_Q<0$ for a specified fixed update map $M\succeq0$ | Global non-convergence, or harm under every optimizer |
| Learning consequence | Independent reward change or held-out behavior after an intervention | A mechanism unless mean, variance, and step size were separated |

A low cosine is a diagnostic, not a conclusion. In particular, a positive-definite preconditioner can rotate a vector. Conversely, for a single nonzero pair $g,h$ with $g^Th>0$, some symmetric positive-definite matrix can map $g$ to $h$. Therefore, fitting such a matrix after observing one reward does not explain anything.

At a frozen checkpoint and for a fixed preconditioner $M$, a small random step has the expansion

$$
\mathbb E[J(\theta+\eta M\widehat g)]
=J(\theta)+\eta g^TM\mu
+\frac{\eta^2}{2}\left[
\mu^TM\nabla^2J\,M\mu
+\operatorname{tr}(M\nabla^2J\,M\Sigma)
\right]+O(\eta^3),
\tag{2}
$$

under suitable bounded-moment and smoothness assumptions. Here $\mu=\mathbb E\widehat g$ and $\Sigma=\operatorname{Cov}(\widehat g)$.

Thus mean direction enters at first order, while curvature-mediated variance enters at second order. This motivates averaged-gradient, small-step interventions. It does not imply that variance is unimportant over long training runs. Baseline effects beyond a simple variance story are already part of classical policy-optimization research. [R10, R11]

For adaptive Adam updates, generally

$$\mathbb E[M(\widehat g)\widehat g]\ne M(\mathbb E\widehat g)\mathbb E\widehat g.$$

Use SGD or a preconditioner frozen from independent history for mechanism experiments. Treat ordinary AdamW training as downstream validation, not a direct implementation of the mean-field theorem.

---

## 4. General theory: dependence defines a contrast operator

### 4.1 The known pairwise identity, expressed as an operator

Define the partner operator

$$
(Tf)(y)=\mathbb E[f(Y_2)\mid Y_1=y],\qquad L=I-T.
$$

Then

$$
\boxed{g_Q=\mathbb E_p[s(Y)Lr(Y)].}
\tag{3}
$$

Equivalently,

$$
g_Q=\frac12\mathbb E_Q[(s_1-s_2)(r_1-r_2)].
\tag{4}
$$

**Proof.** The diagonal term in (1) has expectation $\mathbb E[sr]$. Exchangeability makes every off-diagonal term equal to $\mathbb E[s_1r_2]=\mathbb E[sTr]$. Symmetrizing gives (4).

The pairwise identity and its use for importance correction are established starting points, not proposed novelty. [R1–R4]

### 4.2 Classical structure of the operator

Let $\langle f,h\rangle_p=\mathbb E_p[fh]$. Exchangeability implies

$$\langle f,Th\rangle_p=\langle Tf,h\rangle_p.$$

Conditional expectation makes $T$ a contraction, and $T1=1$. Therefore

$$
\langle f,Lf\rangle_p
=\frac12\mathbb E[(f(Y_1)-f(Y_2))^2]\ge0.
\tag{5}
$$

This is the standard Dirichlet form of a reversible Markov kernel. It should be explicitly credited to classical probability, rather than presented as a newly discovered mathematical object. [R7]

Finite group extendibility gives a tighter bound than arbitrary pairwise reversibility:

$$
-\frac1{K-1}I\preceq T\preceq I,
\qquad
0\preceq L\preceq\frac K{K-1}I
\tag{6}
$$

on centered functions. The lower bound follows from

$$
0\le\operatorname{Var}\left(\sum_i f(Y_i)\right)
=K\|f\|_p^2+K(K-1)\langle f,Tf\rangle_p.
$$

An arbitrary reversible pair kernel need not satisfy the lower bound for a specified $K>2$. Do not use a convenient pair matrix as a purported realizable $K$-rollout sampler without checking extendibility.

### 4.3 Relation to sample-mean covariance

For $\bar f=K^{-1}\sum_i f(Y_i)$,

$$
\operatorname{Cov}(\bar f,\bar h)
=\frac1K\langle f,h\rangle_p
+\frac{K-1}{K}\langle f,Th\rangle_p.
\tag{7}
$$

Consequently,

$$
\boxed{
g_Q=\frac K{K-1}\left(g-\operatorname{Cov}(\bar s,\bar r)\right).
}
\tag{8}
$$

This is a covariance identity, not a new general law of Monte Carlo. Its useful implication is that lowering **reward** sample-mean variance controls one quadratic form, while learning depends on a score–reward cross-covariance.

For RLOO, the pair law determines the mean update. Coverage can depend on higher-order joint structure, and gradient-estimator covariance generally involves interactions among up to four group members. The three quantities are not interchangeable. [R3]

### 4.4 Null modes: what a sampler cannot contrast

If $Lf=0$, then $f(Y_1)=f(Y_2)$ almost surely under the pair law. A reward component that is constant across every possible pair receives no RLOO signal.

For a finite pair graph, the null space consists of functions constant on its connected components. Marginal support can be complete while this contrast graph is disconnected. Thus “every answer can be sampled” is weaker than “every reward distinction can drive group-relative learning.”

This is a useful structural interpretation of a classical graph-Laplacian fact. It becomes a paper contribution only if sampler-specific null or near-null modes explain a measurable phenomenon.

---

## 5. The strongest general characterization: compatibility with the score subspace

### 5.1 Geometry

Let

$$
(Sa)(y)=s(y)^Ta,
\qquad
F=S^*S=\mathbb E[ss^T].
$$

Let $\mathcal V=\operatorname{range}(S)$ be the score subspace in centered $L_2(p)$, and let

$$\Pi=SF^\dagger S^*$$

be its orthogonal projection. Finite-dimensional rank deficiency is handled by the pseudoinverse on the identifiable parameter range.

Write

$$r_0=v+e,\qquad v=\Pi r_0,\qquad e\perp\mathcal V.$$

The original natural-gradient reward slope is $\|v\|_p^2$. The slope of a natural-gradient-preconditioned coupled update is

$$
D_Q(r)=g^TF^\dagger g_Q
=\langle v,Lr_0\rangle_p
=\underbrace{\langle v,Lv\rangle_p}_{\ge0}
-\underbrace{\langle v,Te\rangle_p}_{\text{residual interference}}.
\tag{9}
$$

Compatible score-space projections and natural policy gradients are established concepts. The proposed application concerns what a sampling dependence operator does to that projection. [R8, R9]

### 5.2 Characterization theorem

**Theorem 1 — universal local compatibility.** For a finite positive-support policy and a fixed exchangeable sampler at a fixed checkpoint, the following statements are equivalent:

- $D_Q(r)\ge0$ for every parameter-independent reward $r$.
- $T\mathcal V\subseteq\mathcal V$.
- $\Pi T(I-\Pi)=0$, equivalently $[\Pi,T]=0$.
- There exists a reward-independent self-adjoint positive-semidefinite operator $A$ on $\mathcal V$ such that
  $$\Pi Lr=A\Pi r\quad\text{for every }r.$$

In that case $A=\Pi L\Pi|_{\mathcal V}$.

**Proof.** In the decomposition $\mathcal V\oplus\mathcal V^\perp$, write

$$L=\begin{pmatrix}A&B\\B^*&C\end{pmatrix}.$$

Then $D_Q(v+e)=\langle v,Av\rangle+\langle v,Be\rangle$. If $B=0$, positivity of $L$ gives $D_Q\ge0$. If $B\ne0$, choose $v,e$ with nonzero cross term and scale the sign and magnitude of $e$ to make $D_Q<0$. On finite support, an affine rescaling puts the resulting reward in $[0,1]$ without changing the sign. Self-adjointness makes the off-diagonal blocks vanish together, establishing invariance and commutation. The factorization statement is equivalent to independence from $e$. $\square$

**What this adds conceptually.** Dependence is a genuine reward-independent preconditioner exactly when the update is determined by the learnable reward projection alone. Otherwise two rewards with identical original policy gradients can generate different coupled updates; one can have zero original gradient but a nonzero coupled update.

**What this does not add mathematically.** The proof is a short block-operator argument. It is not a new theorem of Hilbert-space geometry. Its research value must come from the specific characterization, sampler consequences, and predictive validation—not proof length or unfamiliar terminology.

### 5.3 Stronger and weaker notions are different

Universal unbiasedness requires

$$\Pi T=0.$$

Universal scalar rescaling by $c$ requires

$$\Pi T=(1-c)\Pi.$$

Universal safe natural preconditioning requires only invariance of $\mathcal V$.

For one particular reward, safety only requires

$$\langle v,Lv\rangle\ge\langle v,Te\rangle.$$

Failure of universal compatibility does not imply that the task reward is harmful. Conversely, a safe update for one reward does not establish compatibility for other rewards.

### 5.4 Whitened finite-dimensional representation

On the identifiable parameter range, let $\phi=F^{\dagger/2}s$, so $\mathbb E[\phi\phi^T]=I$. Define

$$
h=\mathbb E[\phi r_0],\quad
C_Q=\mathbb E[\phi(Y_1)\phi(Y_2)^T],\quad
b_Q=\mathbb E[\phi(Y_1)e(Y_2)].
$$

Then

$$
\boxed{h_Q=(I-C_Q)h-b_Q.}
\tag{10}
$$

The first term is positive-semidefinite preconditioning in the whitened score coordinates. The second is residual interference. This decomposition separates two mechanisms that a raw gradient cosine cannot distinguish.

### 5.5 Essential limits of the theorem

**Natural-gradient specificity.** A fixed general optimizer $M$ gives slope $g^TMg_Q$. Even when $b_Q=0$, noncommutation between the optimizer geometry and the sampler-induced preconditioner can matter. Do not advertise Theorem 1 as an AdamW ascent theorem.

**Saturated-policy control.** If the score subspace contains every centered reward function, then $e=0$ and natural-gradient reversal is impossible. Zero progress can still occur if $Lr=0$.

**Finite-answer LLM trap.** A pretrained model with many parameters and only a handful of constrained answer choices may already be locally saturated on those choices. Such an experiment may be useful for verifying the positive case but cannot establish the proposed residual-interference mechanism in unrestricted generation. Artificially restricting to one adapter direction can manufacture the missing dimension. Label it a controlled mathematical intervention, not primary LLM evidence.

**No global objective claim.** $L_\theta r$ is a local effective reward in the score update. Because $L_\theta$ changes with the policy and the sampled surrogate stops gradients through its construction, it need not be the gradient of a single scalar objective. Non-conservative policy-update fields are already studied elsewhere; merely exhibiting curl would not be a strong separate contribution. [R14, R15]

---

## 6. Sampler-specific theory: bounded reweighting versus filtering

The following analysis applies to a fixed measurable decoder $Y=G_\theta(U)$ with $U$ uniform on $[0,1)$. Sequence-level arithmetic decoding is one construction. Fix the token-ordering convention because it changes the decoder and therefore the coupling, even though it preserves marginal sampling probabilities. [R1]

Let $\mathcal C f=f\circ G_\theta$. This is an isometry from $L_2(p)$ into uniform-$U$ function space. Its adjoint is conditional expectation given the decoded sequence. Compression back to sequence space matters: a latent projection need not remain a projection after compression.

### 6.1 Stratified sampling

Partition $[0,1)$ into $K$ equal bins, draw one independent uniform point from each, and randomly permute group members. Let $P_B$ be conditional expectation on the bin label. On centered latent functions,

$$T_U=-\frac1{K-1}P_B.$$

Therefore, in sequence space,

$$
\boxed{L_{\rm strat}=I+\frac1{K-1}H_B,\qquad
H_B=\mathcal C^*P_B\mathcal C,\qquad 0\preceq H_B\preceq I.}
\tag{11}
$$

**Proof.** Given the bin of one randomly labeled group member, another member is uniformly distributed over the other bins. For centered $f$, its conditional mean is minus the mean in the excluded bin divided by $K-1$. Push the operator through the decoder. $\square$

Thus

$$I\preceq L_{\rm strat}\preceq\frac K{K-1}I.$$

Stratification does not remove reward-space modes in this clean setting. It adds a bounded anisotropic reweighting. That does not, by itself, guarantee ascent after projection into a restricted score subspace.

### 6.2 A directly testable decomposition

For bin $j$, define

$$\mu_{s,j}=\mathbb E[s(Y)\mid U\in B_j],\qquad
\mu_{r,j}=\mathbb E[r(Y)\mid U\in B_j].$$

Then

$$
\boxed{g_{\rm strat}=g+\frac1{K-1}\operatorname{Cov}_B(\mu_{s,B},\mu_{r,B}).}
\tag{12}
$$

Equivalently,

$$
g_{\rm strat}
=\mathbb E_B[\operatorname{Cov}(s,r\mid B)]
+\frac K{K-1}\operatorname{Cov}_B(\mu_{s,B},\mu_{r,B}).
\tag{13}
$$

This is a statistical within-/between-stratum decomposition. It is **not** the gradient of the probability of selecting a bin: bins have fixed uniform probability in latent space, while their relation to output sequences depends on the policy.

**Predictive experiment.** Estimate the bin means using independent IID trajectories, cross-fit their products, and predict $g_{\rm strat}$ before measuring it on a new stratified rollout bank. No fitted sampler-bias model is needed.

The prediction remains checkpoint-, prompt-, and decoder-specific. It does not assert that stratification always improves the objective.

### 6.3 A quantitative natural-gradient safety boundary

Let

$$\eta=\frac{\|\Pi r_0\|_p^2}{\|r_0\|_p^2}\in(0,1]$$

be the fraction of centered reward variance explainable by the score subspace. For any self-adjoint $L$ with $mI\preceq L\preceq MI$,

$$
\frac{\langle v,Lr_0\rangle}{\|v\|^2}
\ge\frac{m+M}{2}-\frac{M-m}{2\sqrt\eta}.
\tag{14}
$$

**Proof.** Normalize $\|v\|=1$, write $r_0=v+e$, and let $\kappa=\|e\|$. The symmetric rank-two form $(r_0v^*+vr_0^*)/2$ has eigenvalues $(1\pm\sqrt{1+\kappa^2})/2$. The minimum of its trace pairing with an operator in $[mI,MI]$ assigns $m$ to the positive eigendirection and $M$ to the negative one. Since $\sqrt{1+\kappa^2}=1/\sqrt\eta$, (14) follows. The bound is sharp given only these spectral constraints. $\square$

For stratification, $m=1$ and $M=K/(K-1)$, giving

$$
\boxed{
\frac{D_{\rm strat}(r)}{\|v\|^2}
\ge
\frac{2K-1-\eta^{-1/2}}{2(K-1)}.
}
\tag{15}
$$

In particular,

$$
\boxed{\eta>\frac1{(2K-1)^2}\quad\Longrightarrow\quad D_{\rm strat}(r)>0.}
\tag{16}
$$

The threshold is $1/9$ for $K=2$, $1/49$ for $K=4$, and $1/225$ for $K=8$.

**Interpretation.** Stratified sampling cannot reverse the exact natural-gradient reward direction unless the reward has sufficiently little projection into the policy's score subspace. Falling below the threshold permits reversal; it does not predict reversal.

**A measurable sufficient condition without a full Fisher inverse.** For any fixed parameter direction $d$,

$$\eta\ge\operatorname{Corr}_p(r_0,s^Td)^2.$$

Therefore,

$$|\operatorname{Corr}_p(r_0,s^Td)|>\frac1{2K-1}$$

is a population-level sufficient condition for natural-gradient ascent under stratification. Choose $d$ on discovery data and estimate the correlation on independent data. A noisy empirical correlation is not automatically a rigorous certificate; report confidence limits and the population assumptions. This condition says nothing directly about SGD or AdamW ascent.

### 6.4 Lattice sampling

Generate the orbit

$$U_i=(V+i/K)\bmod1,$$

and randomly permute it. Let

$$P_Of(u)=\frac1K\sum_{j=0}^{K-1}f((u+j/K)\bmod1).$$

This is the orthogonal projection onto functions invariant under the cyclic shift. Then

$$
\boxed{L_{\rm lattice}=\frac K{K-1}(I-H_O),\qquad
H_O=\mathcal C^*P_O\mathcal C.}
\tag{17}
$$

Unlike stratification, lattice sampling can suppress modes. A decoded reward whose lifted function is constant along each lattice orbit lies in the null space. Whether a useful LLM reward has a material near-null component is an empirical question.

This provides a structural distinction between two samplers, rather than saying one has “more bias.” It does **not** prove that lattice sampling has worse task-aligned updates for every reward or checkpoint.

### 6.5 Shared-prefix sampling is a control, not the novelty claim

If a sampled prefix $Z$ is shared and suffixes are conditionally independent, then $T=P_Z$ and $L=I-P_Z$. The estimator removes the prefix-conditioned reward component. Shared-prefix cancellation and related group-objective effects are already discussed in the literature. [R12, R13]

Use this case to verify the operator interpretation and implementation. Do not make it the main discovery, and do not confuse sampling one shared generated prefix with merely caching the same deterministic prompt prefix.

---

## 7. Binary rewards: separate count effects from example selection

This section is the practical center of the project. It avoids requiring a full LLM Fisher matrix and extends cleanly to group-standardized binary rewards.

### 7.1 Exact per-group decomposition

Let $r_i\in\{0,1\}$ and $N=\sum_i r_i$. Let $\bar s_+$ and $\bar s_-$ be the mean scores of successful and unsuccessful responses in the group. Define the product below as zero when $N=0$ or $N=K$.

For RLOO,

$$
\boxed{\widehat g_Q=a_K(N)(\bar s_+-\bar s_-),\qquad
 a_K(n)=\frac{n(K-n)}{K(K-1)}.}
\tag{18}
$$

This follows directly by collecting the positive and negative terms in (1).

Let $p_+=\Pr(r=1)$ and

$$m_+=\mathbb E[s\mid r=1],\quad m_-=\mathbb E[s\mid r=0],\quad
\Delta=m_+-m_-.$$

Then

$$g=p_+(1-p_+)\Delta.$$

Define

$$\Delta_n=\mathbb E[\bar s_+-\bar s_-\mid N=n].$$

The coupled update has the exact decomposition

$$
\boxed{
g_Q=c_Qg+\zeta_Q,
\quad c_Q=\frac{\mathbb E[a_K(N)]}{p_+(1-p_+)},
\quad \zeta_Q=\mathbb E[a_K(N)(\Delta_N-\Delta)].
}
\tag{19}
$$

For RLOO specifically,

$$c_Q=1-\frac{\operatorname{Cov}(r_1,r_2)}{p_+(1-p_+)}\ge0.$$

**Interpretation.** Counts determine a nonnegative within-prompt rescaling. The remaining term measures how success and failure **identities**, through their scores, depend on the count profile.

Under IID sampling, conditional on $N$, successful responses retain the success-conditioned policy distribution, and failures retain the failure-conditioned distribution. Thus $\Delta_n=\Delta$, $\zeta_Q=0$, and $c_Q=1$ for RLOO. Under dependent sampling this conditional-distribution property can fail.

Existing finite-group objective analyses already use success counts and success-conditioned score statistics. The new target must be their failure under dependence and the causal separation of that failure, not the observation that counts influence update magnitude. [R5, R6]

### 7.2 Exact extension to binary standardized GRPO

Consider the on-policy sequence-score estimator

$$
\widehat g_{\rm std}
=\frac1K\sum_i\frac{r_i-N/K}{d_N}s_i,
\qquad
 d_n=\sqrt{(n/K)(1-n/K)}+\epsilon,
$$

with a declared $\epsilon>0$ and population-style group standard deviation. Then

$$
\widehat g_{\rm std}
=a_K^{\rm std}(N)(\bar s_+-\bar s_-),
\qquad
 a_K^{\rm std}(n)=\frac{n(K-n)}{K^2d_n}.
\tag{20}
$$

Equation (19) holds with $a_K$ replaced by $a_K^{\rm std}$. A different standard-deviation convention changes the coefficient and must be specified.

Thus the count/identity decomposition covers binary normalized GRPO **exactly at the score point**, despite the normalizer depending on the entire group. It does not rely on pretending that normalization is independent of the action.

### 7.3 What successful and failed behaviors are actually reinforced?

For RLOO, define the conditional weights

$$
w_+(y)=\mathbb E\left[\frac{K-N}{K-1}\mid Y_1=y\right],\quad r(y)=1,
$$

$$
w_-(y)=\mathbb E\left[\frac{N}{K-1}\mid Y_1=y\right],\quad r(y)=0.
$$

Their positive and negative total masses agree; call the common mass $Z_Q$. When $Z_Q>0$,

$$g_Q=Z_Q\left(\mathbb E_{P_Q^+}[s]-\mathbb E_{P_Q^-}[s]\right),$$

where

$$P_Q^+(y)\propto p(y)r(y)w_+(y),\qquad
P_Q^-(y)\propto p(y)(1-r(y))w_-(y).$$

The sampler thereby creates implicit positive and negative training distributions. Even with the same binary verifier, some correct solution types and some incorrect solution types can receive disproportionate learning weight.

This gives an interpretable empirical target: do counts preferentially weight particular successful strategies, failure categories, or early decisions? Features must be fixed before examining their association with gradient changes. Do not substitute hand-picked examples for the full-gradient test.

### 7.4 A stronger counterexample than “bias exists”

Take four responses with

$$p=(1/4,1/4,1/4,1/4),\quad r=(1,1,0,0),\quad s=(5,-7,0,2).$$

These scores are realized at $\theta=0$ by

$$p_\theta(i)\propto p_i\exp(\theta s_i),$$

since $\mathbb E_p[s]=0$. The true gradient is $g=-1/2$.

For $K=2$, define two symmetric pair laws:

| Ordered pair | $Q_A$ probability | $Q_B$ probability |
|---|---:|---:|
| $(1,3),(3,1)$ | $1/4$ each | 0 |
| $(2,3),(3,2)$ | 0 | $1/4$ each |
| $(2,4),(4,2)$ | $1/8$ each | 0 |
| $(1,4),(4,1)$ | 0 | $1/8$ each |
| $(2,2)$ | $1/8$ | 0 |
| $(1,1)$ | 0 | $1/8$ |
| $(4,4)$ | $1/8$ | $1/8$ |

Both have the same individual marginal $p$ and exactly the same count distribution:

$$\Pr(N=0)=1/8,\quad\Pr(N=1)=3/4,\quad\Pr(N=2)=1/8.$$

Both have pass@2 $=7/8$, better than IID's $3/4$. Every binary reward-only group statistic is identical. Yet

$$\boxed{g_{Q_A}=1/8,\qquad g_{Q_B}=-11/8.}$$

The first update reverses the original reward gradient; the second aligns with it. Their count-only component is the same, $c_Qg=(3/2)(-1/2)=-3/4$.

This is also a natural-gradient reversal in the one-parameter policy because its Fisher information is a positive scalar. It is not caused by a coordinate choice or Adam. Mixing each pair law with 10% IID gives full pair support and preserves the reversal for $Q_A$.

**The conclusion is limited but exact:** success-count information cannot identify the expected learning direction. The counterexample does not establish prevalence in LLMs, and it should be an illustrative theorem consequence, not the experimental centerpiece.

### 7.5 An additional exact warning about variance reduction

For four equally likely responses, divide the latent space into strata $\{1,2\}$ and $\{3,4\}$. Use one independent response from each stratum, then randomize their order. Let

$$r=(1,0,0,0),\qquad s=(1,-4,3,0).$$

Then

$$g=1/4,\qquad g_{\rm strat}=-1/8.$$

Pass@2 rises from $7/16$ to $1/2$. Moreover, ordinary stratification reduces the variance of the sample mean for every scalar integrand. Nonetheless, its group-baseline update reverses the reward direction in this restricted policy.

This does not say the **gradient estimator's** variance is reduced. That is a different random object. The example prevents the paper from conflating general Monte Carlo efficiency with safe relative learning.

---

## 8. The key causal intervention: count-preserving rewiring

### 8.1 Construction

For one prompt and one frozen checkpoint, store $B$ independently generated coupled groups of size $K$.

Keep all group success/failure slots exactly as observed. Randomly permute the successful trajectory identities over the successful slots, and independently permute failed identities over failed slots.

This preserves:

- the exact multiset of stored trajectories and their rewards;
- every group's success count, hence its coverage indicator, zero-variance status, mean reward, and binary reward standard deviation;
- the total number of tokens and all pooled trajectory statistics.

It changes the association between a trajectory's identity and the count-dependent advantage it receives. It does not preserve within-group lexical similarity, prefix structure, or gradient covariance. Those are deliberate or measured consequences, not hidden invariants.

### 8.2 Integrate out the randomization analytically

Let

$$\bar a=\frac1B\sum_{b=1}^B a_K(N_b),\qquad
\widehat p_+=\frac1{BK}\sum_i r_i.$$

The exact conditional expectation over rewiring is

$$
\boxed{
\mathbb E_{\rm rewire}[\widehat g\mid\text{stored pool}]
=\bar a\left(\bar s_{+,\rm pool}-\bar s_{-,\rm pool}\right).
}
\tag{21}
$$

It can be computed by one weighted backward pass, using advantages

$$
A_i^{\rm count}=
\begin{cases}
\bar a/\widehat p_+,&r_i=1,\\
-\bar a/(1-\widehat p_+),&r_i=0.
\end{cases}
\tag{22}
$$

Use the standardized $a_K^{\rm std}$ for the corresponding normalized estimator. If the whole pool contains only one reward class, both the within-pool and rewired contrast are zero; do not divide by zero.

Equation (21) is a finite-pool conditional statement. Its interpretation as $c_Qg$ is a population limit or requires independently estimated population quantities. Do not label a finite ratio of dependent sample means exactly unbiased.

**Why this is useful:** a deterministic analytic average eliminates extra permutation noise. The scientific comparison is between two fixed gradients computed from exactly the same responses and reward-count profile.

### 8.3 The complementary independent-group baseline

For at least two independent groups at the same prompt, use the rewards of other groups as a baseline:

$$b_i^{\rm cross}=\text{mean reward in groups independent of the group containing }i.$$

Then, without a response-dependent normalizer,

$$\mathbb E[(r_i-b_i^{\rm cross})s_i]=g.$$

With two groups, cross the baselines symmetrically. The within-group arm also consumes both groups, so this intervention does not require extra responses relative to its matched control.

**Important restriction:** dividing this estimator by the current group's realized reward standard deviation reintroduces action dependence. It is not an unbiased correction for normalized GRPO. Use the binary count decomposition for that case, or explicitly define a different fixed-scale target.

### 8.4 What the three-way comparison identifies

Let $W$ be the original within-group estimator, $C$ the analytic count-preserving estimator, and $X$ the independent-group estimator.

| Contrast | Primary interpretation | Remaining qualification |
|---|---|---|
| $W-C$ | Selection of success/failure identities beyond the fixed count profile | Finite-pool target; variance and joint text structure may change |
| $C-X$ | Count-associated weighting, including prompt weighting | Finite-pool ratios need uncertainty analysis |
| $W-X$ | Total baseline-associated mean-update change | Does not alone isolate coverage, rescaling, or variance |
| IID $W-C$ | Negative control for the entire rewiring pipeline | Clustered uncertainty is still required |

The IID control should have zero rewiring effect in expectation: conditional on binary labels, IID trajectory identities are exchangeable within each reward class.

Do not use naive permutation-test $p$-values under a weaker “zero mean tilt” null that does not imply exchangeability. Bootstrap independent original groups, and use independent trajectory banks for replication.

### 8.5 Dose interventions

On fixed stored trajectories, define

$$A_i^{(\lambda)}=(1-\lambda)A_i^C+\lambda A_i^W.$$

Use a small preregistered set such as $\lambda\in\{0,1/2,1\}$. This provides a controllable dose of identity-dependent contrast assignment.

Linearity of the stored-data gradient in $\lambda$ is an algebraic identity, **not** an experimental discovery. The test is whether a forecast made on discovery data predicts held-out functional changes or reward slopes under these interventions.

### 8.6 A stronger variance control: hold the update noise fixed

Count-preserving rewiring changes both the mean and the covariance of the estimator. Equal sampled responses do not, by themselves, eliminate this variance explanation.

Use independently estimated mean directions $\widetilde\mu_W,\widetilde\mu_C$ and a common reference noise draw $\epsilon_b$, generated from a fixed, separately centered bank of reference batch gradients. Compare the artificial diagnostic updates

$$
\widetilde g_b^{W}=\widetilde\mu_W+\epsilon_b,\qquad
\widetilde g_b^{C}=\widetilde\mu_C+\epsilon_b.
$$

Conditional on the fixed means and noise bank, the two update distributions have **exactly the same noise covariance**, and each paired update differs only by the fixed mean shift. A deterministic zero-noise version is also useful. This is a diagnostic intervention, not a proposed training algorithm and not a claim that estimated means equal population means.

Use the same fixed preconditioner and scalar step for this control; separately rescaling each noisy draw to an equal KL would destroy the common-noise property. Choose a conservative common step on an independent calibration bank, report the resulting KL values, and retain equal-KL deterministic-mean comparisons as a complementary test. Reuse weighted backward passes on the stored trajectories; new rollout generation is unnecessary.

The strongest causal evidence is agreement between the same-noise mean-shift test, the equal-KL mean-direction test, and the independently predicted reward slope. Run a small paired version in the expanded frozen-policy audit rather than adding another full training matrix.

---

## 9. Prediction before intervention: use IID data to forecast stratified learning

### 9.1 RLOO forecast

Collect IID trajectories with the latent bin label induced by the declared decoder. Use independent sub-banks to estimate $\mu_{r,j}$ and score averages, or compute cross-fitted weighted gradients directly.

The predicted update is

$$\widetilde g_{\rm strat}=\widehat g_{\rm IID}
+\frac1{K-1}\widehat{\operatorname{Cov}}_B(\mu_{s,B},\mu_{r,B}).$$

Freeze this forecast before evaluating independently generated stratified groups. Assess functional direction, magnitude, and task-conditioned effects. Compare against a positive-scalar-only predictor and a reward-count-only predictor.

An equivalent finite-bank implementation constructs virtual groups by selecting one stored IID trajectory from each latent bin. Integrating over the combinations is a Rao–Blackwellized counterfactual calculation. Reusing trajectories does not create additional independent observations; uncertainty is based on the original IID bank.

### 9.2 Exact binary normalized forecast

For stratum $j$, let

$$p_j=\Pr(r=1\mid B=j),\qquad m_{j,+}=\mathbb E[s\mid B=j,r=1],\qquad
m_{j,-}=\mathbb E[s\mid B=j,r=0].$$

Under stratified sampling, the rewards across strata are independent Bernoulli variables. Let

$$M_{-j}=\sum_{\ell\ne j}\operatorname{Bernoulli}(p_\ell).$$

Define

$$u_j^+=\mathbb E\left[\frac{K-1-M_{-j}}{K d_{1+M_{-j}}}\right],\qquad
u_j^-=\mathbb E\left[\frac{M_{-j}}{K d_{M_{-j}}}\right].$$

Then

$$
\boxed{
g_{\rm strat,std}
=\frac1K\sum_{j=1}^K
\left[p_ju_j^+m_{j,+}-(1-p_j)u_j^-m_{j,-}\right].
}
\tag{23}
$$

The Poisson-binomial probabilities can be calculated by polynomial convolution. At the proposed small $K$, this is a negligible CPU calculation. Missing reward classes in a stratum are handled through their zero mass, not an undefined empirical score mean.

Equation (23) makes practical normalization a testable part of the theory. Its population statement is exact; plug-in estimates have finite-sample uncertainty and can be poor in sparsely populated bins. Use cross-fitting, bank-size sensitivity, or finite-pool integrated predictions. Do not claim an estimated formula is automatically unbiased.

### 9.3 Forecasting what will be learned

Before the confirmation experiment, register a small set of behavioral quantities, such as held-out verifier reward, probability of a valid final-answer format, and a prespecified partition of arithmetic solution/failure types.

For an update direction $d$, predict

$$\Delta \mathbb E[f(Y)]\approx\eta\,\mathbb E[f(Y)s(Y)]^Td$$

using independent IID evaluation trajectories. Test it with small interventions and independently evaluated responses or stable likelihood-ratio estimates.

Calculus alone is not the contribution. The meaningful result is that the **sampler-specific forecast from different data** predicts the direction and magnitude of changes that count-only models miss.

---

## 10. Novelty audit and reviewer-facing positioning

### 10.1 Closest prior literature

| Literature | Already established | Remaining claim worth testing |
|---|---|---|
| **QuasiMoTTo** [R1] | Marginal-preserving QMC rollouts, baseline-induced bias, pair correction, and sampler-dependent training behavior | Exact contrast geometry; quantitative sampler safety/filtering boundaries; independent-data prediction and count-preserving causal attribution |
| **CARMS / ARMS** [R2, R4] | Antithetic estimators, pairwise products, unbiased corrections; scalar correction in binary-variable settings | Shared-parameter sequence policies, score-subspace interference, and response-identity selection within binary reward classes |
| **PAIR** [R3a] | Pairwise inclusion weighting for adaptively observed rollout pairs | Dependence in complete marginal-preserving groups; no outcome-selection or inclusion-weighting method as the central contribution |
| **GRPO U-statistic theory** [R3] | Pairwise structure, variance and asymptotic analysis, practical-estimator distinctions | Structured dependent group laws and their interaction with learnable reward directions |
| **RL2ML and finite-group objective theory** [R5] | Success-count-dependent update scales and IID success-conditioned score calculations | Failure of count sufficiency when score identity depends on the count profile |
| **SoftmaxGRPO / objective-bias analyses** [R6, R14] | Prompt weighting and limits of scalar-objective interpretations | Marginal-preserving sampling dependence, rather than another advantage transform |
| **Control variates and baseline theory** [R10, R11] | Orthogonality conditions, variance reduction, and nontrivial optimization effects | A useful predictive characterization of the coupled-rollout setting, not “baselines matter” |
| **Reversible Markov operators / Dirichlet forms** [R7] | Self-adjointness, positive quadratic forms, spectral bounds, conditional projections | The specific score-subspace compatibility application and experimentally validated consequences |
| **Compatible approximation / natural gradients** [R8, R9] | Fisher geometry and score-space projections | What sampling operators do to the projection and its unexplained residual |
| **Shared-prefix / group-objective work** [R12, R13] | Cancellation and credit-assignment effects | Use as a control; do not claim shared-prefix cancellation as new |

The distinction between ARMS binary **random variables** and binary **rewards** is particularly important. An LLM with a binary verifier still has a huge trajectory space and many different scores within each reward class. A binary reward does not reduce the policy to a single Bernoulli parameter.

### 10.2 What can responsibly be called novel now?

The primary-source audit did not identify the following exact combined program: the universal score-subspace compatibility characterization; the stratified-versus-lattice compressed-operator analysis with the bound in (15); and the count-preserving identity-selection intervention validated through IID-only update forecasts.

That is a **candidate novelty claim**, not a statement that no equivalent result exists. Several components are short deductions from classical mathematics. Before submission, conduct a final targeted audit of operator invariance/Galerkin approximation, conditional covariance identities, survey-sampling stratification, and coupled score-function estimators.

A paper is not strong merely because its exact terminology is absent from search results. The decisive question is whether the framework predicts a practically consequential effect that the nearest work leaves unresolved.

### 10.3 A defensible positioning paragraph

> Marginal-preserving dependence changes group-relative learning through the distribution of contrasts, not only through coverage. We characterize when the resulting update is a reward-independent positive preconditioner and when it depends on reward components invisible to the original policy gradient. For important sampling laws, we derive explicit operators and independently testable update predictions. Count-preserving interventions on identical trajectories then isolate response-identity selection from reward-count effects and establish when it changes downstream learning.

Do not add “for the first time” without a much broader bibliographic search. Do not call a locally transformed reward a globally optimized objective unless integrability is separately proved.

### 10.4 The strongest reviewer objections

**“This is CARMS without importance weights.”** The answer must be the compatibility characterization and causal predictive result, not a different notation for bias. If those empirical predictions fail, the objection is substantially correct.

**“You rediscovered a Markov-chain Dirichlet form.”** Agree about the mathematics. Explain which sampler-specific prediction and learning phenomenon follows from its interaction with the score subspace.

**“This is just difficult prompts being reweighted.”** Report $W-C$ separately from $C-X$, perform within-prompt analyses, and keep prompt weights fixed in the relevant contrast.

**“A gradient angle is an optimizer artifact.”** Show raw mean updates, functional KL geometry, a fixed-preconditioner intervention, and independently measured objective slopes. Reserve natural-gradient guarantees for their actual assumptions.

**“Your toy policy created the problem.”** Use full-parameter unrestricted-response audits on pretrained models. Make constrained subspaces and exact finite policies controls, not the main empirical evidence.

**“You only explain one sampler.”** Establish a general characterization, test stratified and lattice laws, and use a known conditional-prefix kernel as a structural control. Do not add unrelated samplers merely to enlarge a table.

**“A one-step identity is not a prediction.”** Make sampler forecasts from IID discovery data, lock them, and evaluate independently generated groups and held-out behavior. Separate algebra checks from predictive evidence in every figure caption.

---

## 11. Experimental program: evidence in the order it should be earned

### 11.1 Models and tasks

Use **Qwen2.5-0.5B-Instruct** for the primary pilot and training. Replicate important effects with **Qwen2.5-1.5B-Instruct** and use **SmolLM2-1.7B-Instruct** for a separate-family frozen-policy audit. Pin exact model revisions and tokenizer/chat-template versions. These public checkpoints provide the proposed sizes without requiring a large-model centerpiece. [R18–R20]

The task hierarchy is deliberate:

| Task | Role | Restriction |
|---|---|---|
| Exact categorical policies and small autoregressive trees | Proof checks, counterexamples, saturation and metric controls | Never the only positive evidence |
| Generated three-/four-number arithmetic expressions | Verifiable outcomes, known solution structure, fresh held-out instances | Use an AST-based checker and rational arithmetic; no unrestricted `eval` |
| SVAMP | Short natural-language math, primary natural task | Keep data partitions and prompt format fixed [R21] |
| GSM8K | Limited external validation, especially on 1.5B | No benchmark tuning; no claim of newly acquired frontier reasoning [R22] |

Use distinct calibration, discovery, confirmation, training, and evaluation partitions. For generated tasks, split both random seeds and structural templates. Check textual duplicates in fixed datasets.

Do not select prompts because the two samplers disagree. A prespecified intermediate-success subset may improve measurement power, but report the unfiltered population and use only IID calibration outcomes to define that subset.

### 11.2 Stage A — CPU exact science before GPU rental

Verify (3), (9), (18), (21), the two counterexamples, and the spectral bound. Include a saturated softmax policy where natural reversal cannot occur and a restricted exponential-family policy where it can.

For the normalized forecast in (23), enumerate all $2^K$ reward configurations at small $K$ and compare against the Poisson-binomial calculation. Verify finite-pool rewiring by exhaustive permutations on a tiny pool.

Test the sampler's pair law and marginal probabilities on small autoregressive trees. Randomly permute group members. Confirm that a $K=8$ group arbitrarily subsampled to four elements is **not** silently treated as the native $K=4$ law. Any valid coarsening must be derived from the specific sampler construction.

This dossier's CPU checks obtained exact counterexample gradients $(-1/2,1/8,-11/8)$ and $(1/4,-1/8)$. Numerical projection and spectral-bound checks agreed within approximately $2.5\times10^{-15}$ in the tested finite cases. The normalized stratified forecast also matched exhaustive reward-configuration enumeration through $K=8$, with maximum observed error about $4.5\times10^{-16}$. These are implementation checks, not substitutes for the proofs.

### 11.3 Stage B — sampler and loss implementation validation

Fix temperature at 1, use full sampling support, disable dropout, and use the same response cap and EOS rule. Compare IID arithmetic decoding against the intended marginal policy before testing dependence.

A float-valued arithmetic decoder is not proven exact for arbitrarily long strings merely because its relative-coordinate recurrence avoids underflow. Audit finite-precision behavior on small trees and long synthetic prefixes, and document any correction or approximation. Do not silently switch to independent token-level uniforms midway and retain the original sequence-level theorem.

Log latent stratum/orbit metadata, group IDs, independent randomization-block IDs, token IDs, response lengths, rewards, log-probabilities, model revision, and the complete loss normalization. The marginal-invariance claim needs both a mathematical implementation argument and empirical checks; histogram agreement alone is not proof.

If implementing a valid sampler becomes the principal project, stop. This proposal is not a mandate to spend the budget building a custom inference framework.

### 11.4 Stage C — frozen-policy mean-update audit

For each prompt and sampler, generate independent groups and compute $W$, $C$, and $X$ on the same stored pool.

Use weighted sequence-log-probability losses to compute aggregate gradients. Do not store a full parameter gradient for every trajectory. Stream a small number of aggregate gradients to CPU or disk, or compute directional projections as needed.

The primary comparison uses the full trainable parameter set on 0.5B. A fixed low-dimensional probe subspace is allowed for score-space visualization, but every such figure must identify the restricted subspace explicitly.

### 11.5 Functional geometry without inverting a billion-dimensional Fisher matrix

For an actual fixed update map $M_0$, define $d_W=M_0g_W$ and $d_C=M_0g_C$. On independent evaluation trajectories, estimate

$$\|d\|_{F_{\rm eval}}^2=\mathbb E[(s(Y)^Td)^2].$$

The non-scalar distortion statistic is

$$
\delta_{\rm func}
=\min_{a\ge0}\frac{\|d_W-a d_C\|_{F_{\rm eval}}}
{\|d_X\|_{F_{\rm eval}}}.
\tag{24}
$$

Use SGD ($M_0=I$) as the first mechanism setting. A second setting can use a preconditioner frozen independently from the treatment gradients. Match intervention size by held-out sequence-level KL, not merely parameter norm.

Directional scores can be obtained by validated automatic differentiation or central finite differences of log probabilities. Check step-halving agreement. This measures the geometry of full parameter directions without claiming to estimate the full natural gradient.

Avoid upward-biased “norm of a noisy difference” conclusions. Use independent group-bank estimates, cross-bank inner products where appropriate, and cluster bootstrap uncertainty. If the denominator is unresolved or close to zero, report the measurement as unidentifiable rather than a large percentage.

### 11.6 Stage D — predictive test

From IID discovery trajectories, predict the stratified RLOO update using (12), and the standardized binary update using (23). Freeze the positive-scalar and count-only comparator models at the same time.

Test on independent rollout randomization, held-out prompts, and subsequently a second checkpoint. Compare predictive error in functional coordinates, not just agreement on the total loss gradient's norm.

The operator identity evaluated on its own estimation bank is a unit test. A result only becomes predictive when the estimation and measurement banks are separated.

**Specify the split correctly.** Discovery prompts choose and freeze the procedure, thresholds, and comparator fits. For each confirmation prompt, obtain an IID-only calibration bank at the checkpoint being tested, form the prompt-specific forecast, and lock it before inspecting that prompt's coupled-gradient bank. A new checkpoint likewise needs new IID calibration unless a separate transfer model has been specified. This is an out-of-treatment prediction on held-out prompts, not an unsupported zero-shot claim that the conditional bin means learned from one prompt or checkpoint apply to another. The pilot already allocates IID and coupled banks at each prompt; blind the coupled results until forecasts are committed.

### 11.7 Stage E — small-step causal consequences

Apply $W$, $C$, and the preregistered mixture doses from the same starting checkpoint. Match their KL size using a separate calibration bank and keep the update map fixed. Include a half-size step to check that the first-order regime is informative.

Evaluate held-out reward slopes and actual response changes. Small-step correctness differences may be too small for a cheap fresh-generation estimate. In that case use an independent IID bank with sequence likelihood ratios,

$$\widehat J(\theta')=\frac1n\sum_i
r(Y_i)\frac{p_{\theta'}(Y_i)}{p_\theta(Y_i)},$$

and report effective sample size, weight tails, and uncertainty. Do not clip weights and call the estimate unbiased. If overlap is poor, the estimate is not decisive and fresh generation is required.

Reference-answer likelihood or formatting changes alone do not establish improved reasoning. At least one prespecified verifier-reward consequence must be measured or reliably estimated.

### 11.8 Stage F — downstream training, only after the pilot passes

The primary factorial design is

$$\{\text{IID},\text{stratified},\text{lattice}\}
\times
\{\text{within-group RLOO},\text{cross-group RLOO}\}.$$

Each prompt consumes two independent groups of size four in **every** arm. Keep the total responses, prompt distribution, token cap, optimizer, and update schedule matched. This is not a comparison between one group of four in one arm and eight responses in another.

Run three seeds on the primary natural task. Replicate a prespecified focused four-arm contrast on the generated task. In the second-size study, keep the parameterization comparable; use higher-memory hardware for full tuning when necessary rather than silently changing only one arm to LoRA.

Online policies diverge. Their subsequent rollouts cannot remain identical, and matched seeds do not restore that property. The strict causal same-trajectory result belongs to the frozen and branched experiments. Training is evidence that the diagnosed effect matters downstream.

Evaluate all arms with a common IID sampler and report pass@1, fixed-budget learning curves, cumulative generated tokens, GPU time, and update KL. Estimate coupled-group coverage using actual independent complete groups rather than an IID pass@k formula.

---

## 12. Cheapest decisive pilot: at most 16 GPU-hours, about ¥44.8

### 12.1 Fixed pilot design

Start with Qwen2.5-0.5B-Instruct, $K=4$, and IID/stratified/lattice sampling.

Use 48 prompts: 16 discovery and 32 confirmation, divided between the generated arithmetic task and SVAMP. Generate 16 independent groups per prompt per sampler. With four responses per group, this is 9,216 responses before calibration/evaluation additions.

Cap responses at 160 tokens, so the main bank contains at most about 1.47 million generated tokens. This is a workload proposal, not a throughput assertion. Predeclare the interpretation of truncation. Keep additional calibration and evaluation banks inside the time envelope.

Compute the full-parameter $W$, $C$, and $X$ contrasts. Use the IID bank to form the independent stratified-update forecast. Perform a small set of equal-KL interventions on a discovery-fixed combination of task cells rather than searching over prompts for a reversal.

### 12.2 Time allocation

| Pilot component | Maximum 5090 hours |
|---|---:|
| Memory/throughput check and sampler/loss validation | 2 |
| Calibration plus rollout collection | 6 |
| Stored-data gradients, rewiring, IID-only forecast | 4 |
| Held-out directional tests and uncertainty analysis | 4 |
| **Total** | **16** |

At the supplied rate, $16\times¥2.8=¥44.8$. An earlier decisive failure should terminate the pilot. The time cap is not permission to interpret an underpowered estimate as absence of an effect.

### 12.3 A preregistered practical threshold

Use **10% functional non-scalar distortion** as a practical effect-size threshold for authorizing the full study. It is a research-allocation threshold, not a universal scientific constant.

The pilot passes only when all of the following hold:

**Implementation gate.** Marginal and score-point checks pass, the IID rewiring negative control is compatible with zero, and the reference update is measurably nonzero.

**Mechanism gate.** On confirmation data, at least one prespecified unrestricted-response coupled-sampler condition has a lower 95% confidence bound for $\delta_{\rm func}$ above 0.10. The effect survives the count-preserving control and positive-scalar matching. Use simultaneous or multiplicity-adjusted inference for the prespecified sampler comparisons.

**Prediction gate.** The discovery-frozen sampler forecast improves held-out functional prediction error by at least 25% relative to the preregistered scalar/count-only predictor, with uncertainty supporting a real improvement. The stratified IID-only forecast is the preferred test. Do not count same-bank operator reconstruction as passing this gate.

**Consequence gate.** A direction of held-out verifier-reward change predicted before the intervention is confirmed, and its magnitude is at least 10% of the resolved cross-baseline reference slope at matched KL. Improvement or harm can both pass; raw reversal is not required. The effect must not disappear entirely in the half-step or fixed-preconditioner check.

The four gates are intentionally stricter than detecting a nonzero bias. A project meeting only the familiar bias claim should not consume the remaining budget.

### 12.4 Pilot failure interpretations

If the mechanism estimate's upper confidence limit is below 0.10, **kill the strong-paper program under the tested scope**. This is evidence of a practically small effect, not universal safety.

If intervals remain too wide at 16 hours, **do not authorize the full project**. Record an underpowered or measurement-limited pilot. There should be no indefinite sequence of revised thresholds and new tasks.

If only the restricted finite policy or a specially chosen adapter subspace reverses, **kill the LLM-mechanism claim**.

If a material update difference is measured but the independent prediction or reward-consequence gate fails, retain the diagnostic as a technical result, but **do not yet treat it as the second ICLR paper**.

If stratification is empirically safe and predictable while lattice exhibits material count-independent selection, the full paper can remain viable. However, the positive lattice finding must still pass a genuinely held-out prediction test; a safety inequality that never constrains the observed setting is not a substitute.

---

## 13. Full project budget: a hard 80-hour envelope

The complete plan includes the pilot rather than adding to it.

| Stage | GPU-hours | Scientific output |
|---|---:|---|
| Implementation checks and cheapest pilot | 16 | A pass/fail decision under the four gates |
| Expanded frozen-policy and normalization audits | 8 | More independent groups; $K=4,8$; sampler forecasts and finite-bank sensitivity |
| Primary task: six arms, three seeds | 27 | 18 matched training runs, with checkpoints reserved for prediction |
| Second task: focused four arms, two seeds | 8 | Transfer of the key factorial contrast |
| 1.5B replication: focused four arms, two seeds | 8 | Larger-small-model validation; full tuning on higher-memory GPU if needed |
| Separate-family 1.7B frozen-policy audit | 5 | Architectural external validity without a training expansion |
| Final held-out evaluation and rerun reserve | 8 | Reproducibility, uncertainty, failed-run allowance |
| **Total** | **80** | |

At 80 hours entirely on the quoted 5090, the rental cost is **¥224**. With 8 hours moved to the quoted PRO 6000-class hardware, the estimate becomes

$$72\times¥2.8+8\times¥6=\boxed{¥249.6}.$$

Even 80 hours at ¥6/hour is ¥480. Storage, network, and other charges are not specified and must be tracked separately. The ¥1000 ceiling is therefore less restrictive than the 80-hour scientific-work envelope; unused money is not a reason to expand the study.

### 13.1 Training workload must be benchmarked, not assumed

A starting primary-run workload is 100 updates, four prompts per update, two groups of four responses, and a 160-token cap. Across 18 runs this is at most about 9.22 million generated tokens. Finishing it in the allocated 27 hours requires about 95 generated tokens/second averaged over **generation, backward passes, synchronization, and checkpoint overhead**.

This rate has not been measured. Before launching the matrix, measure end-to-end throughput for all sampler arms and set one common token/update budget that fits the slowest relevant pipeline. Do not let faster arms receive more learning data unless that is a separately labeled wall-clock-efficiency experiment.

A 64-update second-task schedule and a shorter 1.5B replication are starting plans, not guaranteed sufficient training. The IID control must show resolved learning in the development setup. If it does not, a flat comparison cannot validate the theory. A single prespecified correction of difficulty or format is reasonable; open-ended task tuning is not.

### 13.2 What to cut first

Protect the independent-data forecast, count-preserving intervention, primary three-seed result, and one external model audit. Cut long-run training duration, extra benchmark evaluations, and optional extensions before cutting those pieces.

Do not replace a missing causal result with more training curves. Do not spend the remaining funds searching larger models for one positive case.

---

## 14. Statistical and causal validity checklist

The observational unit for uncertainty is an independently randomized original group, nested within a prompt. Rewired groups and virtual stratified groups are not new independent units.

Use prompt-level resampling for prompt-generalization statements and group-level resampling for conditional checkpoint/prompt estimates. Keep both sources of variation visible. Three training seeds and hundreds of sampled groups answer different uncertainty questions.

Estimate mean differences on independent banks or with cross-fitting. A sample cosine of two noisy gradients can be misleading; report noise-corrected or replicated inner products and their uncertainty. Do not silently clip negative estimated squared norms caused by noise and then compute a confident ratio.

Fit scaling factors on discovery data and evaluate them on confirmation data. Distinguish global scalar matching from prompt-specific count-associated scaling. Do not optimize a flexible preconditioner against the same rewards and then call the resulting fit an explanation.

For functional comparisons, use identical evaluation prefixes, the declared policy law, and the same response masking. For actual reward consequences, use fresh generation or a valid independently sampled likelihood-ratio estimate with adequate overlap.

A correct finite-pool gradient identity is not proof of a population effect. Report bank-size sensitivity, the exact finite-pool estimand, and the comparison to the population formula separately.

The code should assert that rewiring preserves the reward vector in every group slot, the complete count histogram, trajectory IDs with multiplicities, and the total response-token count. These are machine-checkable invariants, not prose claims.

---

## 15. Failure modes and what would remain publishable

| Failure mode | Interpretation | Decision |
|---|---|---|
| The nearest literature already contains the score-space and sampler-specific results | Novelty collapses | **KILL** unless a genuinely independent causal discovery survives |
| Only positive scalar changes are observed | The main phenomenon is step size or prompt weighting | Usually **KILL** this framing; do not relabel scale as rotation |
| Count-preserving rewiring removes no meaningful effect | Coverage/count explanations may be adequate in the tested settings | **KILL** the stronger empirical claim |
| IID-only forecasts fail despite a correct implementation | Estimation may be too weak, or the practical sampler differs from the assumed law | Fix one identified issue inside the pilot cap; otherwise stop |
| Effects disappear outside a hand-picked score subspace | A mathematical example, not an LLM mechanism | **KILL** the intended paper |
| Mean effects are real but vanish at matched KL or fixed preconditioning | Optimizer/step-size explanation dominates | Reframe honestly; weak basis for the current main claim |
| Count-independent mean effects predict reward changes but do not improve benchmark accuracy | Mechanism can still be substantive | **GO** if external replication and relevance are strong |
| Stratified natural updates satisfy a meaningful measured safety boundary | Positive theoretical result, not a failed negative-bias story | Valuable when the bound is genuinely constraining and predictive |
| All experiments are noisy under the budget | Measurement-limited project | Stop rather than claim a negative result |

A strong negative paper would need an explanatory boundary—such as a quantitatively tight safety result that predicts negligible practical effects—not merely “we failed to find harm.” A toy counterexample plus a small null experiment is unlikely to support the intended venue.

Theoretical identities surviving an empirical kill remain useful. They should be written up accurately, but mathematical correctness is not the same as sufficient research significance.

---

## 16. Extensions that could materially strengthen the paper

### 16.1 A natural method follows, but it is not the headline

Independent-group baselines recover the original RLOO mean without requiring joint sequence probabilities. This follows directly from conditional independence. It is operationally useful as an audit or fallback when the measured selection term is undesirable, but the underlying control-variate principle is not new. [R2, R10, R11]

An oracle minimal-change safe baseline also follows from the geometry. If the original response-dependent baseline is $b=Tr$, then its closest $L_2(p)$ replacement satisfying $S^*b_{\rm safe}=0$ is

$$b_{\rm safe}=(I-\Pi)b.$$

This is an orthogonal-projection solution, not a ready-to-use LLM method. Estimating $\Pi$ well may be more expensive or less stable than crossing baselines. Do not build a learned projection module merely to claim an algorithm.

### 16.2 Counterfactual reward transfer

The general characterization predicts that two rewards with the same learnable projection may have different coupled updates. Test this exactly in finite policies and, only as a labeled subspace experiment, with prespecified score-orthogonal residuals in a pretrained model.

For natural tasks, reuse stored responses with additional preregistered verifier-derived components—correctness, syntactic validity, and a mechanically defined constraint check. This provides several reward directions without new rollout collection. Do not tune reward mixtures to manufacture a reversal and present them as naturally occurring failures.

### 16.3 Multiple discrete reward levels

Rewiring can preserve slots at each exact reward level. The integrated update is then a combination of reward-level-conditioned score means rather than necessarily a scalar multiple of $g$. This separates changes caused by reward-level weighting from identity selection within each level.

This extension is mathematically natural, but it should not displace the binary-reward core. Existing objective-transform work already shows that richer reward alphabets have additional complications. [R6, R14]

### 16.4 Policy evolution and decoder order

The operator changes with the checkpoint because $G_\theta$, the score subspace, and reward explainability change. A sampler can remain marginally valid while its learning effect evolves during training.

A high-value extension is to freeze forecasts at an early checkpoint and test their failure or update at a later checkpoint. Do not expect one global calibration to remain valid indefinitely.

Changing the deterministic token ordering inside arithmetic decoding preserves marginals but changes its coupling to sequence semantics. A tightly controlled ordering intervention could test the operator mechanism. It should be an external validity test with one prespecified alternative, not an ordering sweep or a new optimized sampler.

### 16.5 No automatic monotonic benefit from model scale or adapter rank

Theorem 1 does not imply that larger models always suffer less interference. Increasing the score subspace reduces the unexplained reward residual but also changes the operator coupling and optimization geometry. Even nested subspaces do not make every practical update metric monotone.

Therefore, model scale and adapter rank are controls, not an optimization axis for the project. A clean phenomenon on 0.5B and 1.5B is sufficient for the proposed scientific claim if the theorem and causal tests are strong.

### 16.6 Optional non-conservativity analysis

One may test whether $\theta\mapsto g_Q(\theta)$ integrates to a scalar objective on a two-parameter finite policy. This can clarify the phrase “what is being optimized.” It is not a priority experiment: non-gradient policy-update fields and finite-group non-conservative transforms already have direct precedents. [R14, R15]

Do not claim “the sampler optimizes a different objective” when the defensible statement is only “the sampler changes the expected update.”

---

## 17. What the final paper could claim if the program succeeds

### Contribution 1 — a characterization with concrete sampler consequences

A reward-independent characterization of when marginal-preserving dependence is compatible with the policy score subspace, separating unbiasedness, scalar rescaling, positive natural preconditioning, and residual interference. Explicit stratified/lattice operators supply a quantitative safety boundary and explain why different dependence structures need not have the same learning effect.

The paper must acknowledge that Dirichlet forms, Fisher projections, and the block-operator proof are classical ingredients.

### Contribution 2 — a predictive decomposition beyond reward counts

An exact binary-reward decomposition, including score-point standardized GRPO, showing how response identities can receive different learning weights despite identical success-count statistics. Independent IID trajectories predict stratified updates, and count-preserving rewiring identifies the otherwise hidden selection component on identical stored rollouts.

This contribution is justified only if the forecasts and interventions work on unrestricted pretrained-model responses, not just in finite examples.

### Contribution 3 — causal evidence of learning consequences

Compute-matched, small-model evidence that the identified mean-update component predicts held-out functional and verifier-reward changes, with fixed-preconditioner controls and downstream replication. This can support either a harmful-selection result or a beneficial-selection result; a new method is unnecessary.

Do not claim universal sampler superiority, globally altered objectives, or a general convergence theorem unless additional results actually establish them.

### Suggested paper structure

The introduction should lead with a short equal-count example, not a long survey of bias. The main theory should introduce the contrast operator, give the compatibility characterization, and immediately specialize it to stratified/lattice predictions. The experimental centerpiece should be an IID-only prediction plot and a count-preserving causal intervention. Training curves come afterward.

A strong result is: **“The same reward-count statistics support different learning directions, and a model of the contrast assignment predicts which difference matters.”** A weak result is: **“Our alternative baseline trains a little better.”**

---

## 18. Reproducibility contract and execution handoff

Before using the GPU, freeze a manifest containing model revisions, tokenizer/chat templates, dataset splits, prompt format, verifier implementation, token cap, sampler specification and ordering, independent-group definition, loss equation, optimizer/update map, random seeds, and the four pilot gates.

A minimal execution outline is:

```text
CPU:
    verify exact pair-gradient, count, projection, and sampler identities
    enumerate the counterexamples and normalized binary forecast
    validate count-rewiring invariants

PILOT:
    validate model marginal and score-point implementation
    collect IID / stratified / lattice banks with independent group IDs
    split discovery and confirmation before computing mechanism results

    on discovery IID data:
        estimate stratum statistics with cross-fitting
        freeze sampler-update forecasts and scalar/count-only comparators
        choose update directions and KL step sizes without confirmation rewards

    on each stored bank:
        compute W: original within-group weighted gradient
        compute C: analytic count-preserving rewiring gradient
        compute X: independent-group baseline gradient
        retain original groups as statistical sampling units

    on confirmation data:
        measure full-parameter functional distortion with uncertainty
        evaluate independent sampler forecasts
        test preregistered small-step verifier-reward consequences
        check IID sham and finite-bank sensitivity

    if any pilot gate fails or remains unresolved at the cap:
        stop full-project authorization; save the exact failure interpretation
    otherwise:
        execute the fixed-budget factorial training and external audits
```

Artifacts needed for an auditable paper are the frozen manifest; immutable rollout/reward metadata; per-bank aggregate gradients or reproducible weighted-loss specifications; prediction files timestamped before confirmation; finite-policy checks; group-aware confidence intervals; and a claims table distinguishing proven statements, measured findings, and untested extensions.

Do not write the final claims around whichever training curve wins. Write them around the mechanism that the preregistered intervention actually identifies.

---

## 19. Primary-source references

These references support the literature audit and implementation choices. Equations labeled as derivations in this dossier are supplied with proofs; citing a related source does not claim that source established the proposed application.

**[R1]** *QuasiMoTTo: Quasi-Monte Carlo Test-Time Scaling.* arXiv:2607.01179, July 2026. [Primary text](https://arxiv.org/html/2607.01179v1). Particularly relevant: the RLOO discussion and sampler/training interpretation. 

**[R2]** Dimitriev and Zhou. *CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator.* NeurIPS 2021; arXiv:2110.14002. [Primary text](https://arxiv.org/html/2110.14002). Relevant for pairwise corrections and categorical antithetic estimators.

**[R3]** *Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic.* arXiv:2603.01162, March 2026, inspected version 3. [Primary text](https://arxiv.org/html/2603.01162v3). Distinguishes the analyzed estimator from practical normalization and optimizer choices.

**[R3a]** *PAIR: Pairwise-Aware Inclusion Reweighting for Adaptive Rollout Allocation in RLVR.* arXiv:2608.11368, August 2026. [Primary text](https://arxiv.org/html/2608.11368v1). Its inclusion-design target differs from complete marginal-preserving coupled groups.

**[R4]** Dimitriev and Zhou. *ARMS: Antithetic-REINFORCE-Multi-Sample Gradient for Binary Variables.* ICML 2021; arXiv:2105.14141. [Primary text](https://arxiv.org/html/2105.14141).

**[R5]** *RL2ML: Finite-Rollout Surrogate Objectives from Reinforcement Learning to Maximum Likelihood.* arXiv:2605.30154, May 2026. [Primary text](https://arxiv.org/html/2605.30154v1). Relevant for count-dependent update scales and IID success-conditioned score reasoning.

**[R6]** *SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation.* arXiv:2608.09271, August 2026. [Primary text](https://arxiv.org/html/2608.09271v1). Relevant for finite-group prompt weighting and scalar-objective limits.

**[R7]** Sherlock. *Reversible Markov chains: variational representations and ordering.* arXiv:1809.01903. [Author's mathematical exposition](https://arxiv.org/html/1809.01903). Used for the classical reversible-kernel and Dirichlet-form background, not a novelty claim.

**[R8]** Sutton, McAllester, Singh, and Mansour. *Policy Gradient Methods for Reinforcement Learning with Function Approximation.* NeurIPS 1999. [Proceedings](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation).

**[R9]** Kakade. *A Natural Policy Gradient.* NeurIPS 2001. [Proceedings](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html).

**[R10]** Greensmith, Bartlett, and Baxter. *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning.* JMLR 5, 2004. [Journal record](https://www.jmlr.org/papers/v5/greensmith04a.html).

**[R11]** Chung, Thomas, Machado, and Le Roux. *Beyond variance reduction: Understanding the true impact of baselines on policy optimization.* arXiv:2008.13773. [Primary record](https://arxiv.org/abs/2008.13773).

**[R12]** *On the Hidden Objective Biases of Group-based Reinforcement Learning.* arXiv:2601.05002, January 2026. [Primary text](https://arxiv.org/html/2601.05002v1).

**[R13]** *GTPO: Stabilizing Group Relative Policy Optimization via Gradient and Entropy Control.* arXiv:2508.03772, inspected version 5. [Primary text](https://arxiv.org/html/2508.03772v5). Relevant for shared-prefix gradient effects. Also see the author project [*GRPO is Secretly a Process Reward Model*](https://coli-saar.github.io/grpo_prm) for closely related cancellation/credit-assignment discussion.

**[R14]** *SoftmaxGRPO*, [R6], especially its finite-reward-alphabet non-conservativity discussion. This is intentionally a cross-reference, not an additional independent source.

**[R15]** Nota and Thomas. *Is the Policy Gradient a Gradient?* arXiv:1906.07073. [Primary record](https://arxiv.org/abs/1906.07073).

**[R16]** Dong, Mnih, and Tucker. *Coupled Gradient Estimators for Discrete Latent Variables.* arXiv:2106.08056. [Primary text](https://arxiv.org/html/2106.08056v2). Additional coupled-estimator prior art for the final novelty audit.

**[R17]** Hugging Face TRL. [RLOO trainer documentation](https://huggingface.co/docs/trl/en/rloo_trainer) and [GRPO trainer documentation](https://huggingface.co/docs/trl/en/grpo_trainer). Pin a software version rather than assuming current defaults match this dossier.

**[R18]** Qwen. [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

**[R19]** Qwen. [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

**[R20]** Hugging Face. [SmolLM2-1.7B-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

**[R21]** Patel, Bhattamishra, and Goyal. *Are NLP Models really able to Solve Simple Math Word Problems?* NAACL 2021. [ACL Anthology](https://aclanthology.org/2021.naacl-main.168/). Source for SVAMP.

**[R22]** Cobbe et al. *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168. [Primary record](https://arxiv.org/abs/2110.14168). Source for GSM8K.

**[R23]** Wu et al. *Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines.* arXiv:1803.07246 / ICLR 2018. [Primary record](https://arxiv.org/abs/1803.07246). Additional control-variate prior art; action dependence alone is not sufficient to conclude a baseline is invalid.

---

## 20. Final verdict

# CONDITIONAL GO

**Authorize the 16-hour, approximately ¥44.8 pilot. Do not yet authorize the entire paper as a high-confidence project.**

The theory is sufficiently coherent to determine the experiment. The most promising contribution is not a bias correction: it is a predictive account of which contrasts are learned, including a sampler-specific safety/filtering distinction and an intervention that holds the reward-count explanation fixed.

**The result that determines the decision is this:** on unrestricted pretrained-model responses, count-preserving rewiring must cause a resolved, greater-than-10% non-scalar change in the mean functional update; an independently frozen sampler forecast must improve held-out prediction over scalar/count-only alternatives; and the predicted change must produce a resolved, practically nontrivial verifier-reward consequence at matched KL. The IID sham, marginal-law checks, and fixed-preconditioner controls must pass.

If those conditions hold, proceed with the 80-hour program. If only known bias, toy reversal, positive rescaling, or noisy training-curve differences remain, **KILL this direction as the intended strong second ICLR paper rather than spending the remaining budget to manufacture a method.**

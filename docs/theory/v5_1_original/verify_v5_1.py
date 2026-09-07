"""CPU checks for the v4 diagnostic / v5 stratified-estimator correction.
Python >=3.10, NumPy. No model, GPU, remote experiment data, or inference.
All finite examples use scores of explicit categorical policies.
"""
from __future__ import annotations
from itertools import product, combinations, permutations
from math import comb, factorial
from pathlib import Path
import json
import numpy as np

EPS = 1e-6
CHECKS = {}


def check(name, a, b, tol=3e-11):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    CHECKS[name] = err
    if not np.isfinite(err) or err > tol:
        raise AssertionError((name, err, tol))


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for i in range(total + 1):
            for tail in compositions(total - i, parts - 1):
                yield (i,) + tail


def advantage(r, partner_successes, k):
    n = r + partner_successes
    if n in (0, k):
        return 0.
    return (r - n/k) / (np.sqrt(n*(k-n))/k + EPS)


def hypergeom_pmf(population, successes, draws):
    if not 0 <= successes <= population or not 0 <= draws <= population:
        raise ValueError('Invalid hypergeometric parameters.')
    den = choose(population, draws)
    return np.array([choose(successes, n)*choose(population-successes, draws-n)/den
                     for n in range(draws + 1)], dtype=float)


def full_stratified_weights(rewards, k):
    """Complete stratified U estimator: shape B x m, B >= K.
    Output w means G = sum_i w_i S_i/(B*m). Independent samples within
    AND across strata are required for the target-preserving interpretation.
    Partner label counts are Multinomial(K-1, 1/m); each label uses a
    hypergeometric distribution on its remaining actual samples.
    """
    r = np.asarray(rewards)
    if r.ndim != 2 or not np.all((r == 0) | (r == 1)):
        raise ValueError('Expected a binary B x m array.')
    B, m = r.shape
    if not 2 <= k <= B:
        raise ValueError('Require 2 <= K <= B.')
    totals = r.sum(axis=0).astype(int)
    w = np.zeros(r.shape, dtype=float)
    for j in range(m):
        for focal in (0, 1):
            if not np.any(r[:, j] == focal):
                continue
            # Exponential generating function, with w_label^draws=1/m^draws.
            dp = np.zeros((k, k)); dp[0, 0] = 1.
            for ell in range(m):
                pop = B - int(ell == j)
                suc = int(totals[ell]) - int(ell == j)*focal
                local = [(a, hypergeom_pmf(pop, suc, a)/factorial(a)/m**a)
                         for a in range(k)]
                new = np.zeros_like(dp)
                for used in range(k):
                    for a, pmf in local:
                        if used + a >= k:
                            continue
                        for oldn in range(used + 1):
                            new[used+a, oldn:oldn+a+1] += dp[used, oldn]*pmf
                dp = new
            partner = factorial(k-1)*dp[k-1]
            check('partner_pmf_last', partner.sum(), 1.)
            val = sum(partner[n]*advantage(focal, n, k) for n in range(k))
            w[r[:, j] == focal, j] = val
    return w


def cross_weights(rewards, k):
    r = np.asarray(rewards, dtype=int); B, m = r.shape
    out = np.zeros_like(r, dtype=float)
    for b in range(B):
        dp = np.zeros((k, k)); dp[0, 0] = 1.
        for c in range(B):
            if c == b:
                continue
            q = r[c].mean(); new = dp.copy()
            new[1:] += (1-q)*dp[:-1]
            new[1:, 1:] += q*dp[:-1, :-1]
            dp = new
        pmf = dp[k-1]/comb(B-1, k-1)
        for j in range(m):
            out[b, j] = sum(pmf[n]*advantage(int(r[b,j]), n, k) for n in range(k))
    return out


def iid_weights(rewards, k):
    r = np.asarray(rewards, dtype=int)
    n, total = r.size, int(r.sum())
    out = np.zeros_like(r, dtype=float)
    for focal in (0, 1):
        if not np.any(r == focal):
            continue
        pmf = hypergeom_pmf(n-1, total-focal, k-1)
        out[r == focal] = sum(pmf[s]*advantage(focal, s, k) for s in range(k))
    return out


def brute_full_weights(rewards, k):
    r = np.asarray(rewards, dtype=int); B, m = r.shape; out = np.zeros_like(r, dtype=float)
    for ks in compositions(k, m):
        mass = factorial(k)/m**k
        for kj in ks:
            mass /= factorial(kj)
        choices = [list(combinations(range(B), kj)) for kj in ks]
        den = np.prod([comb(B, kj) for kj in ks])
        for subsets in product(*choices):
            ix = [(b,j) for j in range(m) for b in subsets[j]]
            count = sum(int(r[b,j]) for b,j in ix)
            for b,j in ix:
                out[b,j] += mass/den*advantage(int(r[b,j]), count-int(r[b,j]), k)/k
    return B*m*out


def moments(vals, probs):
    vals = np.asarray(vals).reshape(len(probs), -1)
    mean = probs@vals; dev = vals-mean
    return mean, (dev.T*probs)@dev


def kernel_table(p, reward, score, k):
    d = len(p); tab = np.zeros((d,)*k + (score.shape[1],))
    for ys in product(range(d), repeat=k):
        rs = reward[list(ys)]; n = int(rs.sum())
        aa = np.array([advantage(int(v), n-int(v), k) for v in rs])
        tab[ys] = (aa[:,None]*score[list(ys)]).mean(axis=0)
    return tab


def stratified_orthogonal_covariance(table, strata, k, B):
    """Exact multi-sample Hoeffding formula for finite state spaces.
    h_alpha = product_(j,l)(delta_x - P_j) P^(K-|alpha|) H.
    Coefficient c_alpha = (K)_s / (m^s product alpha_j!).
    """
    strata = np.asarray(strata); m, d = strata.shape; p = strata.mean(axis=0)
    reduced = [None]*(k+1); reduced[k] = table
    for s in range(k-1, -1, -1):
        reduced[s] = np.tensordot(reduced[s+1], p, axes=([s],[0]))
    cov = np.zeros((table.shape[-1],)*2); parts = []
    for s in range(1,k+1):
        for alpha in compositions(s,m):
            labels = [j for j,a in enumerate(alpha) for _ in range(a)]
            h = reduced[s].copy()
            probs = np.array(1.)
            for axis,j in enumerate(labels):
                shape = [1]*h.ndim; shape[axis] = d
                h = h-(h*strata[j].reshape(shape)).sum(axis=axis, keepdims=True)
                probs = np.multiply.outer(probs, strata[j])
            flat = h.reshape(-1, h.shape[-1]); pr = probs.reshape(-1)
            second = (flat.T*pr)@flat
            coefficient = factorial(k)/factorial(k-s)/m**s
            denominator = 1
            for a in alpha:
                coefficient /= factorial(a)
                denominator *= comb(B,a)
            term = coefficient**2/denominator*second
            cov += term
            parts.append({'alpha':list(alpha),'trace':float(np.trace(term))})
    return reduced[0], cov, parts


def exact_model(name, strata, reward, k, B):
    strata = np.asarray(strata,dtype=float); m,d = strata.shape
    p = strata.mean(axis=0); score = np.eye(d)-p
    check(name+'_mixture_normalized', strata.sum(axis=1), 1.)
    N=B*m
    banks = np.indices((d,)*N).reshape(N,-1).T
    rewards = np.asarray(reward,dtype=int)[banks]
    bits = 1<<np.arange(N)
    masks = rewards@bits
    lookups = {}
    for method,fn in [('full',full_stratified_weights),('cross',cross_weights),('iid',iid_weights)]:
        lookup = np.array([fn(((n & bits)>0).astype(int).reshape(B,m),k).reshape(N)
                           for n in range(2**N)])
        lookups[method] = lookup
    probs_strat = np.prod(strata[np.arange(N)%m,banks],axis=1)
    probs_iid = np.prod(p[banks],axis=1)
    values={}
    for method,lookup in lookups.items():
        w=lookup[masks]
        # Sum scores without allocating bank x N x d.
        val=np.empty((len(banks),d))
        for y in range(d):
            val[:,y]=((banks==y)*w).sum(axis=1)/N-p[y]*w.mean(axis=1)
        values[method]=val
    mf,cf=moments(values['full'],probs_strat)
    mc,cc=moments(values['cross'],probs_strat)
    mi,ci=moments(values['iid'],probs_iid)
    h=kernel_table(p,np.asarray(reward),score,k)
    target,formula,parts=stratified_orthogonal_covariance(h,strata,k,B)
    check(name+'_full_mean',mf,target)
    check(name+'_cross_mean',mc,target)
    check(name+'_iid_mean',mi,target)
    check(name+'_full_covariance_formula',cf,formula)
    # Rao-Blackwell residual orthogonal to conditional expectation.
    _,res=moments(values['cross']-values['full'],probs_strat)
    check(name+'_conditional_variance_identity',cc-cf,res)
    mineig=float(np.linalg.eigvalsh(cc-cf).min())
    if mineig < -3e-11:
        raise AssertionError('Rao-Blackwell dominance failed')
    return {'name':name,'K':k,'B':B,'m':m,'enumerated_banks':len(banks),
            'target':target.tolist(),'trace_full':float(np.trace(cf)),
            'trace_cross':float(np.trace(cc)),'trace_iid_all':float(np.trace(ci)),
            'full_over_cross':float(np.trace(cf)/np.trace(cc)),
            'full_over_iid_all':float(np.trace(cf)/np.trace(ci)),
            'min_eigenvalue_cross_minus_full':mineig,'orthogonal_parts':parts}


def chi_penalty(s,B,m):
    val=0.
    for a in compositions(s,m):
        pm=factorial(s)/m**s
        ph=1./comb(B*m,s)
        for aj in a:
            pm/=factorial(aj); ph*=comb(B,aj)
        val+=(pm-ph)**2/ph
    return val


def baseline_check():
    # Categorical p, deterministic reward; fixed direction score z has zero mean.
    p=np.array([.4,.3,.2,.1]); r=np.array([1.,1.,0.,0.])
    d=np.array([.8,-.7,1.4,-1.]); z=d-p@d
    n=4; ids=np.array(list(product(range(4),repeat=n)))
    mass=np.prod(p[ids],axis=1); rs=r[ids]; zs=z[ids]
    a=rs-(rs.sum(axis=1,keepdims=True)-rs)/(n-1)
    values=(a*zs).mean(axis=1)
    g=float(p@(r*z)); pr=float(p@r)
    oracle=(r-pr)*z
    var_oracle=float(p@(oracle-g)**2)
    formula=var_oracle/n+(pr*(1-pr)*float(p@z**2)+g*g)/(n*(n-1))
    mean,cov=moments(values,mass)
    check('rloo_mean',mean,g);check('rloo_exact_variance',cov,formula)
    fixed1=((rs-1)*zs).mean(axis=1)
    check('fixed_one_baseline_mean',mass@fixed1,g)
    var1=(float(p@((r-1)*z)**2)-g*g)/n
    check('fixed_one_baseline_variance',moments(fixed1,mass)[1],var1)
    return {'target':g,'rloo_variance':formula,'fixed_one_variance':var1}


def permutation_check():
    r=np.array([[0,1],[1,0],[1,1],[0,0]])
    rng=np.random.default_rng(52026); scores=rng.normal(size=(4,2,3))
    original=(full_stratified_weights(r,4)[...,None]*scores).mean(axis=(0,1))
    # Fix first column; average all relative permutations of other column.
    vals=[]
    for perm in permutations(range(4)):
        rr=r.copy(); ss=scores.copy(); rr[:,1]=r[list(perm),1];ss[:,1]=scores[list(perm),1]
        vals.append((cross_weights(rr,4)[...,None]*ss).mean(axis=(0,1)))
    check('permutation_average_vs_full',np.mean(vals,axis=0),original)


def invalid_dependence_example():
    # Each block (R_b, R_b) copies one Bernoulli. Blocks independent.
    # The full-stratum formula is not valid for these dependent strata.
    vals=[];full=[]
    for a,b in product((0,1),repeat=2):
        r=np.array([[a,a],[b,b]]); s=r-.5
        vals.append(float((cross_weights(r,2)*s).mean()))
        full.append(float((full_stratified_weights(r,2)*s).mean()))
    target=.25/(1+2*EPS)
    check('dependent_blocks_cross_valid',np.mean(vals),target)
    if abs(np.mean(full)-target)<1e-5:
        raise AssertionError('Expected dependence counterexample')
    return {'target':target,'cross_mean':float(np.mean(vals)),
            'invalid_full_mean':float(np.mean(full))}



def grid_advantage(focal, partner_sum, partner_square_sum, k, scale):
    total=focal+partner_sum; sq=focal*focal+partner_square_sum
    disc=k*sq-total*total
    if disc < 0:
        raise AssertionError('Negative exact integer variance')
    if disc == 0:
        return 0.
    return (focal-total/k)/(np.sqrt(disc)/k+scale*EPS)


def subset_grid_laws(values, k):
    """Exact subset-count polynomial, one column of integer-grid rewards."""
    values=[int(v) for v in values]; dp={(0,0,0):1}
    for v in values:
        out=dict(dp)
        for (a,t,q),count in dp.items():
            if a+1<k:
                key=(a+1,t+v,q+v*v)
                out[key]=out.get(key,0)+count
        dp=out
    return {(a,t,q):count/comb(len(values),a) for (a,t,q),count in dp.items()}


def full_stratified_grid_weights(rewards, k, scale):
    """Integer rewards represent reward/scale. Return average A, A+, A-.
    Reference implementation; state count can grow with K and grid size.
    No quantization of a continuous reward is performed here.
    """
    r=np.asarray(rewards)
    if (r.ndim!=2 or not np.issubdtype(r.dtype,np.integer)
        or np.any(r<0) or np.any(r>scale) or scale<1):
        raise ValueError('Expected integer B x m rewards in [0,scale].')
    B,m=r.shape
    if not 2<=k<=B:
        raise ValueError('Require 2 <= K <= B.')
    w=np.zeros(r.shape+(3,))
    for j in range(m):
        for focal in np.unique(r[:,j]):
            dp={(0,0,0):1.}
            for ell in range(m):
                vals=list(map(int,r[:,ell]))
                if ell==j: vals.remove(int(focal))
                local=subset_grid_laws(vals,k); out={}
                for (a,t,q),mass in dp.items():
                    for (b,u,v),prob in local.items():
                        if a+b>=k: continue
                        key=(a+b,t+u,q+v)
                        out[key]=out.get(key,0.)+mass*prob/(m**b*factorial(b))
                dp=out
            total_prob=0.; sums=np.zeros(3)
            for (a,t,q),mass in dp.items():
                if a!=k-1: continue
                prob=mass*factorial(k-1); total_prob+=prob
                aa=grid_advantage(int(focal),t,q,k,scale)
                sums+=prob*np.array([aa,max(aa,0.),min(aa,0.)])
            check('grid_partner_pmf_last',total_prob,1.)
            w[r[:,j]==focal,j]=sums
    return w


def brute_grid_weights(r,k,scale):
    B,m=r.shape; out=np.zeros(r.shape+(3,))
    for ks in compositions(k,m):
        mass=factorial(k)/m**k
        for kj in ks: mass/=factorial(kj)
        choices=[list(combinations(range(B),kj)) for kj in ks]
        den=np.prod([comb(B,kj) for kj in ks])
        for subsets in product(*choices):
            ix=[(b,j) for j in range(m) for b in subsets[j]]
            total=sum(int(r[b,j]) for b,j in ix)
            sq=sum(int(r[b,j])**2 for b,j in ix)
            for b,j in ix:
                focal=int(r[b,j]); aa=grid_advantage(focal,total-focal,sq-focal*focal,k,scale)
                out[b,j]+=mass/den/k*np.array([aa,max(aa,0.),min(aa,0.)])
    return B*m*out


def grid_and_clipping_checks():
    examples=[(np.array([[0,1],[1,0]]),2,1),
              (np.array([[0,3],[1,2],[2,1],[3,0]]),4,3),
              (np.array([[0,2,1],[2,1,0],[1,0,2]]),3,2)]
    errors=[]
    for idx,(r,k,L) in enumerate(examples):
        w=full_stratified_grid_weights(r,k,L); brute=brute_grid_weights(r,k,L)
        check(f'grid_dp_vs_brute_{idx}',w,brute)
        check(f'grid_zero_sum_{idx}',w[...,0].sum(),0.)
        check(f'grid_sign_sum_{idx}',w[...,1]+w[...,2],w[...,0])
        rho=np.linspace(.6,1.6,r.size).reshape(r.shape)
        clip=np.clip(rho,.8,1.2)
        compiled=w[...,1]*np.minimum(rho,clip)+w[...,2]*np.maximum(rho,clip)
        # Direct average each virtual group's clipped contribution per sample.
        B,m=r.shape; direct=np.zeros(r.shape); direct_grad=np.zeros(r.shape)
        for ks in compositions(k,m):
            mass=factorial(k)/m**k
            for kj in ks: mass/=factorial(kj)
            choices=[list(combinations(range(B),kj)) for kj in ks]
            den=np.prod([comb(B,kj) for kj in ks])
            for subsets in product(*choices):
                ix=[(b,j) for j in range(m) for b in subsets[j]]
                total=sum(int(r[b,j]) for b,j in ix); sq=sum(int(r[b,j])**2 for b,j in ix)
                for b,j in ix:
                    a=int(r[b,j]); adv=grid_advantage(a,total-a,sq-a*a,k,L)
                    coeff=B*m*mass/den/k
                    direct[b,j]+=coeff*min(rho[b,j]*adv,clip[b,j]*adv)
                    derivative=adv if (adv>=0 and rho[b,j]<1.2) or (adv<0 and rho[b,j]>.8) else 0.
                    direct_grad[b,j]+=coeff*derivative
        check(f'grid_clipped_loss_{idx}',compiled,direct)
        compiled_grad=w[...,1]*(rho<1.2)+w[...,2]*(rho>.8)
        # The sample ratios used above avoid clipping kinks.
        if np.any(np.isclose(rho,.8)) or np.any(np.isclose(rho,1.2)):
            mask=~(np.isclose(rho,.8)|np.isclose(rho,1.2))
        else: mask=np.ones(rho.shape,dtype=bool)
        check(f'grid_clipped_derivative_{idx}',compiled_grad[mask],direct_grad[mask])
        errors.append(float(np.max(abs(w-brute))))
    return {'cases':len(examples),'max_dp_error':max(errors),
            'continuous_rewards_quantized':False,'clip_kinks_excluded_from_derivative_check':True}


def diagnostic_identity_checks():
    # Bilinear independent-bank variance checked by complete discrete enumeration.
    ug=np.array([[.2,-.4],[-.2,.4]]); ud=np.array([[.5,.1],[-.5,-.1]])
    g=np.array([.7,-.3]); d=np.array([.2,.8])
    Vg=ug.T@ug/2; Vd=ud.T@ud/2
    vals=np.array([(g+a)@(d+b) for a in ug for b in ud])
    formula=d@Vg@d+g@Vd@g+np.trace(Vg@Vd)
    check('two_bank_bilinear_mean',vals.mean(),g@d)
    check('two_bank_bilinear_variance',vals.var(),formula)
    # Exact polynomial isolates common-direction curvature; not an LLM check.
    F=lambda x: np.array([x[0]**2+3*x[0]*x[1]+2*x[1]**2,x[0]**3])
    x=np.array([.4,-.2]); ds=np.array([2.,.3]); di=np.array([1.7,-.4])
    delta=ds-di; midpoint=(ds+di)/2
    J=np.array([[2*x[0]+3*x[1],3*x[0]+4*x[1]],[3*x[0]**2,0.]])
    Hess=np.array([[[2.,3.],[3.,4.]],[[6*x[0],0.],[0.,0.]]])
    h=.015
    D=lambda t:F(x+t*ds)-F(x+t*di)
    even=(D(h)+D(-h))/2
    check('common_curvature_even_part',even,h*h*np.einsum('i,aij,j->a',midpoint,Hess,delta))
    pure=(F(x+h*delta/2)-F(x-h*delta/2))/h
    expected=J@delta+np.array([0.,h*h*delta[0]**3/4])
    check('pure_central_difference_polynomial',pure,expected)
    return {'bilinear_variance':float(formula),'polynomial_only':True}



def equal_reward_influence_checks():
    answers=[]
    for name,strata in [('identity',np.array([[.5,0,.5,0],[0,.5,0,.5]])),
                        ('uninformative',np.ones((2,4))/4)]:
        r=np.array([1,1,0,0]); p=strata.mean(axis=0); pr=float(p@r)
        score=np.eye(4)-p; k=4
        tab=kernel_table(p,r,score,k); cond=tab
        for axis in range(k-1,0,-1):
            cond=np.tensordot(cond,p,axes=([axis],[0]))
        target=p@cond; psi=cond-target
        u=[]
        for focal in (0,1):
            u.append(sum(comb(k-1,n)*pr**n*(1-pr)**(k-1-n)*advantage(focal,n,k)
                         for n in range(k)))
        g=(p*r)@score
        predicted=(u[1]-u[0])/k*(strata@((r-pr)[:,None]*score)-g)
        check('equal_reward_influence_'+name,strata@psi,predicted)
        answers.append({'name':name,'stratum_influence_means':(strata@psi).tolist()})
    return answers


def main():
    rng=np.random.default_rng(62026)
    for B,m,k in [(2,2,2),(4,2,4),(4,3,4),(5,2,3)]:
        r=rng.integers(0,2,size=(B,m))
        check(f'dp_bruteforce_B{B}_m{m}_K{k}',full_stratified_weights(r,k),brute_full_weights(r,k))
        check(f'zero_sum_B{B}_m{m}_K{k}',full_stratified_weights(r,k).sum(),0.)
    permutation_check()
    models=[]
    for q in [(.2,.8),(.5,.5)]:
        strata=[[1-qj,qj] for qj in q]
        models.append(exact_model(f'K2_q{q}',strata,[0,1],2,2))
    for q in [(.2,.8),(.5,.5),(.1,.6)]:
        strata=[[1-qj,qj] for qj in q]
        models.append(exact_model(f'K4_q{q}',strata,[0,1],4,4))
    models.append(exact_model('K4_identity_strata',[[.5,0,.5,0],[0,.5,0,.5]],[1,1,0,0],4,4))
    models.append(exact_model('K4_reward_strata',[[.5,.5,0,0],[0,0,.5,.5]],[1,1,0,0],4,4))
    check('uninformative_K2_chi_ratio',models[1]['full_over_iid_all'],1+chi_penalty(2,2,2))
    result={'status':'PASS_CPU_FINITE_MODEL_ONLY','models':models,
            'baseline':baseline_check(),'dependence_counterexample':invalid_dependence_example(),
            'grid_and_clipping':grid_and_clipping_checks(),
            'diagnostic_identities':diagnostic_identity_checks(),
            'equal_reward_influence':equal_reward_influence_checks(),
            'uninformative_chi_penalties':{str(s):chi_penalty(s,4,2) for s in range(1,5)},
            'report_derived_missing_projection':{'T_fixed1':'-z_fail/32',
                'T_RLOO':'-0.7801849729962018 - z_fail/32',
                'RLOO_positive_requires_z_fail_below':-32*.7801849729962018,
                'z_fail_measured_here':False},
            'checks':CHECKS,'max_identity_error':max(CHECKS.values()),
            'limitations':['No real model loaded','No remote v4 receipts independently checked',
                           'No GPU experiments','Mathematical examples do not establish real-model effect size']}
    path=Path(__file__).with_name('verify_v5_1_results.json')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()

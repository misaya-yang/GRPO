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

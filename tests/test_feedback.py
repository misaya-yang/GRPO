"""Independent CPU enumeration and protocol failures, not pretrained evidence."""

import itertools
import json
import sys

import numpy as np
import pytest

from dependent_rollouts.artifacts import write_json
from reward_coupling.advantages import advantages
from reward_coupling.budget import run_budgeted
from reward_coupling.expectation import (
    expected_advantages,
    independent_binary,
    independent_finite,
    independent_grid,
    rubric_matrix,
)
from reward_coupling.statistics import (
    pilot_decision,
    positive_scale,
    projected_two_bank_variance,
    two_bank_bootstrap,
)
from reward_coupling.theory import additive_decomposition, binary_population, coupling_envelope


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_binary_dp_enumeration_and_ties(epsilon):
    rng = np.random.default_rng(42)
    for k in range(2, 9):
        q = rng.uniform(size=k)
        q[:2] = [0, 1]
        states = np.array(list(itertools.product([0, 1], repeat=k)))
        mass = np.where(states, q, 1 - q).prod(axis=1)
        np.testing.assert_allclose(
            independent_binary(q, epsilon), mass @ advantages(states, epsilon=epsilon), atol=2e-14
        )
        for constant in (0.1, 1 / 3, 1):
            np.testing.assert_allclose(advantages(np.full(k, constant), epsilon=epsilon), 0, atol=0)


def test_shared_independent_negative_controls_and_equal_marginal_credit():
    rng = np.random.default_rng(1)
    for k in (2, 3, 4, 8):
        b = rng.integers(0, 2, size=(k, 7))
        mu = rng.dirichlet(np.ones(7))
        a = expected_advantages(b, mu)
        if k < 4:
            np.testing.assert_allclose(a["shared"], a["independent"], atol=2e-14)
        r = expected_advantages(b, mu, kind="rloo")
        np.testing.assert_allclose(r["shared"], r["independent"], atol=2e-14)
    b = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    a = expected_advantages(b)
    np.testing.assert_allclose(a["independent"], 0, atol=2e-14)
    np.testing.assert_allclose(a["shared"], [0.03867496793] * 2 + [-0.03867496793] * 2, atol=1e-11)


@pytest.mark.parametrize("k,nv", [(2, 3), (3, 4), (4, 3), (5, 4)])
def test_grid_dp_against_independent_enumeration(k, nv):
    p = np.random.default_rng(k).dirichlet(np.ones(nv), size=k)
    exact = np.zeros(k)
    for state in itertools.product(range(nv), repeat=k):
        exact += np.prod(p[np.arange(k), state]) * advantages(np.array(state) / (nv - 1))
    np.testing.assert_allclose(independent_grid(p, nv - 1), exact, atol=1e-13)


def test_continuous_minimum_k_and_rubric_replay():
    a = expected_advantages([[1, 0], [0, 0.2]], epsilon=0, grid_scale=5)
    np.testing.assert_allclose(a["shared"], [0, 0], atol=1e-14)
    np.testing.assert_allclose(a["independent"], [0.25, -0.25])
    b = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0]])
    matrix, masks = rubric_matrix(b, 2)
    assert len(masks) == 6
    np.testing.assert_allclose(matrix.mean(axis=1), b.mean(axis=1))
    a = expected_advantages(matrix, grid_scale=2)
    np.testing.assert_allclose(
        a["independent"], independent_finite(matrix, np.ones(6) / 6), atol=2e-14
    )
    with pytest.raises(ValueError, match="declared exact grid"):
        expected_advantages([[0, 0.213], [1, 0.72]], grid_scale=2)


@pytest.mark.parametrize("k,m", [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)])
def test_complete_finite_invariant_function_space(k, m):
    # Solve the entire finite table constraint system, not sampled functions.
    states = list(itertools.product(range(m), repeat=k))
    index = {s: n for n, s in enumerate(states)}
    n = len(states)
    constraints = []

    def equation(terms):
        row = np.zeros(k * n)
        for i, state, coefficient in terms:
            row[i * n + index[tuple(state)]] += coefficient
        constraints.append(row)

    for s in states:
        equation([(i, s, 1) for i in range(k)])
        for a in range(k - 1):
            t = list(s)
            t[a], t[a + 1] = t[a + 1], t[a]
            permutation = list(range(k))
            permutation[a], permutation[a + 1] = permutation[a + 1], permutation[a]
            for i in range(k):
                equation([(i, t, 1), (permutation[i], s, -1)])
        for a, b in itertools.combinations(range(k), 2):
            t, u, v = list(s), list(s), list(s)
            t[a], u[b] = 0, 0
            v[a] = v[b] = 0
            for i in range(k):
                equation([(i, s, 1), (i, t, -1), (i, u, -1), (i, v, 1)])
    rank = np.linalg.matrix_rank(np.array(constraints), tol=1e-9)
    assert k * n - rank == m - 1  # f up to an additive constant.


def test_interaction_decomposition_and_fixed_marginal_lp():
    for k in (2, 3, 4):
        states = np.array(list(itertools.product([0, 1], repeat=k)))
        table = advantages(states)[:, 0].reshape((2,) * k)
        marginal, interaction = additive_decomposition(table)
        np.testing.assert_allclose(marginal + interaction, table)
        if k < 4:
            np.testing.assert_allclose(interaction, 0, atol=1e-14)
        else:
            assert np.linalg.norm(interaction) > 0.01
    pytest.importorskip("scipy")
    for k in (2, 3, 4):
        p = np.tile([0.5, 0.5], (k, 1))
        bounds = coupling_envelope([0, 1], p, np.arange(k))
        assert bounds["width"] < 1e-10 if k < 4 else bounds["width"] > 0.01
        assert coupling_envelope([0, 1], p, np.arange(k), "rloo")["width"] < 1e-10


def test_reversal_is_an_existence_example():
    b = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
    pi = np.array([0.005, 0.005, 0.495, 0.495])
    s, i = binary_population(b, pi, [0.17, 0.23, 0.3, 0.3], 8)
    g = (np.diag(pi) - np.outer(pi, pi)) @ [1, 1, 0, 0]
    assert s @ g < 0 < i @ g


def test_statistics_preserve_both_uncertainties_and_reverse_collinearity():
    h = np.arange(12).reshape(3, 4)
    interval = two_bank_bootstrap(h, replicates=100)
    assert interval["lower"] < h.mean() < interval["upper"]
    report = projected_two_bank_variance(h.mean(1), h.mean(0))
    assert report["status"] == "interaction_unresolved_no_full_CI"
    report = positive_scale([-1, -2], [1, 2])
    assert report["signed_scale"] == -1 and report["residual"] > 0
    assert positive_scale([1], [0])["status"] == "unresolved_reference"
    assert pilot_decision({}) == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="Missing evidence"):
        pilot_decision({"gates": {"implementation": {"status": "PASS"}}})


def test_hard_budget_and_interrupted_reservation(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    report = run_budgeted([sys.executable, "-c", "import time; time.sleep(2)"], ledger, 0.1)
    assert report["status"] == "budget_exhausted"
    with pytest.raises(TimeoutError):
        run_budgeted([sys.executable, "-c", "pass"], ledger, 0.1)
    ledger = tmp_path / "interrupted.jsonl"
    ledger.write_text(json.dumps({"event": "start"}) + "\n")
    with pytest.raises(RuntimeError, match="Unclosed"):
        run_budgeted([sys.executable, "-c", "pass"], ledger, 1)
    with pytest.raises(TimeoutError):
        run_budgeted(
            [sys.executable, "-c", "pass"], tmp_path / "expired.jsonl", 10, deadline_epoch=1
        )


def test_frozen_sources_unchanged():
    from pathlib import Path

    from dependent_rollouts.artifacts import sha256

    sources = json.loads(Path("docs/theory/feedback_sources.json").read_text())
    for path, expected in sources.items():
        assert sha256(path) == expected


def test_invalid_probabilities_and_nonfinite_advantages():
    for q in ([0.1, float("nan")], [0.1, 2]):
        with pytest.raises(ValueError):
            independent_binary(q)
    with pytest.raises(ValueError):
        expected_advantages([[1, 0], [0, 1]], [0.5, 0.51])
    with pytest.raises(ValueError):
        advantages([0, float("inf")])


def test_contract_refuses_unpinned_model(tmp_path):
    from reward_coupling.contract import preflight

    config = json.loads(__import__("pathlib").Path("configs/feedback/pilot_code.json").read_text())
    report = preflight(config)
    assert report["status"] == "BLOCKED" and report["pilot_decision"] == "NOT_RUN"
    write_json(tmp_path / "preflight.json", report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("executor", "unknown"),
        ("test_timeout_seconds", 0),
        ("test_timeout_seconds", 61),
        ("stage_max_seconds", float("inf")),
        ("token_logp_tolerance", -1),
    ],
)
def test_contract_rejects_invalid_runtime_settings(key, value):
    from pathlib import Path

    from reward_coupling.contract import validate_config

    config = json.loads(Path("configs/feedback/pilot_code.json").read_text())
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, require_pins=False)

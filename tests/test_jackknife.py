"""
Tests for ldsc_jx.jackknife

We test five things:
1. Known coefficients: does LstsqJackknifeFast recover true beta from y = Xb + noise?
2. Fast vs manual: do the vmap-computed delete values match a naive leave-one-block-out loop?
3. Pseudovalue identity: does pseudovalues == n_blocks * est - (n_blocks - 1) * delete_values?
4. RatioJackknife: does it recover a known ratio with reasonable SE?
5. Input validation: do bad shapes raise ValueError?
"""

import numpy as np
import pytest
from ldsc_jx.jackknife import (
    LstsqJackknifeFast,
    RatioJackknife,
    get_separators,
    delete_values_to_pseudovalues,
    _check_shape,
)


# ---------- fixtures ----------

@pytest.fixture
def regression_data():
    """
    Simulate y = 2*x1 - 1*x2 + noise.
    n=1000 observations, p=2 predictors, noise sd=0.5.
    Returns x (1000,2), y (1000,1), and true_beta [2, -1].
    """
    rng = np.random.default_rng(42)
    n, p = 1000, 2
    x = rng.standard_normal((n, p))
    true_beta = np.array([2.0, -1.0])
    y = (x @ true_beta + rng.standard_normal(n) * 0.5).reshape((n, 1))
    return x, y, true_beta


N_BLOCKS = 200


# ---------- test 1: recover known coefficients ----------

def test_recovers_true_beta(regression_data):
    """
    With low noise and 1000 data points, the jackknife estimate
    should be close to the true coefficients [2, -1].
    """
    x, y, true_beta = regression_data
    jk = LstsqJackknifeFast(x, y, n_blocks=N_BLOCKS)

    # point estimate should be within 0.05 of true values
    np.testing.assert_allclose(jk.est.ravel(), true_beta, atol=0.05)

    # SE should be positive and small (noise_sd / sqrt(n) ~ 0.016)
    assert np.all(jk.jknife_se > 0)
    assert np.all(jk.jknife_se < 0.1)


# ---------- test 2: fast delete values match manual leave-one-block-out ----------

def test_delete_values_match_manual(regression_data):
    """
    Compute delete values the naive way (remove each block, re-run lstsq)
    and verify they match the fast subtraction-based approach.
    """
    x, y, _ = regression_data
    jk = LstsqJackknifeFast(x, y, n_blocks=N_BLOCKS)
    s = jk.separators
    p = x.shape[1]

    # manual leave-one-block-out
    manual_delete = np.zeros((N_BLOCKS, p))
    for i in range(N_BLOCKS):
        # remove block i by concatenating everything before and after it
        x_del = np.vstack([x[:s[i]], x[s[i + 1]:]])
        y_del = np.vstack([y[:s[i]], y[s[i + 1]:]])
        manual_delete[i] = np.linalg.lstsq(x_del, y_del, rcond=None)[0].ravel()

    np.testing.assert_allclose(jk.delete_values, manual_delete, atol=1e-6)


# ---------- test 3: pseudovalue identity ----------

def test_pseudovalue_identity(regression_data):
    """
    Verify the pseudovalue formula:
    pseudovalues = n_blocks * est - (n_blocks - 1) * delete_values
    """
    x, y, _ = regression_data
    jk = LstsqJackknifeFast(x, y, n_blocks=N_BLOCKS)

    expected = N_BLOCKS * jk.est - (N_BLOCKS - 1) * jk.delete_values
    np.testing.assert_allclose(jk.pseudovalues, expected, atol=1e-10)


# ---------- test 4: RatioJackknife recovers known ratio ----------

def test_ratio_jackknife():
    """
    If numer ~ 2 * denom with small noise, the ratio estimate should
    be close to 2.0 with a small SE.
    """
    rng = np.random.default_rng(99)
    n_blocks = 200

    # denominator delete values: random positive values
    denom = rng.uniform(1.0, 3.0, (n_blocks, 1))
    # numerator: 2x denominator plus very small noise
    numer = 2.0 * denom + rng.standard_normal((n_blocks, 1)) * 0.01

    est = np.array([[2.0]])
    rjk = RatioJackknife(est, numer, denom)

    # est is the raw ratio (what we passed in)
    np.testing.assert_allclose(rjk.est, 2.0)
    # jknife_est (bias-corrected) should be close to 2.0 with low noise
    assert abs(rjk.jknife_est.item() - 2.0) < 0.05
    # SE should be positive and small
    assert rjk.jknife_se.item() > 0
    assert rjk.jknife_se.item() < 0.1


# ---------- test 5: input validation ----------

def test_check_shape_rejects_1d():
    """1D arrays should be rejected."""
    with pytest.raises(ValueError, match="2D"):
        _check_shape(np.ones(5), np.ones(5))


def test_check_shape_rejects_wrong_y_cols():
    """y must have exactly 1 column."""
    with pytest.raises(ValueError, match="n_snp, 1"):
        _check_shape(np.ones((5, 2)), np.ones((5, 2)))


def test_check_shape_rejects_mismatched_rows():
    """x and y must have the same number of rows."""
    with pytest.raises(ValueError, match="Number of datapoints"):
        _check_shape(np.ones((5, 2)), np.ones((3, 1)))


def test_rejects_more_blocks_than_data():
    """n_blocks > n should raise."""
    x = np.ones((10, 2))
    y = np.ones((10, 1))
    with pytest.raises(ValueError):
        LstsqJackknifeFast(x, y, n_blocks=20)


def test_ratio_rejects_shape_mismatch():
    """Numerator and denominator must have the same shape."""
    with pytest.raises(ValueError, match="shape"):
        RatioJackknife(
            np.array([[1.0]]),
            np.ones((10, 1)),
            np.ones((5, 1)),
        )

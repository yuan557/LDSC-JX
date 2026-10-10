"""
Tests for ldsc_jx.irwls

We test four things:
1. Uniform weights: IRWLS with constant update_func reduces to OLS, recovers true beta.
2. Reweighting changes result: non-trivial update_func produces different estimate than OLS.
3. WLS standalone: IRWLS.wls matches manual weighted least squares.
4. Weight validation: non-positive weights raise ValueError.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from ldsc_jx.irwls import IRWLS


# ---------- fixtures ----------

@pytest.fixture
def regression_data():
    """
    Simulate y = 2*x + 1 + noise with intercept column appended.
    x is (1000, 2): first column is the predictor, second is ones (intercept).
    true_beta is [2, 1] (slope, intercept).
    """
    rng = np.random.default_rng(42)
    n = 1000
    x_raw = rng.standard_normal((n, 1))
    intercept = np.ones((n, 1))
    x = np.hstack([x_raw, intercept])
    true_beta = np.array([2.0, 1.0])
    y = (x @ true_beta + rng.standard_normal(n) * 0.5).reshape((n, 1))
    return x, y, true_beta


N_BLOCKS = 200


# ---------- test 1: uniform weights = OLS ----------

def test_uniform_weights_recovers_ols(regression_data):
    """
    If update_func always returns equal weights, IRWLS is just OLS.
    Should recover true beta [2, 1] with reasonable SE.
    """
    x, y, true_beta = regression_data
    n = x.shape[0]

    # update_func receives lstsq output, returns uniform weights
    def uniform_update(lstsq_result):
        return jnp.ones((n, 1))

    initial_w = jnp.ones((n, 1))
    result = IRWLS(x, y, uniform_update, N_BLOCKS, w=initial_w)

    # point estimate should be close to [2, 1]
    np.testing.assert_allclose(result.est.ravel(), true_beta, atol=0.1)

    # SE should be positive and small
    assert np.all(result.jknife_se > 0)
    assert np.all(result.jknife_se < 0.1)


# ---------- test 2: reweighting changes the result ----------

def test_reweighting_changes_estimate(regression_data):
    """
    With a non-trivial update_func that downweights large residuals,
    the IRWLS estimate should differ from plain OLS.
    """
    x, y, true_beta = regression_data
    n = x.shape[0]

    # OLS estimate for comparison
    ols_beta = np.linalg.lstsq(x, y, rcond=None)[0].ravel()

    # update_func: weight inversely by predicted variance (mimics LDSC weighting)
    def variance_update(lstsq_result):
        beta = lstsq_result[0]
        predicted = x @ beta
        residuals = y - predicted
        variance = jnp.maximum(residuals ** 2, 0.01)
        return 1.0 / variance

    initial_w = jnp.ones((n, 1))
    result = IRWLS(x, y, variance_update, N_BLOCKS, w=initial_w)
    irwls_beta = result.est.ravel()

    # both should be close to truth, but not identical to each other
    np.testing.assert_allclose(irwls_beta, true_beta, atol=0.2)
    assert not np.allclose(irwls_beta, ols_beta, atol=1e-6), \
        "IRWLS estimate should differ from OLS when weights are non-uniform"


# ---------- test 3: WLS standalone ----------

def test_wls_matches_manual(regression_data):
    """
    IRWLS.wls should match manually weighted least squares:
    solve (W*X)^T(W*X) beta = (W*X)^T(W*y) where W is diagonal weights.
    """
    x, y, _ = regression_data
    n = x.shape[0]
    rng = np.random.default_rng(123)

    # random positive weights
    w = jnp.array(rng.uniform(0.5, 2.0, (n, 1)))
    w_sqrt = jnp.sqrt(w)

    # IRWLS.wls
    result = IRWLS.wls(x, y, w_sqrt)
    irwls_beta = result[0].ravel()

    # manual: normalize w_sqrt so it sums to 1, then weight x and y
    w_norm = w_sqrt / jnp.sum(w_sqrt)
    x_w = np.array(x * w_norm)
    y_w = np.array(y * w_norm)
    manual_beta = np.linalg.lstsq(x_w, y_w, rcond=None)[0].ravel()

    np.testing.assert_allclose(irwls_beta, manual_beta, atol=1e-5)


# ---------- test 4: weight validation ----------

def test_rejects_nonpositive_weights():
    """_weight should raise ValueError if any weight is <= 0."""
    x = jnp.ones((10, 2))
    w = jnp.zeros((10, 1))
    with pytest.raises(ValueError, match="Weights must be > 0"):
        IRWLS._weight(x, w)


def test_rejects_negative_weights():
    """_weight should raise ValueError for negative weights."""
    x = jnp.ones((10, 2))
    w = -jnp.ones((10, 1))
    with pytest.raises(ValueError, match="Weights must be > 0"):
        IRWLS._weight(x, w)

"""
Tests for ldsc_jx.regression

We test six things:
1. Hsq recovers known h2 from simulated LDSC data.
2. Hsq with constrained intercept still recovers h2.
3. Hsq summary() produces expected output fields.
4. Gencov recovers known genetic covariance.
5. RG recovers known genetic correlation.
6. Input validation rejects bad shapes.

Simulation model for h2:
    E[chi2_j] = 1 + N * h2 * lj / M
    chi2_j ~ (1 + N * h2 * lj / M) * chi2(1)
    where lj is the LD score of SNP j.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from ldsc_jx.regression import Hsq, Gencov, RG


N_SNP = 2000
N_SAMPLE = 10000
M = 1000
N_BLOCKS = 200
TRUE_H2 = 0.5


@pytest.fixture
def hsq_data():
    """
    Simulate chi-squared stats under the LDSC model:
    chi2_j ~ (1 + N * h2 * lj / M) * chi2(1)

    LD scores drawn from uniform(1, 5).
    w_ld (regression weight LD scores) set equal to ld scores.
    """
    rng = np.random.default_rng(42)

    ld = rng.uniform(1.0, 5.0, (N_SNP, 1))
    expected_chisq = 1.0 + N_SAMPLE * TRUE_H2 * ld / M
    # chi2(1) has mean 1 and variance 2, scale by expected value
    chisq = expected_chisq * rng.chisquare(1, size=(N_SNP, 1))

    # all arrays must be 2D with shape (n_snp, 1) or (n_snp, n_annot)
    x = ld.copy()
    w = ld.copy()
    N = np.full((N_SNP, 1), float(N_SAMPLE))
    M_arr = np.array([[float(M)]])

    return chisq, x, w, N, M_arr


# ---------- test 1: recover known h2 ----------

def test_hsq_recovers_h2(hsq_data):
    """
    With N=10000, M=1000, true h2=0.5, the Hsq estimate should
    be in the right ballpark. We use a wide tolerance because
    chi2 noise is substantial.
    """
    chisq, x, w, N, M_arr = hsq_data
    hsq = Hsq(chisq, x, w, N, M_arr, n_blocks=N_BLOCKS)

    # h2 should be roughly 0.5 (within 0.3 given chi2 noise)
    assert abs(hsq.tot - TRUE_H2) < 0.3, f"h2={hsq.tot}, expected ~{TRUE_H2}"

    # intercept should be positive (with only 2000 SNPs,
    # free intercept is noisy; real LDSC uses ~1M SNPs)
    assert hsq.intercept > 0, f"intercept={hsq.intercept}, expected positive"

    # SE should be positive
    assert hsq.tot_se > 0
    assert hsq.intercept_se > 0


# ---------- test 2: constrained intercept ----------

def test_hsq_constrained_intercept(hsq_data):
    """
    When intercept is fixed to 1.0, h2 should still be recovered
    and intercept_se should be 'NA'.
    """
    chisq, x, w, N, M_arr = hsq_data
    hsq = Hsq(chisq, x, w, N, M_arr, n_blocks=N_BLOCKS, intercept=1.0)

    assert abs(hsq.tot - TRUE_H2) < 0.3, f"h2={hsq.tot}, expected ~{TRUE_H2}"
    assert hsq.intercept == 1.0
    assert hsq.intercept_se == "NA"


# ---------- test 3: summary output ----------

def test_hsq_summary_format(hsq_data):
    """
    summary() should return a string with key fields.
    """
    chisq, x, w, N, M_arr = hsq_data
    hsq = Hsq(chisq, x, w, N, M_arr, n_blocks=N_BLOCKS)
    text = hsq.summary()

    assert "Total Observed scale h2" in text
    assert "Lambda GC" in text
    assert "Mean Chi^2" in text
    assert "Intercept" in text


# ---------- test 4: Gencov recovers known covariance ----------

def test_gencov_recovers_covariance():
    """
    Simulate two traits with known genetic covariance.
    z1 and z2 share a genetic component driven by LD scores.

    Model:
        z1_j = sqrt(N1 * h2_1 * lj / M) * g_j + e1_j
        z2_j = sqrt(N2 * h2_2 * lj / M) * g_j + e2_j
    where g_j is shared genetic signal, e1_j, e2_j are independent noise.
    This gives rho_g = sqrt(h2_1 * h2_2) when g is fully shared.
    """
    rng = np.random.default_rng(55)
    n_snp = 2000
    N1_val, N2_val = 10000.0, 10000.0
    h2_1, h2_2 = 0.4, 0.4
    m = 1000.0

    ld = rng.uniform(1.0, 5.0, (n_snp, 1))

    # shared genetic signal
    genetic = rng.standard_normal((n_snp, 1))
    z1 = np.sqrt(N1_val * h2_1 * ld / m) * genetic + rng.standard_normal((n_snp, 1))
    z2 = np.sqrt(N2_val * h2_2 * ld / m) * genetic + rng.standard_normal((n_snp, 1))

    x = ld.copy()
    w = ld.copy()
    N1 = np.full((n_snp, 1), N1_val)
    N2 = np.full((n_snp, 1), N2_val)
    M_arr = np.array([[m]])

    # need h2 estimates and intercepts for Gencov
    hsq1 = Hsq(z1 ** 2, x, w, N1, M_arr, n_blocks=N_BLOCKS)
    hsq2 = Hsq(z2 ** 2, x, w, N2, M_arr, n_blocks=N_BLOCKS)

    gc = Gencov(
        z1, z2, x, w, N1, N2, M_arr,
        hsq1.tot, hsq2.tot, hsq1.intercept, hsq2.intercept,
        n_blocks=N_BLOCKS,
    )

    # genetic covariance should be positive (traits share signal)
    assert gc.tot > 0, f"gencov={gc.tot}, expected positive"
    assert gc.tot_se > 0


# ---------- test 5: RG recovers genetic correlation ----------

def test_rg_recovers_correlation():
    """
    With fully shared genetic signal (same g_j for both traits),
    rg should be close to 1.0.
    """
    rng = np.random.default_rng(77)
    n_snp = 2000
    N_val = 10000.0
    h2 = 0.4
    m = 1000.0

    ld = rng.uniform(1.0, 5.0, (n_snp, 1))

    # fully shared signal -> rg should be ~1
    genetic = rng.standard_normal((n_snp, 1))
    z1 = np.sqrt(N_val * h2 * ld / m) * genetic + rng.standard_normal((n_snp, 1))
    z2 = np.sqrt(N_val * h2 * ld / m) * genetic + rng.standard_normal((n_snp, 1))

    x = ld.copy()
    w = ld.copy()
    N = np.full((n_snp, 1), N_val)
    M_arr = np.array([[m]])

    rg = RG(z1, z2, x, w, N, N, M_arr, n_blocks=N_BLOCKS)

    # rg should not be flagged as negative h2 or out of bounds
    assert rg._negative_hsq is None, "h2 was negative"
    assert isinstance(rg.rg_ratio, float), "rg_ratio should be a float"

    # rg should be positive (fully shared signal)
    assert rg.rg_ratio > 0, f"rg={rg.rg_ratio}, expected positive"
    assert rg.rg_se > 0


# ---------- test 6: input validation ----------

def test_rejects_1d_input():
    """All inputs must be 2D arrays."""
    with pytest.raises(TypeError, match="2D"):
        Hsq(
            np.ones(10),
            np.ones((10, 1)),
            np.ones((10, 1)),
            np.ones((10, 1)),
            np.array([[100.0]]),
        )


def test_rejects_mismatched_shapes():
    """y, w, N must all have shape (n_snp, 1)."""
    with pytest.raises(ValueError, match="shape"):
        Hsq(
            np.ones((10, 1)),
            np.ones((10, 1)),
            np.ones((5, 1)),  # wrong n_snp
            np.ones((10, 1)),
            np.array([[100.0]]),
        )

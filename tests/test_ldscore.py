import numpy as np
import jax.numpy as jnp
from ldsc_jx.parse import read_genotypes
from ldsc_jx.ldscore import get_block_lefts, standardize, compute_ld_scores

DATA = "tests/data/g1k_chr22_test.bed"


def test_block_lefts_shape():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    block_left, block_right = get_block_lefts(snp_pos, 1000000)
    assert len(block_left) == X.shape[1]
    assert len(block_right) == X.shape[1]


def test_block_lefts_order():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    block_left, block_right = get_block_lefts(snp_pos, 1000000)
    for i in range(len(block_left)):
        assert block_left[i] <= i
        assert block_right[i] >= i + 1


def test_standardize_mean_zero():
    X, _, _, _, _ = read_genotypes(DATA)
    X_std = standardize(jnp.array(X))
    means = jnp.mean(X_std, axis=0)
    assert jnp.allclose(means, 0, atol=1e-5)


def test_standardize_unit_variance():
    X, _, _, _, _ = read_genotypes(DATA)
    X_std = standardize(jnp.array(X))
    stds = jnp.std(X_std, axis=0)
    polymorphic = stds > 0
    assert jnp.allclose(stds[polymorphic], 1, atol=1e-5)


def test_ld_scores_shape():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    block_left, block_right = get_block_lefts(snp_pos, 1000000)
    X_std = standardize(jnp.array(X))
    ld_scores = compute_ld_scores(X_std, block_left, block_right)
    assert ld_scores.shape == (X.shape[1],)


def test_ld_scores_positive():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    block_left, block_right = get_block_lefts(snp_pos, 1000000)
    X_std = standardize(jnp.array(X))
    ld_scores = compute_ld_scores(X_std, block_left, block_right)
    polymorphic = np.std(np.array(X), axis=0) > 0
    assert np.all(ld_scores[polymorphic] > 0)


def test_ld_scores_self_contributes_one():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    block_left, block_right = get_block_lefts(snp_pos, 1000000)
    X_std = standardize(jnp.array(X))
    ld_scores = compute_ld_scores(X_std, block_left, block_right)
    n = X.shape[0]
    adj_one = 1.0 - (1.0 - 1.0) / (n - 2)
    polymorphic = np.std(np.array(X), axis=0) > 0
    assert np.all(ld_scores[polymorphic] >= adj_one - 0.05)


def test_ld_scores_match_bulik():
    X, snp_pos, _, _, _ = read_genotypes(DATA)
    kept = np.loadtxt("tests/data/bulik_kept_snps_chr22.txt", dtype=int)
    bulik_scores = np.loadtxt("tests/data/bulik_ld_scores_chr22.txt").flatten()

    X_filt = X[:, kept]
    pos_filt = snp_pos[kept]

    # use full window to match bulik, which does not enforce per-SNP
    # window boundaries within chunks on sparse data
    block_left, block_right = get_block_lefts(pos_filt, 50000000)
    X_std = standardize(jnp.array(X_filt))
    scores = compute_ld_scores(X_std, block_left, block_right)

    assert np.allclose(scores, bulik_scores, atol=1e-3)

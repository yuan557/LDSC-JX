import numpy as np
import jax
import jax.numpy as jnp

# function to get left and right index for each SNV within defined window (default 1000kb)
def get_block_lefts(pos, max_dist):
    block_left = np.searchsorted(pos, pos - max_dist)
    block_right = np.searchsorted(pos, pos + max_dist, side="right")
    return block_left, block_right

# function to standardize allele dosage 
@jax.jit
def standardize(X):
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    std = jnp.where(std == 0, 1, std)
    return (X - mean) / std

# function to get LD matrix (r2, with unbiased adjustment for sample size)
@jax.jit
def _compute_r2_adj(X_window, X_chunk, n):
    R = (1 / n) * (X_window.T @ X_chunk)
    r_sq = R ** 2
    denom = jnp.where(n > 2, n - 2, n)
    return r_sq - (1 - r_sq) / denom

# function to compute LD score, batched by block of SNVs (50 by default), summing adj R2 across all SNVs within window for each SNV
def compute_ld_scores(X_std, block_left, block_right, chunk_size=50):
    n, m = X_std.shape
    ld_scores = np.zeros(m)

    # calculate LD scores 50 SNVs each
    for chunk_start in range(0, m, chunk_size):
        chunk_end = min(chunk_start + chunk_size, m)
        win_left = int(block_left[chunk_start])
        win_right = int(block_right[chunk_end - 1])

        X_window = X_std[:, win_left:win_right]
        X_chunk = X_std[:, chunk_start:chunk_end]

        R_sq = _compute_r2_adj(X_window, X_chunk, n)

        # sum adj r2 within each SNV's individual window
        for i in range(chunk_start, chunk_end):
            snp_left = int(block_left[i])
            snp_right = int(block_right[i])
            col = i - chunk_start
            row_start = snp_left - win_left
            row_end = snp_right - win_left
            ld_scores[i] = float(jnp.sum(R_sq[row_start:row_end, col]))

    return ld_scores

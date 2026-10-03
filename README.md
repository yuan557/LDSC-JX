# LDSC-JX

GPU-accelerated LD Score Regression (LDSC) in Python 3, powered by JAX.

`ldsc-jx` is a reimplementation of LDSC ([Bulik-Sullivan et al. *Nature Genetics*, 2015](https://www.nature.com/articles/ng.3211); [repo](https://github.com/bulik/ldsc)) that offloads the compute-heavy LD score calculation to GPU via JAX. It is designed to accelerate windowed LD computation on GPUs while maintaining numerical compatibility with the reference LDSC implementation.

## Why ldsc-jx

The original `bulik/ldsc` is written in Python 2 and runs on CPU with NumPy. `ldsc-jx` addresses this by:

- **GPU acceleration** via JAX: the GEMM-heavy LD score calculation runs on GPU through XLA
- **Python 3**: no Python 2 dependency
- **pip-installable**: a self-contained package with no manual setup

The core algorithm is the same. LD scores computed by `ldsc-jx` are numerically concordant with the original (validated against `bulik/ldsc` on 1000 Genomes Phase 3 data).

## Quickstart

```bash
pip install git+https://github.com/yuan557/LDSC-JX.git
```

### CLI

```bash
ldsc-jx --bfile data/chr22 --l2 --ld-wind-kb 1000 --out output/chr22
```

This outputs `chr22.l2.ldscore.gz`, `chr22.l2.M`, and `chr22.l2.M_5_50`, matching the format of bulik/ldsc.

### Python API

```python
from ldsc_jx.parse import read_genotypes
from ldsc_jx.ldscore import get_block_lefts, standardize, compute_ld_scores
import jax.numpy as jnp

# read genotypes from PLINK .bed file
X, snp_pos, snp_cm, chrom, snp_id = read_genotypes("data/chr22.bed")

# compute LD scores with a 1000 kb window
block_left, block_right = get_block_lefts(snp_pos, 1000000)
X_std = standardize(jnp.array(X))
ld_scores = compute_ld_scores(X_std, block_left, block_right)
```

## Computing LD Scores

Given a genotype matrix $X \in \mathbb{R}^{n \times m}$ where $n$ is individuals and $m$ is SNVs, the LD score for SNV $j$ is the sum of squared correlations within a window:

$$\ell_j = \sum_{k \in W(j)} r_{jk}^2$$

Each column of $X$ is standardized to zero mean and unit variance. For a chunk of SNVs, the correlation matrix is computed as:

```
R = (1/n) * X_chunk.T @ X_window
```

This is a general matrix-matrix multiplication (GEMM), where `matmul` in JAX replaces millions of individual dot products. The LD scores are then:

```
l2 = sum(R ** 2, axis=1)
```

In the original `bulik/ldsc`, normalized genotype blocks are multiplied as `np.dot(A.T, B / n)`, followed by squaring and summation to obtain LD Scores. In this notation, `A` corresponds to `X_chunk` and `B` corresponds to `X_window`, the subsets of standardized genotype columns for the target chunk and its LD window, respectively. The reference implementation performs these operations on the CPU using NumPy. `ldsc-jx` preserves this computational structure while replacing NumPy with JAX, allowing the same GEMM-heavy calculation to target GPUs. JAX's `jit` compiler then optimizes the surrounding array operations (matrix multiplication, squaring, and reductions) together via XLA.

## Bias Correction

As per [Bulik-Sullivan et al. *Nature Genetics*, 2015](https://www.nature.com/articles/ng.3211), the sample $r^2$ is biased upward by approximately $1/n$. The unbiased estimator used in the original `bulik/ldsc` applies the following adjustment:

$$r^2_{\text{adj}} = r^2 - \frac{1 - r^2}{n - 2}$$

which is algebraically equivalent to:

$$r^2_{\text{adj}} = \frac{(n - 1) r^2 - 1}{n - 2}$$

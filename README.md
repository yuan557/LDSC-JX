# LDSC-JX

GPU-accelerated LD Score Regression (LDSC) in Python 3, powered by JAX.

`ldsc-jx` is a reimplementation of LDSC ([Bulik-Sullivan et al. *Nature Genetics*, 2015](https://www.nature.com/articles/ng.3211); [repo](https://github.com/bulik/ldsc)) that offloads the compute-heavy LD score calculation to GPU via JAX. It is designed to accelerate windowed LD computation on GPUs while maintaining numerical compatibility with the reference LDSC implementation.

## LDSC Overview

LD Score Regression uses linkage disequilibrium (LD) information from a reference genotype dataset to estimate the correlation between the strength of association of genetic variants and how much they are inherited together with other surrounding genetic variants. The slope of the regression is proportional to heritability estimated by the GWAS study, and its intercept minus 1 measures confounding bias that inflates GWAS summary statistics.

Given a genotype matrix **X** of shape `(n, m)` where `n` is individuals and `m` is SNVs, for each SNV *j*, its LD score is defined as the sum of *r*<sup>2</sup> of all other genetic variants within a selected window.

$$\ell_j = \sum_{k \in W(j)} r_{jk}^2$$

where:

- $r_{jk}^2$ is the squared correlation between SNV *j* and SNV *k*
- $W(j)$ is the set of SNVs within the selected LD window, typically defined in units of cM or kb

The LD Score therefore measures the amount of local LD surrounding a given SNV: SNVs that are correlated with many nearby variants will have larger LD Scores.

`ldsc-jx` aims to accelerate pairwise correlation calculation as matrix multiplication, which is optimized on GPUs.

## Computing LD Scores

Let the genotype matrix be $X \in \mathbb{R}^{n \times m}$, where $n$ is the number of individuals and $m$ is the number of SNVs. Each column is standardized to zero mean and unit variance. For a chunk of SNVs, the correlation matrix is computed as:

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
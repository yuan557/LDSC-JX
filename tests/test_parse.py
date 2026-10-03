import numpy as np
from ldsc_jx.parse import read_genotypes

DATA = "tests/data/g1k_chr22_test.bed"


def test_shape():
    X, snp_pos, snp_cm, chrom = read_genotypes(DATA)
    assert X.shape == (2504, 1000)


def test_no_nans():
    X, _, _, _ = read_genotypes(DATA)
    assert not np.any(np.isnan(X))


def test_dtype():
    X, _, _, _ = read_genotypes(DATA)
    assert X.dtype == np.float32


def test_metadata_lengths():
    X, snp_pos, snp_cm, chrom = read_genotypes(DATA)
    assert len(snp_pos) == X.shape[1]
    assert len(snp_cm) == X.shape[1]
    assert len(chrom) == X.shape[1]


def test_chromosome():
    _, _, _, chrom = read_genotypes(DATA)
    assert all(c == "22" for c in chrom)

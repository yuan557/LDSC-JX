import numpy as np
from bed_reader import open_bed

def read_genotypes(file_prefix):
    """read plink2 .pgen .pvar .psam files"""
    with open_bed(file_prefix) as bed:
        X = bed.read(dtype = "float32")
        snp_pos = bed.bp_position
        snp_cm = bed.cm_position
        chrom = bed.chromosome
    # mean impute missing values that are coded as NaN
    col_means = np.nanmean(X, axis = 0)
    mask = np.isnan(X)
    X_imputed = np.where(mask, col_means, X)
    return X_imputed, snp_pos, snp_cm, chrom

import argparse
import gzip
import os
import sys

import numpy as np
import jax.numpy as jnp

from ldsc_jx.parse import read_genotypes
from ldsc_jx.ldscore import get_block_lefts, standardize, compute_ld_scores


def main():
    parser = argparse.ArgumentParser(
        description="ldsc-jx: GPU-accelerated LD Score computation"
    )
    parser.add_argument("--bfile", required=True, type=str,
        help="Prefix for PLINK .bed/.bim/.fam files")
    parser.add_argument("--l2", action="store_true", default=False,
        help="Estimate LD Scores (l2)")
    parser.add_argument("--ld-wind-kb", default=1000, type=float,
        help="Window size in kilobases (default: 1000)")
    parser.add_argument("--chunk-size", default=50, type=int,
        help="Number of SNPs per chunk (default: 50)")
    parser.add_argument("--out", required=True, type=str,
        help="Output file prefix")

    args = parser.parse_args()

    if not args.l2:
        parser.error("--l2 is required (only LD score estimation is supported)")

    bed_path = args.bfile + ".bed"
    if not os.path.exists(bed_path):
        print(f"Error: {bed_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading genotypes from {args.bfile}")
    X, snp_pos, snp_cm, chrom, snp_id = read_genotypes(bed_path)
    n, m = X.shape
    print(f"Read {n} individuals, {m} SNPs")

    max_dist = int(args.ld_wind_kb * 1000)
    print(f"Using window size {args.ld_wind_kb} kb")

    block_left, block_right = get_block_lefts(snp_pos, max_dist)
    X_std = standardize(jnp.array(X))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Computing LD Scores (chunk size {args.chunk_size})")
    ld_scores = compute_ld_scores(X_std, block_left, block_right,
                                  chunk_size=args.chunk_size)

    # write .l2.ldscore, then gzip
    out_ldscore = args.out + ".l2.ldscore"
    with open(out_ldscore, "w") as f:
        f.write("CHR\tSNP\tBP\tL2\n")
        for i in range(m):
            f.write(f"{chrom[i]}\t{snp_id[i]}\t{snp_pos[i]}\t{ld_scores[i]:.3f}\n")

    with open(out_ldscore, "rb") as f_in:
        with gzip.open(out_ldscore + ".gz", "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(out_ldscore)
    print(f"Wrote LD Scores for {m} SNPs to {out_ldscore}.gz")

    # write .l2.M (total SNP count)
    with open(args.out + ".l2.M", "w") as f:
        f.write(f"{m}\n")

    # write .l2.M_5_50 (SNPs with MAF > 0.05)
    allele_freq = np.mean(X, axis=0) / 2
    maf = np.minimum(allele_freq, 1 - allele_freq)
    m_5_50 = int(np.sum(maf > 0.05))
    with open(args.out + ".l2.M_5_50", "w") as f:
        f.write(f"{m_5_50}\n")

    print(f"Total SNPs: {m}")
    print(f"SNPs with MAF > 0.05: {m_5_50}")
    print("Done")


if __name__ == "__main__":
    main()

import numpy as np
import jax.numpy as jnp
from scipy.stats import norm, chi2
from ldsc_jx.irwls import IRWLS
from ldsc_jx import jackknife as jk


def append_intercept(x):
    n_row = x.shape[0]
    intercept = jnp.ones((n_row, 1))
    return jnp.concatenate((x, intercept), axis=1)


def remove_intercept(x):
    return x[:, :-1]


def h2_obs_to_liab(h2_obs, P, K):
    if np.isnan(P) and np.isnan(K):
        return h2_obs
    if K <= 0 or K >= 1:
        raise ValueError("K must be in the range (0,1)")
    if P <= 0 or P >= 1:
        raise ValueError("P must be in the range (0,1)")
    thresh = norm.isf(K)
    conversion_factor = (
        K ** 2 * (1 - K) ** 2 / (P * (1 - P) * norm.pdf(thresh) ** 2)
    )
    return h2_obs * conversion_factor


def p_z_norm(est, se):
    try:
        Z = est / se
    except (FloatingPointError, ZeroDivisionError):
        Z = float("inf")
    P = chi2.sf(Z ** 2, 1, loc=0, scale=1)
    return P, Z


def update_separators(s, ii):
    maplist = np.arange(len(ii))[np.squeeze(ii)]
    mask_to_unmask = lambda i: maplist[i]
    t = np.apply_along_axis(mask_to_unmask, 0, s[1:-1])
    t = np.hstack(((0), t, (len(ii))))
    return t


class LD_Score_Regression:

    def __init__(self, y, x, w, N, M, n_blocks, intercept=None,
                 slow=False, step1_ii=None, old_weights=False):
        for i in [y, x, w, M, N]:
            if len(i.shape) != 2:
                raise TypeError("Arguments must be 2D arrays.")

        n_snp, self.n_annot = x.shape
        if any(i.shape != (n_snp, 1) for i in [y, w, N]):
            raise ValueError(
                "N, weights and response must have shape (n_snp, 1)."
            )
        if M.shape != (1, self.n_annot):
            raise ValueError("M must have shape (1, n_annot).")

        M_tot = float(jnp.sum(M))
        x_tot = jnp.sum(x, axis=1).reshape((n_snp, 1))
        self.constrain_intercept = intercept is not None
        self.intercept = intercept
        self.n_blocks = n_blocks
        tot_agg = self.aggregate(y, x_tot, N, M_tot, intercept)
        initial_w = self._update_weights(
            x_tot, w, N, M_tot, tot_agg, intercept
        )
        Nbar = jnp.mean(N)
        x = jnp.multiply(N, x) / Nbar
        if not self.constrain_intercept:
            x, x_tot = append_intercept(x), append_intercept(x_tot)
            yp = y
        else:
            yp = y - intercept
            self.intercept_se = "NA"
        del y

        self.twostep_filtered = None
        if step1_ii is not None and self.constrain_intercept:
            raise ValueError(
                "twostep is not compatible with constrain_intercept."
            )
        elif step1_ii is not None and self.n_annot > 1:
            raise ValueError(
                "twostep not compatible with partitioned LD Score yet."
            )
        elif step1_ii is not None:
            n1 = int(jnp.sum(step1_ii))
            self.twostep_filtered = n_snp - n1
            x1 = x[jnp.squeeze(step1_ii), :]
            yp1, w1, N1, initial_w1 = (
                a[step1_ii].reshape((n1, 1))
                for a in (yp, w, N, initial_w)
            )
            update_func1 = lambda a: self._update_func(
                a, x1, w1, N1, M_tot, Nbar, ii=step1_ii
            )
            step1_jknife = IRWLS(
                np.array(x1), np.array(yp1), update_func1, n_blocks,
                w=np.array(initial_w1)
            )
            step1_int, _ = self._intercept(step1_jknife)
            yp = yp - step1_int
            x = remove_intercept(x)
            x_tot = remove_intercept(x_tot)
            update_func2 = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, step1_int
            )
            s = update_separators(step1_jknife.separators, step1_ii)
            step2_jknife = IRWLS(
                np.array(x), np.array(yp), update_func2, n_blocks,
                w=np.array(initial_w), separators=s
            )
            c = float(
                jnp.sum(jnp.multiply(initial_w, x))
                / jnp.sum(jnp.multiply(initial_w, jnp.square(x)))
            )
            jknife = self._combine_twostep_jknives(
                step1_jknife, step2_jknife, M_tot, c, Nbar
            )
        elif old_weights:
            initial_w = jnp.sqrt(initial_w)
            x = IRWLS._weight(x, initial_w)
            yp = IRWLS._weight(yp, initial_w)
            jknife = jk.LstsqJackknifeFast(
                np.array(x), np.array(yp), n_blocks
            )
        else:
            update_func = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, Nbar, intercept
            )
            jknife = IRWLS(
                np.array(x), np.array(yp), update_func, n_blocks,
                w=np.array(initial_w)
            )

        self.coef, self.coef_cov, self.coef_se = self._coef(jknife, Nbar)
        self.cat, self.cat_cov, self.cat_se = self._cat(
            jknife, M, Nbar, self.coef, self.coef_cov
        )
        self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
        self.prop, self.prop_cov, self.prop_se = self._prop(
            jknife, M, Nbar, self.cat, self.tot
        )
        self.enrichment, self.M_prop = self._enrichment(
            M, M_tot, self.cat, self.tot
        )
        if not self.constrain_intercept:
            self.intercept, self.intercept_se = self._intercept(jknife)

        self.jknife = jknife
        self.tot_delete_values = self._delete_vals_tot(jknife, Nbar, M)
        self.part_delete_values = self._delete_vals_part(jknife, Nbar, M)
        if not self.constrain_intercept:
            self.intercept_delete_values = jknife.delete_values[
                :, self.n_annot
            ]
        self.M = M

    @classmethod
    def aggregate(cls, y, x, N, M, intercept=None):
        if intercept is None:
            intercept = cls.__null_intercept__
        num = M * (jnp.mean(y) - intercept)
        denom = jnp.mean(jnp.multiply(x, N))
        return float(num / denom)

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar,
                     intercept=None, ii=None):
        raise NotImplementedError

    def _delete_vals_tot(self, jknife, Nbar, M):
        n_annot = self.n_annot
        tot_delete_vals = jknife.delete_values[:, 0:n_annot]
        return np.dot(tot_delete_vals, np.array(M).T) / float(Nbar)

    def _delete_vals_part(self, jknife, Nbar, M):
        n_annot = self.n_annot
        return jknife.delete_values[:, 0:n_annot] / float(Nbar)

    def _coef(self, jknife, Nbar):
        n_annot = self.n_annot
        Nbar = float(Nbar)
        coef = jknife.est[0, 0:n_annot] / Nbar
        coef_cov = jknife.jknife_cov[0:n_annot, 0:n_annot] / Nbar ** 2
        coef_se = np.sqrt(np.diag(coef_cov))
        return coef, coef_cov, coef_se

    def _cat(self, jknife, M, Nbar, coef, coef_cov):
        M = np.array(M)
        cat = np.multiply(M, coef)
        cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
        cat_se = np.sqrt(np.diag(cat_cov))
        return cat, cat_cov, cat_se

    def _tot(self, cat, cat_cov):
        tot = float(np.sum(cat))
        tot_cov = float(np.sum(cat_cov))
        tot_se = float(np.sqrt(tot_cov))
        return tot, tot_cov, tot_se

    def _prop(self, jknife, M, Nbar, cat, tot):
        M = np.array(M)
        Nbar = float(Nbar)
        n_annot = self.n_annot
        n_blocks = jknife.delete_values.shape[0]
        numer_delete_vals = (
            np.multiply(M, jknife.delete_values[:, 0:n_annot]) / Nbar
        )
        denom_delete_vals = np.sum(
            numer_delete_vals, axis=1
        ).reshape((n_blocks, 1))
        denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, n_annot)))
        prop = jk.RatioJackknife(
            np.atleast_2d(cat / tot), numer_delete_vals, denom_delete_vals
        )
        return prop.est, prop.jknife_cov, prop.jknife_se

    def _enrichment(self, M, M_tot, cat, tot):
        M = np.array(M)
        M_prop = M / M_tot
        enrichment = np.divide(cat, M) / (tot / M_tot)
        return enrichment, M_prop

    def _intercept(self, jknife):
        n_annot = self.n_annot
        intercept = float(jknife.est[0, n_annot])
        intercept_se = float(jknife.jknife_se[0, n_annot])
        return intercept, intercept_se

    def _combine_twostep_jknives(self, step1_jknife, step2_jknife,
                                  M_tot, c, Nbar=1):
        from collections import namedtuple
        n_blocks, n_annot = step1_jknife.delete_values.shape
        n_annot -= 1
        if n_annot > 2:
            raise ValueError(
                "twostep not yet implemented for partitioned LD Score."
            )

        step1_int, _ = self._intercept(step1_jknife)
        est = np.hstack(
            (step2_jknife.est, np.array(step1_int).reshape((1, 1)))
        )
        delete_values = np.zeros((n_blocks, n_annot + 1))
        delete_values[:, n_annot] = step1_jknife.delete_values[:, n_annot]
        delete_values[:, 0:n_annot] = (
            step2_jknife.delete_values
            - c * (
                step1_jknife.delete_values[:, n_annot] - step1_int
            ).reshape((n_blocks, n_annot))
        )
        pseudovalues = jk.delete_values_to_pseudovalues(delete_values, est)
        jknife_est, jknife_var, jknife_se, jknife_cov = jk.jknife_stats(
            pseudovalues
        )
        Result = namedtuple(
            "Result",
            ["est", "jknife_se", "jknife_est", "jknife_var",
             "jknife_cov", "delete_values", "separators"],
        )
        return Result(
            est, jknife_se, jknife_est, jknife_var, jknife_cov,
            delete_values, step1_jknife.separators,
        )


class Hsq(LD_Score_Regression):

    __null_intercept__ = 1

    def __init__(self, y, x, w, N, M, n_blocks=200, intercept=None,
                 slow=False, twostep=None, old_weights=False):
        step1_ii = None
        if twostep is not None:
            step1_ii = y < twostep

        LD_Score_Regression.__init__(
            self, y, x, w, N, M, n_blocks, intercept=intercept,
            slow=slow, step1_ii=step1_ii, old_weights=old_weights,
        )
        self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
        if not self.constrain_intercept:
            self.ratio, self.ratio_se = self._ratio(
                self.intercept, self.intercept_se, self.mean_chisq
            )

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar,
                     intercept=None, ii=None):
        hsq = M * x[0][0] / float(Nbar)
        if intercept is None:
            intercept = float(jnp.max(x[0][1]))
        else:
            if ref_ld_tot.shape[1] > 1:
                raise ValueError(
                    "Design matrix has intercept column for "
                    "constrained intercept regression!"
                )
        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)
        w = self.weights(ld, w_ld, N, M, hsq, intercept, ii)
        return w

    def _summarize_chisq(self, chisq):
        mean_chisq = float(jnp.mean(chisq))
        lambda_gc = float(jnp.median(chisq)) / 0.4549
        return mean_chisq, lambda_gc

    def _ratio(self, intercept, intercept_se, mean_chisq):
        if mean_chisq > 1:
            ratio_se = intercept_se / (mean_chisq - 1)
            ratio = (intercept - 1) / (mean_chisq - 1)
        else:
            ratio = "NA"
            ratio_se = "NA"
        return ratio, ratio_se

    def summary(self, ref_ld_colnames=None, P=None, K=None):
        if P is not None and K is not None:
            T = "Liability"
            c = h2_obs_to_liab(1, P, K)
        else:
            T = "Observed"
            c = 1

        out = [
            f"Total {T} scale h2: {c * self.tot:.4f} ({c * self.tot_se:.4f})"
        ]
        out.append(f"Lambda GC: {self.lambda_gc:.4f}")
        out.append(f"Mean Chi^2: {self.mean_chisq:.4f}")
        if self.constrain_intercept:
            out.append(f"Intercept: constrained to {self.intercept}")
        else:
            out.append(
                f"Intercept: {self.intercept:.4f} ({self.intercept_se:.4f})"
            )
            if self.mean_chisq > 1:
                if self.ratio < 0:
                    out.append(
                        "Ratio < 0 (usually indicates GC correction)."
                    )
                else:
                    out.append(
                        f"Ratio: {self.ratio:.4f} ({self.ratio_se:.4f})"
                    )
            else:
                out.append("Ratio: NA (mean chi^2 < 1)")

        return "\n".join(out)

    def _update_weights(self, ld, w_ld, N, M, hsq, intercept, ii=None):
        if intercept is None:
            intercept = self.__null_intercept__
        return self.weights(ld, w_ld, N, M, hsq, intercept, ii)

    @classmethod
    def weights(cls, ld, w_ld, N, M, hsq, intercept=None, ii=None):
        M = float(M)
        if intercept is None:
            intercept = 1
        hsq = max(hsq, 0.0)
        hsq = min(hsq, 1.0)
        ld = jnp.fmax(ld, 1.0)
        w_ld = jnp.fmax(w_ld, 1.0)
        c = hsq * N / M
        het_w = 1.0 / (2 * jnp.square(intercept + jnp.multiply(c, ld)))
        oc_w = 1.0 / w_ld
        w = jnp.multiply(het_w, oc_w)
        return w


class Gencov(LD_Score_Regression):

    __null_intercept__ = 0

    def __init__(self, z1, z2, x, w, N1, N2, M, hsq1, hsq2,
                 intercept_hsq1, intercept_hsq2, n_blocks=200,
                 intercept_gencov=None, slow=False, twostep=None):
        self.intercept_hsq1 = intercept_hsq1
        self.intercept_hsq2 = intercept_hsq2
        self.hsq1 = hsq1
        self.hsq2 = hsq2
        self.N1 = N1
        self.N2 = N2
        y = z1 * z2
        step1_ii = None
        if twostep is not None:
            step1_ii = jnp.logical_and(z1 ** 2 < twostep, z2 ** 2 < twostep)

        LD_Score_Regression.__init__(
            self, y, x, w, jnp.sqrt(N1 * N2), M, n_blocks,
            intercept=intercept_gencov, slow=slow, step1_ii=step1_ii,
        )
        self.p, self.z = p_z_norm(self.tot, self.tot_se)
        self.mean_z1z2 = float(jnp.mean(jnp.multiply(z1, z2)))

    def summary(self, ref_ld_colnames, P=None, K=None):
        out = []
        if (
            P is not None
            and K is not None
            and all(i is not None for i in P)
            and all(i is not None for i in K)
        ):
            T = "Liability"
            c = h2_obs_to_liab(1, P[0], K[0]) * h2_obs_to_liab(1, P[1], K[1])
            c = np.sqrt(c)
        else:
            T = "Observed"
            c = 1

        out.append(
            f"Total {T} scale gencov: {c * self.tot:.4f} "
            f"({c * self.tot_se:.4f})"
        )
        out.append(f"Mean z1*z2: {self.mean_z1z2:.4f}")
        if self.constrain_intercept:
            out.append(f"Intercept: constrained to {self.intercept}")
        else:
            out.append(
                f"Intercept: {self.intercept:.4f} ({self.intercept_se:.4f})"
            )
        return "\n".join(out)

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar,
                     intercept=None, ii=None):
        rho_g = M * x[0][0] / float(Nbar)
        if intercept is None:
            intercept = x[0][1]

        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)
        if ii is not None:
            N1 = self.N1[ii].reshape(w_ld.shape)
            N2 = self.N2[ii].reshape(w_ld.shape)
        else:
            N1 = self.N1
            N2 = self.N2

        return self.weights(
            ld, w_ld, N1, N2, jnp.sum(M), self.hsq1, self.hsq2,
            rho_g, intercept, self.intercept_hsq1, self.intercept_hsq2, ii,
        )

    def _update_weights(self, ld, w_ld, sqrt_n1n2, M, rho_g,
                        intercept, ii=None):
        return self.weights(
            ld, w_ld, self.N1, self.N2, M, self.hsq1, self.hsq2,
            rho_g, intercept, self.intercept_hsq1, self.intercept_hsq2,
        )

    @classmethod
    def weights(cls, ld, w_ld, N1, N2, M, h1, h2, rho_g,
                intercept_gencov=None, intercept_hsq1=None,
                intercept_hsq2=None, ii=None):
        M = float(M)
        if intercept_gencov is None:
            intercept_gencov = 0
        if intercept_hsq1 is None:
            intercept_hsq1 = 1
        if intercept_hsq2 is None:
            intercept_hsq2 = 1

        h1, h2 = max(h1, 0.0), max(h2, 0.0)
        h1, h2 = min(h1, 1.0), min(h2, 1.0)
        rho_g = max(min(rho_g, 1.0), -1.0)
        ld = jnp.fmax(ld, 1.0)
        w_ld = jnp.fmax(w_ld, 1.0)
        a = jnp.multiply(N1, h1 * ld) / M + intercept_hsq1
        b = jnp.multiply(N2, h2 * ld) / M + intercept_hsq2
        sqrt_n1n2 = jnp.sqrt(jnp.multiply(N1, N2))
        c = jnp.multiply(sqrt_n1n2, rho_g * ld) / M + intercept_gencov
        het_w = 1.0 / (jnp.multiply(a, b) + jnp.square(c))
        oc_w = 1.0 / w_ld
        w = jnp.multiply(het_w, oc_w)
        return w


class RG:

    def __init__(self, z1, z2, x, w, N1, N2, M, intercept_hsq1=None,
                 intercept_hsq2=None, intercept_gencov=None, n_blocks=200,
                 slow=False, twostep=None):
        self.intercept_gencov = intercept_gencov
        self._negative_hsq = None

        hsq1 = Hsq(
            jnp.square(z1), x, w, N1, M, n_blocks=n_blocks,
            intercept=intercept_hsq1, slow=slow, twostep=twostep,
        )
        hsq2 = Hsq(
            jnp.square(z2), x, w, N2, M, n_blocks=n_blocks,
            intercept=intercept_hsq2, slow=slow, twostep=twostep,
        )
        gencov = Gencov(
            z1, z2, x, w, N1, N2, M, hsq1.tot, hsq2.tot,
            hsq1.intercept, hsq2.intercept, n_blocks,
            intercept_gencov=intercept_gencov, slow=slow, twostep=twostep,
        )
        gencov.N1 = None
        gencov.N2 = None
        self.hsq1, self.hsq2, self.gencov = hsq1, hsq2, gencov

        if hsq1.tot <= 0 or hsq2.tot <= 0:
            self._negative_hsq = True
            self.rg_ratio = self.rg = self.rg_se = "NA"
            self.p = self.z = "NA"
        else:
            rg_ratio = np.array(
                gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)
            ).reshape((1, 1))
            denom_delete_values = np.sqrt(
                np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values)
            )
            rg = jk.RatioJackknife(
                rg_ratio, gencov.tot_delete_values, denom_delete_values
            )
            self.rg_jknife = rg.jknife_est.item()
            self.rg_se = rg.jknife_se.item()
            self.rg_ratio = rg_ratio.item()
            self.p, self.z = p_z_norm(self.rg_ratio, self.rg_se)

    def summary(self):
        out = []
        if self._negative_hsq:
            out.append("Genetic Correlation: nan (h2 out of bounds)")
            out.append("WARNING: One of the h2's was out of bounds.")
        elif abs(self.rg_ratio) > 1.2:
            out.append("Genetic Correlation: nan (rg out of bounds)")
            out.append("WARNING: rg was out of bounds.")
        else:
            out.append(
                f"Genetic Correlation: {self.rg_ratio:.4f} ({self.rg_se:.4f})"
            )
            out.append(f"Z-score: {self.z:.4f}")
            out.append(f"P: {self.p:.4g}")
        return "\n".join(out)

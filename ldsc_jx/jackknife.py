import numpy as np
import jax
import jax.numpy as jnp


def _check_shape(x, y):
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Number of datapoints in x != number of datapoints in y."
        )
    if y.shape[1] != 1:
        raise ValueError("y must have shape (n_snp, 1)")
    n, p = x.shape
    if p > n:
        raise ValueError("More dimensions than datapoints.")
    return (n, p)


def _check_shape_block(xty_block_values, xtx_block_values):
    if xtx_block_values.shape[0:2] != xty_block_values.shape:
        raise ValueError(
            "Shape of xty_block_values must equal shape of first two "
            "dimensions of xtx_block_values."
        )
    if len(xtx_block_values.shape) < 3:
        raise ValueError("xtx_block_values must be a 3D array.")
    if xtx_block_values.shape[1] != xtx_block_values.shape[2]:
        raise ValueError(
            "Last two axes of xtx_block_values must have same dimension."
        )
    return xtx_block_values.shape[0:2]


def get_separators(N, n_blocks):
    return np.floor(np.linspace(0, N, n_blocks + 1)).astype(int)


def delete_values_to_pseudovalues(delete_values, est):
    n_blocks, p = delete_values.shape
    if est.shape != (1, p):
        raise ValueError(
            "Different number of parameters in delete_values than in est."
        )
    return n_blocks * est - (n_blocks - 1) * delete_values


def jknife_stats(pseudovalues):
    n_blocks = pseudovalues.shape[0]
    jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / n_blocks)
    jknife_var = np.atleast_2d(np.diag(jknife_cov))
    jknife_se = np.atleast_2d(np.sqrt(jknife_var))
    jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))
    return jknife_est, jknife_var, jknife_se, jknife_cov


class LstsqJackknifeFast:

    def __init__(self, x, y, n_blocks=None, separators=None):
        n, p = _check_shape(x, y)

        if separators is not None:
            self.separators = sorted(separators)
            self.n_blocks = len(separators) - 1
        elif n_blocks is not None:
            self.n_blocks = n_blocks
            self.separators = get_separators(n, n_blocks)
        else:
            raise ValueError("Must specify either n_blocks or separators.")

        xty, xtx = self._block_values(x, y, self.separators)
        self.est = self._block_values_to_est(xty, xtx)
        self.delete_values = self._block_values_to_delete_values(xty, xtx)
        self.pseudovalues = delete_values_to_pseudovalues(
            self.delete_values, self.est
        )
        self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov = \
            jknife_stats(self.pseudovalues)

    @staticmethod
    def _block_values(x, y, s):
        n, p = _check_shape(x, y)
        n_blocks = len(s) - 1
        xtx_block_values = np.zeros((n_blocks, p, p))
        xty_block_values = np.zeros((n_blocks, p))
        for i in range(n_blocks):
            x_block = x[s[i]:s[i + 1], :]
            y_block = y[s[i]:s[i + 1], :]
            xty_block_values[i] = (x_block.T @ y_block).ravel()
            xtx_block_values[i] = x_block.T @ x_block
        return xty_block_values, xtx_block_values

    @staticmethod
    def _block_values_to_est(xty_block_values, xtx_block_values):
        _check_shape_block(xty_block_values, xtx_block_values)
        p = xty_block_values.shape[1]
        xty = np.sum(xty_block_values, axis=0)
        xtx = np.sum(xtx_block_values, axis=0)
        return np.linalg.solve(xtx, xty).reshape((1, p))

    @staticmethod
    def _block_values_to_delete_values(xty_block_values, xtx_block_values):
        _check_shape_block(xty_block_values, xtx_block_values)
        p = xty_block_values.shape[1]
        xty_tot = jnp.sum(xty_block_values, axis=0)
        xtx_tot = jnp.sum(xtx_block_values, axis=0)

        @jax.vmap
        def _solve_one(xtx_block, xty_block):
            return jnp.linalg.solve(xtx_tot - xtx_block, xty_tot - xty_block)

        delete_values = _solve_one(
            jnp.array(xtx_block_values), jnp.array(xty_block_values)
        )
        return np.array(delete_values).reshape((-1, p))


class RatioJackknife:

    def __init__(self, est, numer_delete_values, denom_delete_values):
        if numer_delete_values.shape != denom_delete_values.shape:
            raise ValueError(
                "numer_delete_values.shape != denom_delete_values.shape."
            )
        if len(numer_delete_values.shape) != 2:
            raise ValueError("Delete values must be matrices.")
        if (
            len(est.shape) != 2
            or est.shape[0] != 1
            or est.shape[1] != numer_delete_values.shape[1]
        ):
            raise ValueError(
                "Shape of est does not match shape of delete values."
            )

        self.n_blocks = numer_delete_values.shape[0]
        self.est = est
        n_blocks = self.n_blocks
        self.pseudovalues = (
            n_blocks * est
            - (n_blocks - 1) * numer_delete_values / denom_delete_values
        )
        self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov = \
            jknife_stats(self.pseudovalues)

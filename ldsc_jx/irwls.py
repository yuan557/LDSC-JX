import numpy as np
import jax.numpy as jnp
from ldsc_jx.jackknife import _check_shape, LstsqJackknifeFast


class IRWLS:

    def __init__(self, x, y, update_func, n_blocks, w=None, slow=False,
                 separators=None):
        n, p = _check_shape(x, y)
        if w is None:
            w = jnp.ones_like(y)
        if w.shape != (n, 1):
            raise ValueError(
                f"w has shape {w.shape}. w must have shape ({n}, 1)."
            )

        jknife = self.irwls(
            x, y, update_func, n_blocks, w, separators=separators
        )
        self.est = jknife.est
        self.jknife_se = jknife.jknife_se
        self.jknife_est = jknife.jknife_est
        self.jknife_var = jknife.jknife_var
        self.jknife_cov = jknife.jknife_cov
        self.delete_values = jknife.delete_values
        self.separators = jknife.separators

    @classmethod
    def irwls(cls, x, y, update_func, n_blocks, w, separators=None):
        n, p = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                f"y has shape {y.shape}. y must have shape ({n}, 1)."
            )
        if w.shape != (n, 1):
            raise ValueError(
                f"w has shape {w.shape}. w must have shape ({n}, 1)."
            )

        w = jnp.sqrt(w)
        for _ in range(2):
            new_w = jnp.sqrt(update_func(cls.wls(x, y, w)))
            if new_w.shape != w.shape:
                raise ValueError("New weights must have same shape.")
            w = new_w

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        x = np.array(x)
        y = np.array(y)
        jknife = LstsqJackknifeFast(x, y, n_blocks, separators=separators)
        return jknife

    @classmethod
    def wls(cls, x, y, w):
        n, p = x.shape
        if y.shape != (n, 1):
            raise ValueError(
                f"y has shape {y.shape}. y must have shape ({n}, 1)."
            )
        if w.shape != (n, 1):
            raise ValueError(
                f"w has shape {w.shape}. w must have shape ({n}, 1)."
            )

        x = cls._weight(x, w)
        y = cls._weight(y, w)
        coef = jnp.linalg.lstsq(x, y, rcond=None)
        return coef

    @classmethod
    def _weight(cls, x, w):
        if jnp.any(w <= 0):
            raise ValueError("Weights must be > 0")
        n, p = x.shape
        if w.shape != (n, 1):
            raise ValueError(
                f"w has shape {w.shape}. w must have shape (n, 1)."
            )

        w = w / jnp.sum(w)
        return jnp.multiply(x, w)

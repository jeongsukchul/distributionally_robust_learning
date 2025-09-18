from __future__ import annotations
from typing import Tuple, Sequence, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class BaseDistribution(nn.Module):
    """
    Base distribution for flow models (JAX/Flax version).
    """

    def forward(self, num_samples: int = 1, **kwargs):
        """Return (samples, log_prob) for `num_samples` draws."""
        raise NotImplementedError

    def log_prob(self, z: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Return log p(z) with shape equal to z's batch shape."""
        raise NotImplementedError

    def sample(self, num_samples: int = 1, **kwargs) -> jnp.ndarray:
        """Convenience wrapper around `forward` that returns only samples."""
        z, _ = self.forward(num_samples=num_samples, **kwargs)
        return z


class DiagGaussian(BaseDistribution):
    """
    Multivariate diagonal Gaussian N(loc, diag(scale^2)).
    `shape` is the event shape (e.g., (D,) or (H,W,C)).
    """
    shape: Tuple[int, ...] | int | Sequence[int]
    trainable: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self._n_dim = len(self.shape)
        self._d = int(np.prod(self.shape))

        # Trainable parameters: loc and log_scale of shape (1, *shape)
        loc_init = lambda key: jnp.zeros((1,) + self.shape, dtype=self.dtype)
        log_scale_init = lambda key: jnp.zeros((1,) + self.shape, dtype=self.dtype)

        self._loc = self.param('loc', loc_init)
        self._log_scale = self.param('log_scale', log_scale_init)

    def _effective_log_scale(self, temperature: Optional[float]) -> jnp.ndarray:
        if temperature is None:
            return self._log_scale
        return self._log_scale + jnp.asarray(jnp.log(temperature), dtype=self.dtype)

    def forward(self,sample_key, num_samples: int = 1, temperature: Optional[float] = None):
        """
        Draw `num_samples` iid samples. Requires rngs={'sample': key} in apply().
        Returns:
          z: (num_samples, *shape)
          log_p: (num_samples,)
        """
        eps = jax.random.normal(sample_key, (num_samples,) + self.shape, dtype=self.dtype)

        log_scale = self._effective_log_scale(temperature)
        scale = jnp.exp(log_scale)  # (1, *shape)

        # Broadcast loc/scale to (N, *shape)
        z = self._loc + scale * eps

        # log N = -0.5 d log(2π) - sum(log σ) - 0.5 * sum(eps^2)
        const = -0.5 * self._d * jnp.log(2.0 * jnp.pi)
        # Sum over event dims (axes 1..n_dim)
        axes = tuple(range(1, 1 + self._n_dim))
        log_p = const - jnp.sum(log_scale + 0.5 * eps**2, axis=axes)
        return z, log_p

    def log_prob(self, z: jnp.ndarray, *, temperature: Optional[float] = None) -> jnp.ndarray:
        """
        z: (..., *shape)
        Returns:
          log_p: (...)  (batch dims preserved)
        """
        z = jnp.asarray(z, dtype=self.dtype)
        log_scale = self._effective_log_scale(temperature)
        scale = jnp.exp(log_scale)

        # Broadcast (1,*shape) params to z's shape
        # Standardized residuals
        eps = (z - self._loc) / scale

        const = -0.5 * self._d * jnp.log(2.0 * jnp.pi)
        axes = tuple(range(z.ndim - self._n_dim, z.ndim))  # last len(shape) axes
        log_p = const - jnp.sum(log_scale + 0.5 * eps**2, axis=axes)
        return log_p

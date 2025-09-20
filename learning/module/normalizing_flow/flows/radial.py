from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import Flow
Array = jnp.ndarray


def _u_constrained(u: Array, w: Array, eps: float = 1e-8) -> Array:
    """
    Enforce w^T u > -1 (for invertibility) via the standard planar-flow trick:
      u_hat = u + ((softplus(w^T u) - 1) - w^T u) * w / ||w||^2
    """
    wu = jnp.dot(w, u)                                # scalar
    scale = (jax.nn.softplus(wu) - 1.0 - wu) / (jnp.sum(w * w) + eps)
    return u + scale * w

def _h_and_hprime(activation: Callable[[Array], Array]) -> Tuple[Callable[[Array], Array],
                                                                Callable[[Array], Array]]:
    if activation is jnp.tanh or activation is jax.nn.tanh:
        # tanh'(x) = 1 / cosh(x)^2
        return jnp.tanh, (lambda x: 1.0 / jnp.cosh(x) ** 2)
    elif activation is jax.nn.leaky_relu:
        # We'll use negative_slope=0.01 by default; customize by wrapping activation.
        slope = 0.2
        h = lambda x: jax.nn.leaky_relu(x, negative_slope=slope)
        hprime = lambda x: jnp.where(x < 0, slope, 1.0)
        return h, hprime
    else:
        raise ValueError("Unsupported activation for planar flow. Use tanh or leaky_relu.")

class Radial(Flow):

    """ Planar flow
        f(z) = z + u * h(w*z + b)

    """

    shape: Tuple[int, ...]
    # activation: Callable[[Array], Array] = jnp.tanh
    # Optional custom initial values (arrays) or None to use default initializers
    alpha_init: Array | None = None
    beta_init: Array | None = None
    def setup(self):
        # D = int(jnp.prod(jnp.array(self.shape)))
        self.D = jnp.prod(jnp.array(self.shape))
        if len(self.shape) != 1:
            raise ValueError(f"Planner expects 1D feature shape (D,), got {self.shape}")

        lim = 1.0 / self.D

        def uniform(minv, maxv):
            return lambda key, shape: jax.random.uniform(key, shape, minval=minv, maxval=maxv)
        def normal():
            return lambda key, shape: jax.random.normal(key, shape)
        def constant(arr):
            return lambda key, shape: jnp.asarray(arr).reshape(shape)
        self.alpha = self.param('alpha', uniform(-lim,lim) if self.alpha_init is None else constant(self.alpha_init), (1,))
        self.beta = self.param('beta', uniform(-lim-1,lim-1) if self.beta_init is None else constant(self.beta_init), (1,))
        self.z_0 = self.param('z_0', normal(), self.shape)

    
    def forward(self, z) -> Tuple[Array, Array]:
        beta = jnp.log(1 + jnp.exp(self.beta)) - jnp.abs(self.alpha)
        dz = z - self.z_0
        r = jnp.linalg.vector_norm(dz, axis=list(range(1, self.shape[0])), keepdims=True)
        h_arr = beta / (jnp.abs(self.alpha) + r)
        h_arr_ = -beta * r / (jnp.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        log_det = (self.D - 1) * jnp.log(1 + h_arr) + jnp.log(1 + h_arr + h_arr_)
        log_det = log_det.reshape(-1)
        return z_, log_det

    # def inverse(self, z):
    #     if self.activation is not jax.nn.leaky_relu:
    #         raise NotImplementedError("This flow has no algebraic inverse.")
    #     u_hat = _u_constrained(self.u, self.w)
    #     lin = jnp.einsum('...d,d->...', z, self.w) + self.b 
    #     slope = 0.01
    #     a = jnp.where(lin < 0.0, slope, 1.0)                   # [...]
    #     u_eff = a[..., None] * u_hat                # [..., D]
    #     inner = jnp.einsum('d,...d->...', self.w, u_eff)       # [...]
    #     # z' = z - u_eff * (lin / (1 + inner))
    #     denom = (1.0 + inner)
    #     z_inv = z - u_eff * (lin / (denom))[..., None]
    #     log_det = -jnp.log(jnp.abs(denom) + 1e-12)
    #     return z_inv, log_det

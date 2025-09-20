from typing import Tuple
import jax
import jax.numpy as jnp

from .base import Flow
from .affine.coupling import AffineConstFlow


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    shape: Tuple[int, ...]
    use_scale : bool = True
    use_shift : bool = True
    data_dep_init_done = jnp.asarray(0.0)
    def _maybe_data_init(self, x):
        # Keep a state flag in a non-param collection
        initialized = self.variable('actnorm', 'initialized', lambda: False)
        if initialized.value:
            return

        # Reduce over leading batch axes; keep event dims = last len(shape) axes
        axes = tuple(range(x.ndim - len(self.shape)))  # all batch dims
        if self.use_scale:
            std = jnp.std(x, axis=axes, keepdims=True)  # (..., *shape) -> (1,*shape) by keepdims
            s_init = -jnp.log(std + self.eps)
            # mutate existing param 's' in this module
            s_var = self.variable('params', 's', lambda k, shp: jnp.zeros(shp), self.shape)
            s_var.value = s_init
        else:
            s_init = 0.0

        if self.use_shift:
            mean = jnp.mean(x, axis=axes, keepdims=True)
            t_init = -mean * jnp.exp(s_init)
            t_var = self.variable('params', 't', lambda k, shp: jnp.zeros(shp), self.shape)
            t_var.value = t_init

        initialized.value = True

    def forward(self, z):
        self._maybe_data_init(z)
        return super().forward(z)

    def inverse(self, z):
        self._maybe_data_init(z)
        return super().inverse(z)


# class BatchNorm(Flow):
#     """
#     Batch Normalization with out considering the derivatives of the batch statistics, see [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)
#     """

#     def __init__(self, eps=1.0e-10):
#         super().__init__()
#         self.eps_cpu = torch.tensor(eps)
#         self.register_buffer("eps", self.eps_cpu)

#     def forward(self, z):
#         """
#         Do batch norm over batch and sample dimension
#         """
#         mean = torch.mean(z, axis=0, keepdims=True)
#         std = torch.std(z, axis=0, keepdims=True)
#         z_ = (z - mean) / torch.sqrt(std**2 + self.eps)
#         log_det = torch.log(1 / torch.prod(torch.sqrt(std**2 + self.eps))).repeat(
#             z.size()[0]
#         )
#         return z_, log_det

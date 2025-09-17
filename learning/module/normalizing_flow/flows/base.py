from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class Flow(nn.Module):
    """Generic flow base class."""

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        return self.inverse(x) if reverse else self.forward(x)

    def forward(self, z):
        """Return (z_out, logabsdet) with logabsdet shaped like z's batch dims."""
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class Reverse(Flow):
    """Wraps a flow and swaps forward/inverse."""
    flow: Flow

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        return self.inverse(x) if reverse else self.forward(x)

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow.forward(z)


class Composite(Flow):
    """Compose flows in the given order: f = f_K ∘ ... ∘ f_1."""
    _flows: Tuple[Flow, ...]  # use an immutable tuple for Flax

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        return self.inverse(x) if reverse else self.forward(x)

    @staticmethod
    def _cascade(inputs, funcs):
        # Support arbitrary batch shape: z.shape == (*B, D)
        batch_shape = inputs.shape[:-1]
        total_logabsdet = jnp.zeros(batch_shape, dtype=inputs.dtype)
        outputs = inputs
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet = total_logabsdet + logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        # Calling a Flow module directly invokes its __call__ (i.e., forward)
        funcs = self._flows
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        # Build callables that invoke each flow's inverse in reverse order
        funcs = tuple((lambda z, f=f: f.inverse(z)) for f in self._flows[::-1])
        return self._cascade(inputs, funcs)


def zero_log_det_like_z(z):
    """Zeros with the same batch shape as z (assumes last dim is features)."""
    return jnp.zeros(z.shape[:-1], dtype=z.dtype)

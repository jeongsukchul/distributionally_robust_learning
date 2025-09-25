"""
Implementations of autoregressive transforms.
Code taken from https://github.com/bayesiains/nsf
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from ..base import Flow, zero_log_det_like_z
from ..made import MADE
from . import splines
from typing import Sequence, Optional, Callable, Union, Tuple





class PeriodicFeaturesElementwise(nn.Module):
    """
    Replace selected features f with w1*sin(scale*f) + w2*cos(scale*f),
    elementwise per selected dimension.
    - Output has the same shape as input.
    - Some information may be lost (2→1 mapping per selected dim).
    """
    ndim: int                                   # total number of features (last axis)
    ind: Sequence[int]                          # indices to convert
    scale: Union[float, jnp.ndarray] = 1.0      # scalar, (len(ind),) or (ndim,)
    bias: bool = False
    activation: Optional[Callable] = None       # e.g., jax.nn.tanh

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        inputs: (..., ndim)
        returns: (..., ndim)
        """
        dtype = inputs.dtype
        ind_arr = jnp.asarray(self.ind, dtype=jnp.int32)
        k = ind_arr.shape[0]

        # complement indices (kept as-is)
        ind_set = set(self.ind)
        ind_rest = jnp.asarray([i for i in range(self.ndim) if i not in ind_set],
                               dtype=jnp.int32)

        # permutation to bring [selected, rest] to front, and its inverse to restore order
        perm = jnp.concatenate([ind_arr, ind_rest], axis=0)              # (ndim,)
        inv_perm = jnp.empty_like(perm)
        inv_perm = inv_perm.at[perm].set(jnp.arange(self.ndim, dtype=jnp.int32))

        # learnable weights (k, 2), optional bias (k,)
        w = self.param("weights", lambda key: jnp.ones((k, 2), dtype=dtype))
        w_sin = w[:, 0]
        w_cos = w[:, 1]
        if self.bias:
            b = self.param("bias", lambda key: jnp.zeros((k,), dtype=dtype))
        else:
            b = None

        # handle scale broadcasting: allow scalar, (k,), or (ndim,)
        scale_arr = jnp.asarray(self.scale, dtype=dtype)
        if scale_arr.ndim == 0:
            scale_sel = scale_arr                             # scalar
        elif scale_arr.shape == (self.ndim,):
            scale_sel = scale_arr[ind_arr]                    # pick selected dims
        elif scale_arr.shape == (k,):
            scale_sel = scale_arr                             # per-selected-dim
        else:
            raise ValueError(
                f"scale must be scalar, shape ({k},) or ({self.ndim},), got {scale_arr.shape}"
            )

        # pick selected features, apply periodic mapping
        x_sel = jnp.take(inputs, ind_arr, axis=-1)            # (..., k)
        x_scaled = scale_sel * x_sel
        y_sel = w_sin * jnp.sin(x_scaled) + w_cos * jnp.cos(x_scaled)
        if b is not None:
            y_sel = y_sel + b
        if self.activation is not None:
            y_sel = self.activation(y_sel)

        # keep the rest as-is, then restore original order
        x_rest = jnp.take(inputs, ind_rest, axis=-1)          # (..., ndim - k)
        stacked = jnp.concatenate([y_sel, x_rest], axis=-1)   # (..., ndim)
        out = jnp.take(stacked, inv_perm, axis=-1)            # back to original feature order
        return out

"""
Autoregressive RQS transforms in JAX/Flax
Port of the PyTorch code you shared.
"""

def sum_except_batch(x: jnp.ndarray) -> jnp.ndarray:
    """Sum over all dims except the batch dim."""
    if x.ndim == 1:
        return x
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes)


class Autoregressive(nn.Module):
    """Base autoregressive flow in Flax.

    Subclasses implement `_elementwise(...)` and `_output_dim_multiplier()`.
    The `autoregressive_net` is a Flax module (e.g., MADE) returning params per feature.
    """

    @nn.compact
    def __call__(self,
                 inputs: jnp.ndarray,
                 context: Optional[jnp.ndarray] = None,
                 *,
                 training: bool = False,
                 rngs: Optional[dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward transform (x -> y, log|detJ|)."""
        params = self._net_apply(inputs, context, training, rngs)
        outputs, logabsdet = self._elementwise(inputs, params, inverse=False)
        return outputs, sum_except_batch(logabsdet)

    # forward() alias for parity with your PyTorch API
    def forward(self,
                inputs: jnp.ndarray,
                context: Optional[jnp.ndarray] = None,
                *,
                training: bool = False,
                rngs: Optional[dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.__call__(inputs, context, training=training, rngs=rngs)

    def inverse(self, inputs, context=None, *, training: bool = False, rngs=None):
        D = int(np.prod(inputs.shape[1:]))
        y = jnp.zeros_like(inputs)

        def step(_, y):
            params = self._net_apply(y, context, training, rngs)
            y, _ = self._elementwise(inputs, params, inverse=True)
            return y

        # y_final = jax.lax.fori_loop(0, D, step, y)
        for i in range(D):
            params = self._net_apply(y, context, training, rngs)
            y, _ = self._elementwise(inputs, params, inverse=True)
        y_final = y
        # compute final logdet with the final parameters
        params = self._net_apply(y_final, context, training, rngs)
        _, ladj = self._elementwise(inputs, params, inverse=True)
        return y_final, sum_except_batch(ladj)
    def _net_apply(self,
                   x: jnp.ndarray,
                   context: Optional[jnp.ndarray],
                   training: bool,
                   rngs: Optional[dict]) -> jnp.ndarray:
        # Submodules are traced from fields, so we can just call them
        raise NotImplementedError()
    # to be implemented by subclass
    def _elementwise(self, inputs: jnp.ndarray, autoregressive_params: jnp.ndarray, inverse: bool):
        raise NotImplementedError()

    def _output_dim_multiplier(self) -> int:
        raise NotImplementedError()


class MaskedPiecewiseRationalQuadraticAutoregressive(Autoregressive):
    """
    MADE + elementwise RQS (Durkan et al. 2019).

    Features:
      - linear / circular tails (like the original)
      - optional batch-norm/dropout in MADE blocks
      - identity initialization via a learnable bias and a 0-initialized gate (alpha)
    """
    features: int
    hidden_features: int
    context_features: Optional[int] = None
    num_bins: int = 10
    tails: Optional[str] = None       # None, "linear", or "circular"
    tail_bound: float = 1.0
    num_blocks: int = 2
    use_residual_blocks: bool = True
    random_mask: bool = False
    permute_mask: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu
    dropout_probability: float = 0.0
    use_batch_norm: bool = False
    init_identity: bool = True
    min_bin_width: float = splines.DEFAULT_MIN_BIN_WIDTH
    min_bin_height: float = splines.DEFAULT_MIN_BIN_HEIGHT
    min_derivative: float = splines.DEFAULT_MIN_DERIVATIVE

    def setup(self):
        # MADE submodule (outputs D * output_multiplier)
        self.made = MADE(
            features=self.features,
            hidden_features=self.hidden_features,
            context_features=self.context_features,
            num_blocks=self.num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=self.use_residual_blocks,
            random_mask=self.random_mask,
            permute_mask=self.permute_mask,
            activation=self.activation,
            dropout_probability=self.dropout_probability,
            use_batch_norm=self.use_batch_norm,
            preprocessing=None,  # (PeriodicFeaturesElementwise not ported here)
        )

        # Identity init: y = init_bias + alpha * net_out
        # alpha starts at 0 => pure identity; learned during training.
        if self.init_identity:
            out_mult = self._output_dim_multiplier()
            const = float(np.log(np.exp(1.0 - self.min_derivative) - 1.0))  # so softplus + min_derivative = 1
            # per-feature bias template: widths/heights -> 0; derivatives -> constant
            bias_template = jnp.zeros((self.features, out_mult), dtype=jnp.float32)
            # slice indices
            k = self.num_bins
            d_start = 2 * k
            # linear: (K-1) derivs; circular: K derivs; none: K+1 derivs
            deriv_len = out_mult - 2 * k
            bias_template = bias_template.at[:, d_start:d_start + deriv_len].set(const)

            self.init_bias = self.param(
                "init_bias",
                lambda key: bias_template
            )
            self.alpha = self.param(
                "alpha",
                lambda key: jnp.array(0.0, dtype=jnp.float32)
            )
        else:
            # still define fields for simplicity
            self.init_bias = None
            self.alpha = None

    # override base to use our MADE (so we can apply the identity gate)
    def _net_apply(self,
                   x: jnp.ndarray,
                   context: Optional[jnp.ndarray],
                   training: bool,
                   rngs: Optional[dict]) -> jnp.ndarray:
        net_out = self.made(x, context=context, training=training)#, rngs=(rngs or {}))  # (B, D*out_mult)
        if self.init_identity:
            B = x.shape[0]
            out_mult = self._output_dim_multiplier()
            net_out = jnp.reshape(net_out, (B, self.features, out_mult))
            # outputs = init_bias + alpha * net_out
            outputs = self.init_bias[None, ...] + self.alpha * net_out
            return jnp.reshape(outputs, (B, self.features * out_mult))
        else:
            return net_out

    def _output_dim_multiplier(self) -> int:
        k = self.num_bins
        if self.tails == "linear":
            return 3 * k - 1   # K (w) + K (h) + (K-1) (derivs)
        elif self.tails == "circular":
            return 3 * k       # K + K + K
        else:
            return 3 * k + 1   # K + K + (K+1)

    def _elementwise(self, inputs: jnp.ndarray, autoregressive_params: jnp.ndarray, inverse: bool):
        B, D = inputs.shape
        m = self._output_dim_multiplier()
        k = self.num_bins

        params = jnp.reshape(autoregressive_params, (B, D, m))
        unnormalized_widths = params[..., :k]
        unnormalized_heights = params[..., k:2 * k]
        unnormalized_derivatives = params[..., 2 * k:]

        # (Optional) match the PyTorch reference scaling when MADE is deep
        # (This helps stabilize training initially.)
        if hasattr(self.made, "hidden_features"):  # always true here
            scale = float(np.sqrt(self.hidden_features))
            unnormalized_widths = unnormalized_widths / scale
            unnormalized_heights = unnormalized_heights / scale

        # Choose spline fn
        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = dict(
                left=0.0, right=1.0, bottom=0.0, top=1.0  # will be ignored by wrapper usage here
            )
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = dict(
                tails=self.tails,
                tail_bound=self.tail_bound,
            )

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )
        # logabsdet is (B, D)
        return outputs, logabsdet

class AutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """
    num_input_channels: int
    num_blocks : int
    num_hidden_channels : int
    num_context_channels : Optional[int] =  None
    num_bins : int = 8
    tail_bound : int = 3
    activation : callable = jax.nn.relu
    dropout_prob : float =0.0
    permute_mask : bool = False
    init_identity : bool = True
    def setup(
        self
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch.nn.Module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=self.num_input_channels,
            hidden_features=self.num_hidden_channels,
            context_features=self.num_context_channels,
            num_bins=self.num_bins,
            tails="linear",
            tail_bound=self.tail_bound,
            num_blocks=self.num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=self.permute_mask,
            activation=self.activation,
            dropout_probability=self.dropout_prob,
            use_batch_norm=False,
            init_identity=self.init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        # print("log det forward shape", log_det)
        return z, log_det#.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        # print("log det inverse shape", log_det)
        return z, log_det#.view(-1)

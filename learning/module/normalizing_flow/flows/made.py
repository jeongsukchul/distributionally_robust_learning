# MADE in JAX/Flax (Linen)
# ------------------------------------------------------------
# Requirements:
#   pip install flax jax jaxlib
#
# Notes:
# - Dropout/BatchNorm need a training flag.
# - If random_mask=True, masks are sampled at init time from the 'params' RNG.
# - If permute_mask=True, input degrees are randomly permuted at init time (also from 'params' RNG).
#
# Example use:
#   import jax, jax.numpy as jnp
#   from flax.core import FrozenDict
#
#   key = jax.random.PRNGKey(0)
#   model = MADE(
#       features=8,
#       hidden_features=64,
#       num_blocks=2,
#       use_residual_blocks=True,
#       output_multiplier=1,
#       random_mask=False,
#       permute_mask=False,
#       use_batch_norm=False,
#       dropout_probability=0.0,
#   )
#   x = jnp.zeros((16, 8))
#   variables = model.init({'params': key, 'dropout': key}, x, context=None, training=True)
#   y = model.apply(variables, x, context=None, training=True, rngs={'dropout': key})
# ------------------------------------------------------------

from typing import Optional, Callable, Any, List, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn


def _get_input_degrees(n_features: int) -> jnp.ndarray:
    # 1, 2, ..., D
    return jnp.arange(1, n_features + 1, dtype=jnp.float32)


def _tile_degrees(base: jnp.ndarray, reps: int) -> jnp.ndarray:
    # Equivalent to PyTorch nsf.utils.nn.tile used for degrees
    return jnp.tile(base, (reps,))


def _get_mask_and_degrees(
    in_degrees: jnp.ndarray,
    out_features: int,
    autoregressive_features: int,
    random_mask: bool,
    is_output: bool,
    rng: Optional[jax.random.PRNGKey] = None,
    out_degrees_override: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Replicates the PyTorch logic to produce (mask, out_degrees) given in_degrees.
    If out_degrees_override is provided, it is used directly (and tiled when is_output=True).
    """
    in_degrees = jnp.asarray(in_degrees, dtype=jnp.float32)
    in_features = in_degrees.shape[0]

    if is_output:
        # Output layer: mask[i_out, i_in] = (out_degree > in_degree)
        if out_degrees_override is None:
            out_degrees_base = _get_input_degrees(autoregressive_features)
        else:
            out_degrees_base = jnp.asarray(out_degrees_override, dtype=jnp.float32)

        assert out_features % autoregressive_features == 0, (
            "For output layers, out_features must be a multiple of autoregressive_features."
        )
        out_degrees = _tile_degrees(out_degrees_base, out_features // autoregressive_features)
        mask = (out_degrees[..., None] > in_degrees).astype(jnp.float32)

    else:
        if out_degrees_override is not None:
            out_degrees = jnp.asarray(out_degrees_override, dtype=jnp.float32)
        else:
            if random_mask:
                if rng is None:
                    raise ValueError("random_mask=True requires RNG during init.")
                min_in_degree = int(jnp.min(in_degrees))
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                # randint high is exclusive, matches torch.randint(low, high)
                out_degrees = jax.random.randint(
                    rng,
                    (out_features,),
                    min_in_degree,
                    autoregressive_features,
                    dtype=jnp.float32,
                )
            else:
                # Deterministic rule from the original code
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = (jnp.arange(out_features, dtype=jnp.float32) % max_) + min_

        # Hidden layers: mask[i_out, i_in] = (out_degree >= in_degree)
        mask = (out_degrees[..., None] >= in_degrees).astype(jnp.float32)

    # mask shape: (out_features, in_features) but we want (in_features, out_features) for weight masking
    # We’ll transpose when applying to the kernel so that x @ (W * mask) works with (B, in) x (in, out).
    return mask.T, out_degrees


class MaskedDense(nn.Module):
    """Dense layer with an autoregressive mask applied to the kernel."""
    in_degrees: jnp.ndarray
    out_features: int
    autoregressive_features: int
    is_output: bool
    random_mask: bool = False
    use_bias: bool = True
    # If provided, this fixes the out_degrees (and therefore the mask) deterministically.
    fixed_out_degrees: Optional[jnp.ndarray] = None
    # Initializers (customizable for residual block's last layer)
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    # Store the final degrees for introspection (static field; not a pytree)
    # degrees: Any = None #nn.field(default=None, pytree_node=False)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_degrees = jnp.asarray(self.in_degrees, dtype=jnp.float32)
        in_features = int(in_degrees.shape[0])

        rng = None
        if self.random_mask and self.fixed_out_degrees is None:
            # Pull randomness for degrees from params RNG at init-time
            rng = self.make_rng("params")

        mask, out_degrees = _get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=self.out_features,
            autoregressive_features=self.autoregressive_features,
            random_mask=self.random_mask,
            is_output=self.is_output,
            rng=rng,
            out_degrees_override=self.fixed_out_degrees,
        )
        # Persist mask and degrees as non-trainable variables
        # (so that they are part of the state and accessible if needed)
        mask_var = self.variable("constants", "mask", lambda: mask)
        # deg_var = self.variable("constants", "degrees", lambda: out_degrees)
        mask = mask_var.value
        # self.degrees = deg_var.value  # static copy for read-only (not required downstream)

        kernel = self.param(
            "kernel", self.kernel_init, (in_features, self.out_features), x.dtype
        )
        kernel = kernel * mask  # apply mask

        y = jnp.dot(x, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.out_features,), x.dtype)
            y = y + bias
        return y


class MaskedFeedforwardBlock(nn.Module):
    """Feedforward block with one masked dense layer and optional BN/activation/dropout.

    Matches the PyTorch implementation where out_features == in_features.
    """
    in_degrees: jnp.ndarray
    out_degrees: jnp.ndarray
    autoregressive_features: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_probability: float = 0.0
    use_batch_norm: bool = False

    # For debugging/introspection (static)
    # degrees: Any = None # nn.field(default=None, pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, training: bool, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        del context  # not implemented to mirror original
        features = inputs.shape[-1]

        x = inputs
        if self.use_batch_norm:
            bn = nn.BatchNorm(use_running_average=not training, epsilon=1e-3, axis=-1)
            x = bn(x)

        linear = MaskedDense(
            in_degrees=self.in_degrees,
            out_features=features,
            autoregressive_features=self.autoregressive_features,
            is_output=False,
            random_mask=False,
            fixed_out_degrees=self.out_degrees,
        )
        x = linear(x)
        # self.degrees = linear.degrees  # keep for parity

        x = self.activation(x)
        if self.dropout_probability > 0.0:
            x = nn.Dropout(self.dropout_probability)(x, deterministic=not training)
        return x


class MaskedResidualBlock(nn.Module):
    """Residual block with two masked dense layers + optional BN/dropout and GLU context."""
    in_degrees: jnp.ndarray
    out_degrees_0: jnp.ndarray
    out_degrees_1: jnp.ndarray
    autoregressive_features: int
    context_features: Optional[int] = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_probability: float = 0.0
    use_batch_norm: bool = False
    zero_initialization: bool = True  # near-zero init for final linear

    # degrees: Any = None # nn.field(default=None, pytree_node=False)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *, training: bool, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        features = inputs.shape[-1]

        # Optional context projection for GLU gating
        ctx_proj = None
        if self.context_features is not None:
            ctx_proj_layer = nn.Dense(features)
            if context is None:
                raise ValueError("context_features provided but context=None at call.")
            ctx_proj = ctx_proj_layer(context)  # shape (..., features)

        # BN layers
        bn1 = bn2 = None
        if self.use_batch_norm:
            bn1 = nn.BatchNorm(use_running_average=not training, epsilon=1e-3, axis=-1)
            bn2 = nn.BatchNorm(use_running_average=not training, epsilon=1e-3, axis=-1)

        x = inputs
        if bn1 is not None:
            x = bn1(x)
        x = self.activation(x)

        # First masked dense
        linear0 = MaskedDense(
            in_degrees=self.in_degrees,
            out_features=features,
            autoregressive_features=self.autoregressive_features,
            is_output=False,
            random_mask=False,
            fixed_out_degrees=self.out_degrees_0,
        )
        x = linear0(x)

        if bn2 is not None:
            x = bn2(x)
        x = self.activation(x)
        if self.dropout_probability > 0.0:
            x = nn.Dropout(self.dropout_probability)(x, deterministic=not training)

        # Second masked dense (optionally tiny init)
        if self.zero_initialization:
            tiny = 1e-3
            kinit = nn.initializers.uniform(scale=tiny)
            binit = nn.initializers.uniform(scale=tiny)
        else:
            kinit = nn.initializers.lecun_normal()
            binit = nn.initializers.zeros

        linear1 = MaskedDense(
            in_degrees=self.out_degrees_0,
            out_features=features,
            autoregressive_features=self.autoregressive_features,
            is_output=False,
            random_mask=False,
            fixed_out_degrees=self.out_degrees_1,
            kernel_init=kinit,
            bias_init=binit,
        )
        x = linear1(x)

        # GLU with context: cat(temps, ctx) then GLU -> temps * sigmoid(ctx)
        if ctx_proj is not None:
            x = x * jax.nn.sigmoid(ctx_proj)

        # Save final degrees for parity and validation
        # self.degrees = linear1.degrees
        # # Validate degrees monotonicity (Python assert so it runs at init time)
        # if not bool(jnp.all(self.degrees >= self.in_degrees)):
        #     raise ValueError(
        #         "In a masked residual block, the output degrees can't be less than the input degrees."
        #     )

        return inputs + x


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation (MADE) in Flax.

    Matches the PyTorch implementation:
      - residual or feedforward blocks
      - optional BatchNorm and Dropout inside blocks
      - optional context (added after the first masked layer; GLU in residual blocks)
      - optional random masks and input-degree permutation
    """
    features: int
    hidden_features: int
    context_features: Optional[int] = None
    num_blocks: int = 2
    output_multiplier: int = 1
    use_residual_blocks: bool = True
    random_mask: bool = False
    permute_mask: bool = False
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_probability: float = 0.0
    use_batch_norm: bool = False
    preprocessing: Optional[nn.Module] = None
    residual_zero_initialization: bool = True  # mirrors residual block's zero_initialization

    # --- helpers to precompute degrees per layer at init-time ---
    def _compute_degrees_chain(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, List[Any], jnp.ndarray]:
        """
        Precomputes the out_degrees for:
          - initial masked dense
          - each block (either one or two masked layers per block)
          - final masked dense
        Returns:
          input_degrees, blocks_degrees_list, final_out_degrees_base
        """
        key = rng

        # Input degrees (optionally permuted)
        input_degrees = _get_input_degrees(self.features)
        if self.permute_mask:
            key, pkey = jax.random.split(key)
            perm = jax.random.permutation(pkey, self.features)
            input_degrees = input_degrees[perm]

        # Initial layer degrees (hidden_features)
        key, ikey = jax.random.split(key)
        _, init_out_degrees = _get_mask_and_degrees(
            in_degrees=input_degrees,
            out_features=self.hidden_features,
            autoregressive_features=self.features,
            random_mask=self.random_mask,
            is_output=False,
            rng=ikey if self.random_mask else None,
        )

        blocks_degrees: List[Any] = []
        prev = init_out_degrees

        # Build degrees through blocks
        for _ in range(self.num_blocks):
            if self.use_residual_blocks:
                # Two masked layers per residual block
                key, k0 = jax.random.split(key)
                _, out0 = _get_mask_and_degrees(
                    in_degrees=prev,
                    out_features=self.hidden_features,
                    autoregressive_features=self.features,
                    random_mask=False,  # residual requires deterministic
                    is_output=False,
                )
                key, k1 = jax.random.split(key)
                _, out1 = _get_mask_and_degrees(
                    in_degrees=out0,
                    out_features=self.hidden_features,
                    autoregressive_features=self.features,
                    random_mask=False,
                    is_output=False,
                )
                blocks_degrees.append((out0, out1))
                prev = out1
            else:
                # Single masked layer in a feedforward block; may be random if requested
                key, bkey = jax.random.split(key)
                _, out = _get_mask_and_degrees(
                    in_degrees=prev,
                    out_features=self.hidden_features,
                    autoregressive_features=self.features,
                    random_mask=self.random_mask,
                    is_output=False,
                    rng=bkey if self.random_mask else None,
                )
                blocks_degrees.append(out)
                prev = out

        # Final layer uses input degrees (tiled inside layer)
        final_out_degrees_base = input_degrees  # base to tile over multipliers internally
        return (input_degrees, [init_out_degrees, *blocks_degrees], final_out_degrees_base)
    def setup(self):
        # compute once, deterministically from degrees_seed
        def _init_degrees():
            key = jax.random.PRNGKey(0)
            return self._compute_degrees_chain(key)
        self.degrees_pkg = self.variable('constants', 'degrees_pkg', _init_degrees)
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        *,
        training: bool,
    ) -> jnp.ndarray:
        # Preprocessing module (or identity)
        x = inputs
        if self.preprocessing is not None:
            x = self.preprocessing(x)

        # Precompute degrees deterministically at init (from params RNG)
        input_degrees, degrees_chain, final_base_degrees = self.degrees_pkg.value
        # input_degrees, degrees_chain, final_base_degrees = self._compute_degrees_chain(params_rng)

        # Initial masked linear (to hidden_features)
        initial = MaskedDense(
            in_degrees=input_degrees,
            out_features=self.hidden_features,
            autoregressive_features=self.features,
            is_output=False,
            random_mask=self.random_mask,
            fixed_out_degrees=degrees_chain[0],  # locks exact degrees used above
        )
        h = initial(x)

        # Optional context addition after initial layer
        if self.context_features is not None and context is not None:
            ctx_layer = nn.Dense(self.hidden_features)
            h = h + ctx_layer(context)

        # Blocks
        blocks_out = h
        blocks_defs = degrees_chain[1:]
        for bd in blocks_defs:
            if self.use_residual_blocks:
                out0, out1 = bd  # two masked layers in the block
                block = MaskedResidualBlock(
                    in_degrees=out0 * 0 + out0 - out0 + (bd[0] * 0 + bd[0]).astype(jnp.float32)  # silence static checker
                    if False else bd[0],  # no-op; keep pytrees stable
                    out_degrees_0=bd[0],
                    out_degrees_1=bd[1],
                    autoregressive_features=self.features,
                    context_features=self.context_features,
                    activation=self.activation,
                    dropout_probability=self.dropout_probability,
                    use_batch_norm=self.use_batch_norm,
                    zero_initialization=self.residual_zero_initialization,
                )
                blocks_out = block(blocks_out, training=training, context=context)
            else:
                block = MaskedFeedforwardBlock(
                    in_degrees=bd * 0 + bd - bd + (bd * 0 + bd).astype(jnp.float32) if False else bd,  # no-op
                    out_degrees=bd,
                    autoregressive_features=self.features,
                    activation=self.activation,
                    dropout_probability=self.dropout_probability,
                    use_batch_norm=self.use_batch_norm,
                )
                blocks_out = block(blocks_out, training=training, context=None)

        # Final masked linear to features * output_multiplier
        final = MaskedDense(
            in_degrees=blocks_defs[-1][1] if self.use_residual_blocks else blocks_defs[-1],
            out_features=self.features * self.output_multiplier,
            autoregressive_features=self.features,
            is_output=True,
            random_mask=self.random_mask,
            fixed_out_degrees=final_base_degrees,  # base degrees; layer will tile internally
        )
        outputs = final(blocks_out)
        return outputs

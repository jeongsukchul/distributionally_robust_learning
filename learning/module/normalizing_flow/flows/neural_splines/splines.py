import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3
_EPS = 1e-12


def _pad_last(x: jnp.ndarray, pad_left: int, pad_right: int, value: float = 0.0) -> jnp.ndarray:
    """Pad only the last axis."""
    pads = [(0, 0)] * x.ndim
    pads[-1] = (pad_left, pad_right)
    return jnp.pad(x, pads, mode="constant", constant_values=value)


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.at[..., -1].add(eps)
    idx = jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1
    K = bin_locations.shape[-1] - 1
    return jnp.clip(idx, 0, K - 1)

def _cum_from_unnorm(
    unnormalized: jnp.ndarray,
    num_bins: int,
    min_bin: float,
    left: jnp.ndarray,
    right: jnp.ndarray,
):
    """Softmax -> enforce min_bin -> cumulative -> affine map to [left,right]."""
    w = jax.nn.softmax(unnormalized, axis=-1)
    w = min_bin + (1.0 - min_bin * num_bins) * w
    cum = jnp.cumsum(w, axis=-1)
    zero = jnp.zeros_like(cum[..., :1])
    cum = jnp.concatenate([zero, cum], axis=-1)  # (..., K+1) in [0,1]
    rng = right - left
    cum = left[..., None] + rng[..., None] * cum
    # Ensure exact endpoints
    cum = cum.at[..., 0].set(left)
    cum = cum.at[..., -1].set(right)
    bins = cum[..., 1:] - cum[..., :-1]
    return cum, bins  # (..., K+1), (..., K)


def unconstrained_rational_quadratic_spline(
    inputs: jnp.ndarray,
    unnormalized_widths: jnp.ndarray,
    unnormalized_heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    inverse: bool = False,
    tails: str | tuple | list = "linear",
    tail_bound: float | jnp.ndarray = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    Wrapper that clamps the transform to identity outside [-tail_bound, tail_bound]
    (or per-sample bounds if array), and sets boundary derivatives according to tails.
    """
    inputs = jnp.asarray(inputs)
    tail_bound = jnp.asarray(tail_bound)

    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # Prepare boundary derivatives according to `tails`
    # Result must have length (num_bins + 1) along last axis.
    if isinstance(tails, (list, tuple)):
        raise NotImplementedError("Per-dimension tails list/tuple not implemented in this JAX port.")

    if tails == "linear":
        # pad both ends and set ends so that softplus(end) = 1 - min_derivative
        ud = _pad_last(unnormalized_derivatives, 1, 1, value=0.0)
        constant = jnp.log(jnp.maximum(jnp.exp(1.0 - min_derivative) - 1.0, 1e-8))
        ud = ud.at[..., 0].set(constant)
        ud = ud.at[..., -1].set(constant)
    elif tails == "circular":
        # ensure length K+1 and set last equal to first
        D = unnormalized_derivatives.shape[-1]
        # We don't know K here yet, but padding (0,1) suffices to get '+1'
        ud = _pad_last(unnormalized_derivatives, 0, 1, value=0.0)
        ud = ud.at[..., -1].set(ud[..., 0])
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    # Run spline on full input, then keep identity outside
    out_all, ladj_all = rational_quadratic_spline(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=ud,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = jnp.where(inside_mask, out_all, inputs)
    logabsdet = jnp.where(inside_mask, ladj_all, jnp.zeros_like(inputs))
    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: jnp.ndarray,
    unnormalized_widths: jnp.ndarray,
    unnormalized_heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    inverse: bool = False,
    left: float | jnp.ndarray = 0.0,
    right: float | jnp.ndarray = 1.0,
    bottom: float | jnp.ndarray = 0.0,
    top: float | jnp.ndarray = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    Vectorized RQS (Durkan et al., 2019). Broadcasts over leading dims.
    Shapes:
      inputs:                   (...,)
      unnormalized_widths:      (..., K)
      unnormalized_heights:     (..., K)
      unnormalized_derivatives: (..., K+1)  # after padding in the wrapper
    Returns:
      outputs, log|det J| with shape (...,)
    """
    inputs = jnp.asarray(inputs)
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    left = jnp.asarray(left)
    right = jnp.asarray(right)
    bottom = jnp.asarray(bottom)
    top = jnp.asarray(top)

    # Cumulated knot locations in x and y
    cumwidths, widths = _cum_from_unnorm(
        unnormalized_widths, num_bins, min_bin_width, left, right
    )
    cumheights, heights = _cum_from_unnorm(
        unnormalized_heights, num_bins, min_bin_height, bottom, top
    )

    # Positive derivatives at knots
    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    # Bin index
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]  # (..., 1)
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]   # (..., 1)

    # Gather helpers
    def gather_last(x, idx):
        return jnp.take_along_axis(x, idx, axis=-1)[..., 0]

    input_cumwidths = gather_last(cumwidths, bin_idx)
    input_bin_widths = gather_last(widths, bin_idx)

    input_cumheights = gather_last(cumheights, bin_idx)
    delta = heights / (widths + _EPS)
    input_delta = gather_last(delta, bin_idx)

    input_derivatives = gather_last(derivatives[..., :-1], bin_idx)
    input_derivatives_plus_one = gather_last(derivatives[..., 1:], bin_idx)
    input_heights = gather_last(heights, bin_idx)

    if inverse:
        # Solve quadratic for theta
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
        root = (2.0 * c) / (-b - jnp.sqrt(disc) + _EPS)  # theta in [0,1]
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus = root * (1.0 - root)
        denom = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus
        )
        denom = jnp.maximum(denom, _EPS)

        deriv_num = (input_delta ** 2) * (
            input_derivatives_plus_one * (root ** 2)
            + 2.0 * input_delta * theta_one_minus
            + input_derivatives * ((1.0 - root) ** 2)
        )
        deriv_num = jnp.maximum(deriv_num, _EPS)

        logabsdet = jnp.log(deriv_num) - 2.0 * jnp.log(denom)
        return outputs, -logabsdet  # inverse logdet sign

    else:
        theta = (inputs - input_cumwidths) / (input_bin_widths + _EPS)
        theta = jnp.clip(theta, 0.0, 1.0)
        theta_one_minus = theta * (1.0 - theta)

        numer = input_heights * (
            input_delta * (theta ** 2) + input_derivatives * theta_one_minus
        )
        denom = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus
        )
        denom = jnp.maximum(denom, _EPS)

        outputs = input_cumheights + numer / denom

        deriv_num = (input_delta ** 2) * (
            input_derivatives_plus_one * (theta ** 2)
            + 2.0 * input_delta * theta_one_minus
            + input_derivatives * ((1.0 - theta) ** 2)
        )
        deriv_num = jnp.maximum(deriv_num, _EPS)

        logabsdet = jnp.log(deriv_num) - 2.0 * jnp.log(denom)
        return outputs, logabsdet

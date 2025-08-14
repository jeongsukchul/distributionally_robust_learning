# ---------- FlowRQS: Flax linen.Module (AR RQS flow with MADE masks) ----------
from __future__ import annotations
from typing import List, Tuple, Dict
import dataclasses
import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from brax.training.networks import FeedForwardNetwork
# ----- config -----
@dataclasses.dataclass
class FlowRQSConfig:
  n_dim: int
  n_bins: int = 8
  hidden_dims: Tuple[int, ...] = (64, 64)
  activation: str = "gelu"
  n_transforms: int = 3
  scale: float = 10.0               # normalized support ~ [-scale/2, +scale/2]
  min_bin_width: float = 1e-3
  min_bin_height: float = 1e-3
  min_knot_slope: float = 1e-3
  # Curriculum learning parameters for gradual transformation
  use_curriculum: bool = True
  curriculum_steps: int = 10000     # Number of steps to gradually increase complexity
  curriculum_start_step: int = 0    # Step to start curriculum (for resuming training)

# ----- small utils -----
def _act(name: str):
  return {"relu": nn.relu, "tanh": jnp.tanh, "silu": nn.silu, "gelu": nn.gelu}.get(name, nn.gelu)

def _normalize(x, low, high, scale):
  mid = (low + high) / 2.0
  rng = (high - low)
  return (x - mid) / rng * scale

def _denormalize(xn, low, high, scale):
  mid = (low + high) / 2.0
  rng = (high - low)
  return rng * (xn / scale) + mid

# ----- masked dense -----
class MaskedDense(nn.Module):
  features: int
  mask: jnp.ndarray  # shape [features, in_features]
  use_bias: bool = True
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_uniform()

  @nn.compact
  def __call__(self, x):
    k = self.param("kernel", self.kernel_init, (self.features, x.shape[-1]))
    y = jnp.dot(x, (k * self.mask).T)
    if self.use_bias:
      b = self.param("bias", nn.initializers.zeros, (self.features,))
      y = y + b
    return y

# ----- MADE that outputs per-dim RQS params [w_K, h_K, s_{K+1}] -----
class MADE_RQS(nn.Module):
  n_dim: int
  n_bins: int
  hidden_dims: Tuple[int, ...]
  activation: str = "gelu"

  def setup(self):
    D = self.n_dim
    act = _act(self.activation)

    # degrees
    deg_in = jnp.arange(1, D + 1)
    hidden_degs = [ (jnp.arange(H) % (D - 1)) + 1 for H in self.hidden_dims ]
    last_deg = hidden_degs[-1] if hidden_degs else deg_in

    # masks
    in_masks = [ (hd[:, None] >= deg_in[None, :]).astype(jnp.float32) for hd in hidden_degs[:1] ]
    for l in range(1, len(hidden_degs)):
      in_masks.append( (hidden_degs[l][:, None] >= hidden_degs[l-1][None, :]).astype(jnp.float32) )
    out_mask = (jnp.arange(1, D + 1)[:, None] > last_deg[None, :]).astype(jnp.float32)  # [D, H_last]

    # layers
    layers = []
    if len(self.hidden_dims) > 0:
      # input -> first hidden
      layers.append(MaskedDense(self.hidden_dims[0], mask=in_masks[0]))
      # hidden -> hidden
      for l in range(1, len(self.hidden_dims)):
        layers.append(MaskedDense(self.hidden_dims[l], mask=in_masks[l]))
    self.layers = layers
    self.act = act
    self.out_mask = out_mask  # used in __call__

    # final linear (masked) will be applied by building a big masked weight on the fly

  @nn.compact
  def __call__(self, y_prefix: jnp.ndarray) -> jnp.ndarray:
    """y_prefix: [B, D] -> [B, D, 3K+1]"""
    x = y_prefix
    for layer in self.layers:
      x = self.act(layer(x))

    # last hidden -> output: [B, D * per_dim], with per-dim mask duplication
    D = self.n_dim
    K = self.n_bins
    per_dim = 3 * K + 1
    H_last = x.shape[-1]

    # Parameterize an unmasked kernel, then mask per output block:
    W = self.param("W_out", nn.initializers.lecun_uniform(), (D * per_dim, H_last))
    b = self.param("b_out", nn.initializers.zeros, (D * per_dim,))
    per_mask = jnp.repeat(self.out_mask, repeats=per_dim, axis=0)  # [D*per_dim, H_last]
    out = x @ (W.T * per_mask.T) + b
    return out.reshape(x.shape[0], D, per_dim)

# ----- single AR RQS transform (forward/inverse passes) -----
class AR_RQS_Transform(nn.Module):
  cfg: FlowRQSConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, forward: bool, training_step: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """If forward: x->y, else inverse: y->x. Returns (out, logdet)."""
    made = MADE_RQS(self.cfg.n_dim, self.cfg.n_bins, self.cfg.hidden_dims, self.cfg.activation)
    B, D = x.shape
    out = jnp.zeros_like(x)
    ldj = jnp.zeros((B,))

    def rqs_from_params(params_1d, training_step: int = 0):
      K = self.cfg.n_bins
      w_raw = params_1d[..., :K]
      h_raw = params_1d[..., K:2*K]
      s_raw = params_1d[..., 2*K:3*K+1]
      
      # Check if this is initialization (all zeros) to create uniform distribution
      is_init = jnp.allclose(params_1d, 0.0)
      
      if is_init:
        # For uniform distribution: equal bin widths, equal bin heights, constant slopes
        range_size = self.cfg.scale
        bin_width = range_size / K
        bin_height = range_size / K
        slope = 1.0
        
        # Create uniform parameters
        bin_w = jnp.full((K,), bin_width)
        bin_h = jnp.full((K,), bin_height)
        slopes = jnp.full((K+1,), slope)
      else:
        # Normal parameter processing
        bin_w = nn.softmax(w_raw, axis=-1) * (1.0 - K * self.cfg.min_bin_width) + self.cfg.min_bin_width
        bin_h = nn.softmax(h_raw, axis=-1) * (1.0 - K * self.cfg.min_bin_height) + self.cfg.min_bin_height
        slopes = nn.softplus(s_raw) + self.cfg.min_knot_slope
        
        # Apply curriculum learning for gradual transformation
        if self.cfg.use_curriculum:
          # Calculate curriculum progress (0.0 to 1.0)
          curriculum_progress = jnp.clip(
              (training_step - self.cfg.curriculum_start_step) / self.cfg.curriculum_steps, 
              0.0, 1.0
          )
          
          # Create uniform parameters for interpolation
          range_size = self.cfg.scale
          uniform_bin_width = range_size / K
          uniform_bin_height = range_size / K
          uniform_slope = 1.0
          
          uniform_bin_w = jnp.full_like(bin_w, uniform_bin_width)
          uniform_bin_h = jnp.full_like(bin_h, uniform_bin_height)
          uniform_slopes = jnp.full_like(slopes, uniform_slope)
          
          # Interpolate between uniform and learned parameters
          alpha = curriculum_progress
          bin_w = (1 - alpha) * uniform_bin_w + alpha * bin_w
          bin_h = (1 - alpha) * uniform_bin_h + alpha * bin_h
          slopes = (1 - alpha) * uniform_slopes + alpha * slopes
      
      rmin = -self.cfg.scale / 2.0
      rmax = +self.cfg.scale / 2.0
      
      # Concatenate all parameters into a single array as expected by RationalQuadraticSpline
      params = jnp.concatenate([bin_w, bin_h, slopes], axis=-1)
      return distrax.RationalQuadraticSpline(
          params=params,
          range_min=rmin, 
          range_max=rmax,
          min_bin_size=min(self.cfg.min_bin_width, self.cfg.min_bin_height),
          min_knot_slope=self.cfg.min_knot_slope
      )

    def body(i, carry):
      out, ldj = carry
      params = made(out if forward else out)  # prefix is 'out' either way
      spline = rqs_from_params(params[:, i, :], training_step)
      if forward:
        yi, ld = spline.forward_and_log_det(x[:, i])
      else:
        xi, ld = spline.inverse_and_log_det(x[:, i])
        yi = xi
      out = out.at[:, i].set(yi)
      ldj = ldj + ld
      return (out, ldj)

    out, ldj = jax.lax.fori_loop(0, D, body, (out, ldj))
    return out, ldj

# ----- full stack of transforms + base -----
class FlowRQS(nn.Module):
  cfg: FlowRQSConfig

  def setup(self):
    self.transforms = [AR_RQS_Transform(self.cfg) for _ in range(self.cfg.n_transforms)]

  @nn.compact
  def __call__(self, x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray, training_step: int = 0) -> jnp.ndarray:
    """Returns log_prob(x)."""
    xn = _normalize(x, low, high, self.cfg.scale)

    # forward through chain (last-defined applies first)
    z = xn
    ldj = jnp.zeros(x.shape[0])
    for tf in reversed(self.transforms):
      z, ld = tf(z, forward=True, training_step=training_step)
      ldj = ldj + ld

    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.cfg.n_dim), scale_diag=jnp.ones(self.cfg.n_dim))
    return base.log_prob(z) + ldj

  def sample(self, n_samples: int, low: jnp.ndarray, high: jnp.ndarray, rng: jax.random.PRNGKey, training_step: int = 0) -> jnp.ndarray:
    """Box-bounded sampling with rejection in original space."""
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.cfg.n_dim), scale_diag=jnp.ones(self.cfg.n_dim))

    def inverse_stack(y):
      x = y
      ldj = jnp.zeros(y.shape[0])
      for tf in self.transforms:          # inverse order of forward chain
        x, ld = tf(x, forward=False, training_step=training_step)
        ldj = ldj + ld
      return x, ldj

    def body(carry):
      rng, acc_x, acc_mask = carry
      rng, sub = jax.random.split(rng)
      needed = jnp.maximum(0, n_samples - acc_mask.sum()).astype(jnp.int32)
      num = jnp.maximum(needed, 1)

      z = base.sample(seed=sub, sample_shape=(num,))
      xn, _ = inverse_stack(z)
      x = _denormalize(xn, low, high, self.cfg.scale)
      ok = jnp.logical_and(x >= low, x <= high).all(axis=-1)

      pad = jnp.maximum(0, n_samples - x.shape[0])
      x = jnp.pad(x, ((0, pad), (0, 0)))
      ok = jnp.pad(ok, ((0, pad),))

      slots = jnp.where(~acc_mask, size=n_samples, fill_value=0)[0]
      picks = jnp.where(ok,        size=n_samples, fill_value=0)[0]
      take = jnp.minimum(slots.shape[0], picks.shape[0])

      acc_x = acc_x.at[slots[:take]].set(x[picks[:take]])
      acc_mask = acc_mask.at[slots[:take]].set(True)
      return (rng, acc_x, acc_mask)

    init_x = jnp.zeros((n_samples, self.cfg.n_dim))
    init_mask = jnp.zeros((n_samples,), dtype=bool)
    carry = (rng, init_x, init_mask)
    for _ in range(10):                   # same spirit as your torch code
      carry = body(carry)
    _, samples, _ = carry
    return samples

# ---------- FeedForwardNetwork factory (init/apply) ----------
def make_flow_network(cfg: FlowRQSConfig) -> FeedForwardNetwork:
  """
  Returns a FeedForwardNetwork:
    init(key) -> params
    apply(processor_params, params, mode, low, high, x=None, rng=None, n_samples=None)
  """
  module = FlowRQS(cfg)

  # Dummy shapes for init
  dummy_x   = jnp.zeros((1, cfg.n_dim))
  dummy_low = jnp.zeros((cfg.n_dim,))
  dummy_high= jnp.ones((cfg.n_dim,))

  def init_fn(key):
    variables = module.init(key, dummy_x, low=dummy_low, high=dummy_high)
    return variables["params"]

  def apply_fn(params, mode: str,
               low: jnp.ndarray, high: jnp.ndarray,
               x: jnp.ndarray = None,
               rng: jax.random.PRNGKey = None,
               n_samples: int = None,
               training_step: int = 0):
    if mode == "log_prob":
      if x is None:
        raise ValueError("mode='log_prob' requires x")
      return module.apply({"params": params}, x, low=low, high=high, training_step=training_step)
    elif mode == "sample":
      if rng is None or n_samples is None:
        raise ValueError("mode='sample' requires rng and n_samples")
      return module.apply({"params": params}, method=FlowRQS.sample,
                          n_samples=int(n_samples), low=low, high=high, rng=rng, training_step=training_step)
    else:
      raise ValueError(f"Unknown mode: {mode}")

  return FeedForwardNetwork(init=init_fn, apply=apply_fn)

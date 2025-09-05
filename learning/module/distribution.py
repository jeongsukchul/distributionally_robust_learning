# ---------- FlowRQS: Flax linen.Module (AR RQS flow with MADE masks) ----------
from __future__ import annotations
from typing import List, Tuple, Dict
import dataclasses
import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from brax.training.networks import FeedForwardNetwork

import numpy as np
import matplotlib.pyplot as plt
import wandb
import math
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

class MaskedDense(nn.Module):
    features: int
    mask: jnp.ndarray  # [in, out] or [out, in] depending on convention

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, in]
        in_dim = x.shape[-1]
        out_dim = self.features
        # weight: [in, out]
        w = self.param("kernel", nn.initializers.lecun_uniform(), (in_dim, out_dim))
        b = self.param("bias", nn.initializers.zeros, (out_dim,))
        # IMPORTANT: mask must be a constant array with same shape as w
        y = x @ (w * self.mask) + b
        return y

# ----- MADE that outputs per-dim RQS params [w_K, h_K, s_{K+1}] -----
class MADE_RQS(nn.Module):
    n_dim: int
    n_bins: int
    hidden_dims: Tuple[int, ...]
    activation: str = "gelu"

    def setup(self):
        D = self.n_dim
        self.act = _act(self.activation)

        # degrees (static)
        deg_in = jnp.arange(1, D + 1)
        hidden_degs = [(jnp.arange(H) % (D - 1)) + 1 for H in self.hidden_dims]
        last_deg = hidden_degs[-1] if hidden_degs else deg_in

        # ----- masks for hidden layers, shape [in_dim, out_dim] -----
        in_masks = []
        if hidden_degs:
            m0 = (hidden_degs[0][:, None] >= deg_in[None, :]).astype(jnp.float32).T  # [D, H1]
            in_masks.append(m0)
            for l in range(1, len(hidden_degs)):
                m = (hidden_degs[l][:, None] >= hidden_degs[l-1][None, :]).astype(jnp.float32).T  # [Hl-1, Hl]
                in_masks.append(m)

        # store as plain attributes (no variable collections)
        self.in_masks = tuple(in_masks)  # tuple of jnp arrays
        self.out_mask = (jnp.arange(1, D + 1)[:, None] > last_deg[None, :]).astype(jnp.float32)  # [D, H_last]

        # Build hidden layers with baked-in constant masks
        layers = []
        if self.hidden_dims:
            layers.append(MaskedDense(self.hidden_dims[0], mask=self.in_masks[0]))
            for l in range(1, len(self.hidden_dims)):
                layers.append(MaskedDense(self.hidden_dims[l], mask=self.in_masks[l]))
        self.layers = tuple(layers)

    @nn.compact
    def __call__(self, y_prefix: jnp.ndarray) -> jnp.ndarray:
        x = y_prefix
        for layer in self.layers:
            x = self.act(layer(x))

        D = self.n_dim
        K = self.n_bins
        per_dim = 3 * K + 1
        H_last = x.shape[-1]

        W = self.param("W_out", nn.initializers.lecun_uniform(), (D * per_dim, H_last))
        b = self.param("b_out", nn.initializers.zeros, (D * per_dim,))

        # use static out_mask attribute
        per_mask = jnp.repeat(self.out_mask, repeats=per_dim, axis=0).astype(x.dtype)  # [D*per_dim, H_last]
        proj = x @ (W.T * per_mask.T) + b  # [B, D*per_dim]
        return proj.reshape(x.shape[0], D, per_dim)

class AR_RQS_Transform(nn.Module):
  cfg: FlowRQSConfig

  def setup(self):
    """Define submodules here. This is run once when the module is initialized."""
    self.made = MADE_RQS(
        self.cfg.n_dim, self.cfg.n_bins, self.cfg.hidden_dims, self.cfg.activation
    )

  def __call__(self, x: jnp.ndarray, forward: bool, training_step: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    B, D = x.shape

    def rqs_from_params(params_1d, training_step: int = 0):
      K = self.cfg.n_bins
      w_raw = params_1d[..., :K]
      h_raw = params_1d[..., K:2*K]
      s_raw = params_1d[..., 2*K:3*K+1]

      # Use curriculum learning to anneal from a simple identity-like transform
      # to the full complex spline transform. This helps stabilize early training.
      range_size = self.cfg.scale
      uniform_bin_w = jnp.full_like(w_raw, range_size / K)
      uniform_bin_h = jnp.full_like(h_raw, range_size / K)
      uniform_slopes = jnp.ones_like(s_raw)

      learned_bin_w = nn.softmax(w_raw, axis=-1) * range_size * (1.0 - K * self.cfg.min_bin_width) + self.cfg.min_bin_width
      learned_bin_h = nn.softmax(h_raw, axis=-1) * range_size * (1.0 - K * self.cfg.min_bin_height) + self.cfg.min_bin_height
      learned_slopes = nn.softplus(s_raw) + self.cfg.min_knot_slope

      alpha = jnp.asarray(1.0)
      if self.cfg.use_curriculum:
          alpha = jnp.clip(
              (jnp.asarray(training_step, dtype=jnp.float32) - self.cfg.curriculum_start_step) / self.cfg.curriculum_steps, 0.0, 1.0
          )
      
      bin_w = (1.0 - alpha) * uniform_bin_w + alpha * learned_bin_w
      bin_h = (1.0 - alpha) * uniform_bin_h + alpha * learned_bin_h
      slopes = (1.0 - alpha) * uniform_slopes + alpha * learned_slopes
      
      # The total width/height must sum to the range size for the transform to be valid.
      # We normalize here to enforce this constraint strictly.
      bin_w = bin_w / jnp.sum(bin_w, axis=-1, keepdims=True) * range_size
      bin_h = bin_h / jnp.sum(bin_h, axis=-1, keepdims=True) * range_size

      params = jnp.concatenate([bin_w, bin_h, slopes], axis=-1)
      return distrax.RationalQuadraticSpline(
          params=params,
          range_min=-self.cfg.scale / 2.0,
          range_max=+self.cfg.scale / 2.0
      )

    if forward:
        # --- FORWARD PASS (x -> z) ---
        # This direction is parallel. We can compute all parameters and apply
        # the transformation for all dimensions at once using vmap.
        all_params = self.made(x) # Shape: [B, D, 3K+1]

        # Define the function to apply to a single dimension
        def apply_spline_1d(params_1d, x_1d):
            # params_1d: [B, 3K+1], x_1d: [B]
            spline = rqs_from_params(params_1d, training_step)
            return spline.forward_and_log_det(x_1d)

        # Vmap this function over the dimension axis (axis=1)
        z, ldj_per_dim = jax.vmap(
            apply_spline_1d, in_axes=1, out_axes=1
        )(all_params, x)

        # Sum log-determinants across dimensions
        ldj = jnp.sum(ldj_per_dim, axis=1)
        return z, ldj

    else: # Inverse pass
        # --- INVERSE PASS (z -> x) ---
        # This direction is sequential. We must compute x_i based on x_0..x_{i-1}.
        # We use a scan loop over the dimensions.
        z = x  # The input to this function is `z` in the inverse case
        x_init = jnp.zeros_like(z)
        ldj_init = jnp.zeros(B)

        def body(carry, i):
            x_so_far, ldj = carry
            # Get parameters, which depend on the x values computed so far
            params_all_dims = self.made(x_so_far)
            params_i = params_all_dims[:, i, :]
            
            spline = rqs_from_params(params_i, training_step)
            
            # Use the input z_i to compute the output x_i
            z_i = z[:, i]
            x_i, ld = spline.inverse_and_log_det(z_i)
            
            # Update the output tensor and the log determinant
            x_new = x_so_far.at[:, i].set(x_i)
            ldj_new = ldj + ld
            return (x_new, ldj_new), None

        (x_final, ldj), _ = jax.lax.scan(body, (x_init, ldj_init), xs=jnp.arange(D))
        return x_final, ldj

# ----- full stack of transforms + base -----
class FlowRQS(nn.Module):
  cfg: FlowRQSConfig

  def setup(self):
    self.transforms = tuple(AR_RQS_Transform(self.cfg) for _ in range(self.cfg.n_transforms))


  @nn.compact
  def __call__(self, x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray, training_step: int = 0) -> jnp.ndarray:
    xn = _normalize(x, low, high, self.cfg.scale)
    z = xn
    ldj = jnp.zeros(x.shape[0])
    for tf in self.transforms:
      z, ld = tf(z, forward=True, training_step=training_step)
      ldj = ldj + ld
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.cfg.n_dim), scale_diag=jnp.ones(self.cfg.n_dim))
    logp_z = base.log_prob(z)
    ldj_norm = jnp.sum(jnp.log(self.cfg.scale / (high - low)))
    return logp_z + ldj + ldj_norm

  def sample(self, n_samples, low, high, rng, training_step: int = 0):
    base = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(self.cfg.n_dim), scale_diag=jnp.ones(self.cfg.n_dim)
    )

    def inverse_stack(y):
      x = y
      ldj = jnp.zeros(y.shape[0])
      # Inverse should be applied in REVERSE order
      for tf in reversed(self.transforms):
        x, ld = tf(x, forward=False, training_step=training_step)
        ldj = ldj + ld
      return x, ldj

    z = base.sample(seed=rng, sample_shape=(int(n_samples),))
    xn, _ = inverse_stack(z)
    x = _denormalize(xn, low, high, self.cfg.scale)
    return jnp.clip(x, low, high)

def make_flow_network(cfg: FlowRQSConfig) -> FeedForwardNetwork:
  module = FlowRQS(cfg)

  dummy_x   = jnp.zeros((1, cfg.n_dim))

  def init_fn(key):
    dummy_low = jnp.zeros((cfg.n_dim,), dtype=jnp.float32)
    dummy_high = jnp.ones((cfg.n_dim,), dtype=jnp.float32)
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

#---------------for plotting --------------------

# If you're logging to W&B, keep this; otherwise you can drop wandb bits.

# -------- deterministic 1D PDF slices without an explicit prior --------
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def _ensure_f32(x):
    return jnp.asarray(x, dtype=jnp.float32)

def _unreplicate_params(params):
    return jax.tree.map(
        lambda a: (a[0] if hasattr(a, "shape") and a.ndim > 0 and a.shape[0] == jax.local_device_count() else a),
        params
    )

def _flow_logpdf_batch(flow_network, params, low, high, x_batch, training_step: int):
    """
    x_batch: (B, D)
    Returns: (B,) log-pdf via your FlowRQS.apply(mode='log_prob')
    """
    return flow_network.apply(
        params,
        mode="log_prob",
        x=x_batch,
        low=low,
        high=high,
        training_step=training_step
    )

def flow_pdf_1d_linspace(flow_network,
                         params,
                         low,              # (D,)
                         high,             # (D,)
                         cond_vec,         # (D,)
                         dim: int,
                         num: int = 400,
                         training_step: int = 0):
    """
    Returns xs (num,), pdf (num,) for a 1D slice along `dim`,
    where only x_dim sweeps linearly from low[dim] to high[dim],
    and other dims are fixed at cond_vec.
    """
    low  = _ensure_f32(low)
    high = _ensure_f32(high)
    cond = _ensure_f32(cond_vec)
    D = int(low.shape[0])

    xs = jnp.linspace(low[dim], high[dim], num, dtype=jnp.float32)
    # Build a batch where all dims are fixed to cond, except `dim`
    x_batch = jnp.broadcast_to(cond, (num, D)).at[:, dim].set(xs)

    logpdf = _flow_logpdf_batch(flow_network, params, low, high, x_batch, training_step)
    pdf = jnp.exp(logpdf)
    return xs, pdf

def render_flow_all_dims_1d_linspace(flow_network,
                                     flow_params,
                                     low,
                                     high,
                                     training_step: int = 0,
                                     cond_mode: str = "mid",  # "mid" or "zeros"
                                     num: int = 400,
                                     cols: int = 4,
                                     tag_prefix: str = "flow_pdf",
                                     wandb_step=None,
                                     use_wandb: bool = False):
    """
    Plots deterministic 1D PDF slices for *every* dimension, using a linspace grid per dim.
    """
    params_host = _unreplicate_params(flow_params)

    low  = _ensure_f32(low)
    high = _ensure_f32(high)
    D = int(low.shape[0])

    if cond_mode == "mid":
        cond = (low + high) / 2.0
    elif cond_mode == "zeros":
        cond = jnp.zeros((D,), dtype=jnp.float32)
    else:
        raise ValueError("cond_mode must be 'mid' or 'zeros'.")

    rows = math.ceil(D / cols)
    fig = plt.figure(figsize=(3.2 * cols, 2.4 * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, cols)

    for d in range(D):
        r, c = divmod(d, cols)
        ax = fig.add_subplot(gs[r, c])

        xs, pdf = flow_pdf_1d_linspace(
            flow_network, params_host, low, high, cond, dim=d, num=num, training_step=training_step
        )
        ax.plot(xs, pdf)
        ax.set_title(f"dim {d}")
        ax.set_xlabel("x")
        ax.set_ylabel("pdf")

    # Hide any empty cells
    for k in range(D, rows * cols):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.suptitle(f"Flow PDF (1D linspace slices across all dims) @ step {int(training_step)}")

    if use_wandb:
        import wandb
        wandb.log({f"{tag_prefix}/{wandb_step}/all_dims_1d_pdf": wandb.Image(fig)}, step=wandb_step)

    return fig


def check_mass(flow_network, params, low, high, training_step=0, n_samples=100000):
  """
  Estimates the total probability mass using Monte Carlo integration.
  This is suitable for high-dimensional spaces where grid-based methods fail.
  """
  low = jnp.asarray(low, dtype=jnp.float32)
  high = jnp.asarray(high, dtype=jnp.float32)
  D = low.shape[0]

  # 1. Create a PRNG key
  key = jax.random.PRNGKey(0)

  # 2. Draw random samples from a uniform distribution over the domain [low, high]
  #    This serves as our sampling distribution for the integration.
  print("n samples ", n_samples)
  print(" D " ,D)
  samples = jax.random.uniform(key, shape=(n_samples, D), minval=low, maxval=high)
  print('samples', samples.shape)
  # 3. Evaluate the log-probability of these samples under your flow model
  log_p = flow_network.apply(
      params,
      mode="log_prob",
      x=samples,
      low=low,
      high=high,
      training_step=training_step
  )
  p = jnp.exp(log_p)

  # 4. Calculate the volume of the integration domain
  volume = jnp.prod(high - low)

  # 5. Estimate the integral: E[p(x)] * Volume
  mass = jnp.mean(p) * volume
  return float(mass)
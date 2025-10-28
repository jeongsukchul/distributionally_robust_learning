import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import chex
from typing import List
import wandb
import numpy as np
def reduce_weighted_logsumexp(logx, w=None, axis=None, keep_dims=False, return_sign=False,):
    if w is None:
      lswe = jax.nn.logsumexp(
          logx,
          axis=axis,
          keepdims=keep_dims)

      if return_sign:
        sgn = jnp.ones_like(lswe)
        return lswe, sgn
      return lswe

    log_absw_x = logx + jnp.log(jnp.abs(w))
    max_log_absw_x = jnp.max(
        log_absw_x, axis=axis, keepdims=True,)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = jnp.where(
        jnp.isinf(max_log_absw_x),
        jnp.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (jnp.sign(w) * jnp.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = jnp.sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = jnp.squeeze(max_log_absw_x, axis)
    sgn = jnp.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + jnp.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe

def visualise(log_prob_fn, dr_range_low:chex.Array, dr_range_high : chex.Array, samples: chex.Array = None, show=False) -> dict:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    low, high = dr_range_low, dr_range_high
    x, y = jnp.meshgrid(jnp.linspace(low[0], high[0], 100), jnp.linspace(low[1], high[1], 100))
    grid = jnp.c_[x.ravel(), y.ravel()]
    pdf_values = jax.vmap(jnp.exp)(log_prob_fn(sample=grid))
    pdf_values = jnp.reshape(pdf_values, x.shape)
    ctf = plt.contourf(x, y, pdf_values, levels=20, cmap='viridis')
    cbar = fig.colorbar(ctf)
    if samples is not None:
        idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
        sample_x = samples[idx,0]
        sample_y = samples[idx,1]
        # sample_x = jnp.clip(samples[idx, 0],low[0], high[0])
        # sample_y = jnp.clip(samples[idx, 1],low[1], high[1])
        ax.scatter(sample_x, sample_y, c='r', alpha=0.5, marker='x')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.xticks([])
    plt.yticks([])
    # plt.xlim(-10, 5)
    # plt.ylim(-5, 5)

    # plt.savefig(os.path.join(project_path('./samples/funnel/'), f"{prefix}funnel.pdf"), bbox_inches='tight', pad_inches=0.1)

    wb = {"figures/model": [wandb.Image(fig)]}

    if show:
        plt.show()

    return wb

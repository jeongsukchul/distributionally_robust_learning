# Denoiser networks for diffusion.

import math
from typing import Optional

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    def setup(self, dim: int):
        self.dim = dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
    dim: int
    required: bool = True
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weights = self.param('weight', jax.random.normal, (self.dim//2,))
        weights = jax.lax.stop_gradient(weights) if self.required else weights
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(weights, 'd -> 1 d') * 2 * math.pi
        fouriered = jnp.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = jnp.cat((x, fouriered), dim=-1)
        return fouriered


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(nn.Module):
    dim_out : int = 50
    activation : callable = nn.relu
    layer_norm : bool = True
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.layer_norm:
            out = nn.LayerNorm()(x)
    
        return x + nn.Dense(self.dim_out)(self.activation(out))


class ResidualMLP(nn.Module):
    width: int
    depth: int
    output_dim: int
    activation: callable = nn.relu
    layer_norm: bool = True
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        network = nn.Sequential(
            nn.Dense(self.width),
            *[ResidualBlock(self.width, self.activation, self.layer_norm) for _ in range(self.depth)],
            nn.LayerNorm() if self.layer_norm else nn.identity(),
        )
        return nn.Dense(self.output_dim)(self.activation(network(x)))


@gin.configurable
class ResidualMLPDenoiser(nn.Module):
    d_in : int = 50
    dim_t : int = 128
    mlp_width : int = 1024
    num_layers : int = 6
    learned_sinusoidal_cond : bool = False
    random_fourier_features : bool = True
    learned_sinusoidal_dim : int = 16
    activation : callable = nn.relu
    layer_norm : bool = True
    cond_dim : Optional[int] = None
    def setup(
            self,
            d_in: int,
            dim_t: int = 128,
            mlp_width: int = 1024,
            num_layers: int = 6,
            learned_sinusoidal_cond: bool = False,
            random_fourier_features: bool = True,
            learned_sinusoidal_dim: int = 16,
            activation = nn.relu,
            layer_norm: bool = True,
            cond_dim: Optional[int] = None,
    ):
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray,
            timesteps: jnp.ndarray,
            cond=None,
    ) -> jnp.ndarray:
        residual_mlp = ResidualMLP(
            width=self.mlp_width,
            depth=self.num_layers,
            output_dim=self.d_in,
            activation=self.activation,
            layer_norm=self.layer_norm,
        )
        proj = nn.Dense(self.dim_t)
        if self.learned_sinusoidal_cond or self.random_fourier_features:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(self.learned_sinusoidal_dim, self.random_fourier_features)
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.dim_t)
        if self.cond_dim is not None:
            assert cond is not None
            x = jnp.cat((x, cond), dim=-1)
        time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Dense(self.dim_t),
            nn.silu(),
            nn.Dense(self.dim_t)
        )
        time_embed = time_mlp(timesteps)
        x = proj(x) + time_embed
        return residual_mlp(x)
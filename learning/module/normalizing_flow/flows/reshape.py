from typing import Literal, Tuple
import jax
import jax.numpy as jnp
from .base import Flow

class Split(Flow):

    split_mode : Literal['channel','channel_inv','checkerboard','checkerboard_inv'] = 'channel'


    def forward(self,z : Tuple[jnp.ndarray, jnp.ndarray]):
        if self.split_mode =="channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.split_mode =="channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        elif "checkerboard" in self.mode:
            n_dims = z.ndim
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = jnp.asarray(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            print("cb shape", cb.shape)
            z_size = z.size()
            z1 = z.reshape(-1)[jnp.nonzero(cb.reshape(-1))].reshape(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[jnp.nonzero((1 - cb).reshape(-1))].reshape(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det
    def inverse(self,z : Tuple[jnp.ndarray, jnp.ndarray]):
        z1, z2 = z 
        if self.mode =="channel":
            z = jnp.concat([z1,z2],1)
        elif self.mode =="channel_inv":
            z = jnp.concat([z2,z1],1)
        elif "checkerboard" in self.split_mode:
            n_dims = z.ndim
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = jnp.asarray(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det
class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """
    split_mode : Literal['channel','channel_inv','checkerboard','checkerboard_inv'] = 'channel'

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)

class Squeeze(Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """
    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.reshape(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute_dims(0, 1, 3, 5, 2, 4)
        z = z.reshape(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det

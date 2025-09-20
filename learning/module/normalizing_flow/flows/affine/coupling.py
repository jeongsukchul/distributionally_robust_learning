from typing import Callable, Literal, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from ..base import Flow, zero_log_det_like_z
from ..reshape import Split, Merge


class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """
    shape: Tuple[int, ...]
    use_scale : bool = True
    use_shift : bool = True
    def setup(self):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        if self.use_scale:
            self.s = self.param('s', lambda key : jnp.zeros(self.shape))
        if self.use_shift:
            self.t = self.param('t', lambda key: jnp.zeros(self.shape))
        self.n_dim = jnp.ndim(self.s)
        self.batch_dims = [i for i, d in enumerate(self.s.shape) if d == 1]
    def forward(self, z):
        z_ = z * jnp.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = jnp.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * jnp.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * jnp.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * jnp.sum(self.s)
        return z_, log_det


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """
    shape : Tuple[int, ...]
    num_classes : int 
    def setup(self, shape, num_classes):
        self.shape = shape
        self.s = self.param('s', lambda key : jnp.zeros(self.shape))
        self.t = self.param('t', lambda key : jnp.zeros(self.shape))
        self.s_cc = self.param('s_cc', lambda key: jnp.zeros(num_classes, np.prod(shape)))
        self.t_cc = self.param('t_cc', lambda key : jnp.zeros(num_classes, np.prod(shape)))
        self.n_dim = jnp.ndim(self.s)
        self.batch_dims = [i for i, d in enumerate(self.s.shape) if d == 1]

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).reshape(-1, *self.shape)
        t = self.t + (y @ self.t_cc).reshape(-1, *self.shape)
        z_ = z * jnp.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * jnp.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).reshape(-1, *self.shape)
        t = self.t + (y @ self.t_cc).reshape(-1, *self.shape)
        z_ = (z - t) * jnp.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * jnp.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """
    param_map_ctor: Callable[[], nn.Module]
    use_scale : bool = True
    scale_map : Literal["exp", "sigmoid", "sigmoid_inv"] = "exp"
    feature_last : bool = False
    @nn.compact
    def __call__(self,  z: Tuple[jnp.ndarray, jnp.ndarray], *, inverse: bool = False):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        z1, z2 = z
        param = self.param_map_ctor()(z1)
        if self.use_scale:
            shift = param[:,0::2, ...]
            scale = param[:, 1::2, ...]
            if self.scale_map=="exp":
                if not inverse:
                    y2 = z2 * jnp.exp(scale) + shift
                    log_det = jnp.sum(scale, axis=tuple(range(1, scale.ndim)))
                else:
                    y2 = (z2-shift) * jnp.exp(-scale)
                    log_det = -jnp.sum(scale, axis=tuple(range(1, scale.ndim)))
            elif self.scale_map=="sigmoid":
                scale_sig = jax.nn.sigmoid(scale + 2.0)
                if not inverse:
                    y2 = z2 / scale_sig + shift
                    log_det = -jnp.sum(jnp.log(scale_sig), axis=tuple(range(1, scale.ndim)))
                else:
                    y2 = (z2-shift) * scale_sig
                    log_det = jnp.sum(jnp.log(scale_sig), axis=tuple(range(1, scale.ndim)))
            elif self.scale_map=="sigmoid_inv":
                scale_sig = jax.nn.sigmoid(scale + 2.0)
                if not inverse:
                    y2 = z2 * scale_sig + shift
                    log_det = jnp.sum(jnp.log(scale_sig), axis=tuple(range(1,scale.ndim)))
                else:
                    y2 = (z2-shift)/scale_sig
                    log_det = -jnp.sum(jnp.log(scale_sig), axis=tuple(range(1, scale.ndim)))
            else:
                raise ValueError("This scale map is not implemented")
        else:
            y2 = z2 + shift if not inverse else z2 - shift
            log_det = jnp.zeros(z2.shape[:1], dtype=z2.dtype)
        
        return (z1,y2), log_det
     # Convenience wrappers
    def forward(self, z: Tuple[jnp.ndarray, jnp.ndarray]):
        return self(z, inverse=False)

    def inverse(self, z: Tuple[jnp.ndarray, jnp.ndarray]):
        return self(z, inverse=True)   


class MaskedAffineFlow(Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """
    b : jnp.ndarray
    t : Callable[[], nn.Module] = jnp.zeros_like
    s : Callable[[], nn.Module] = jnp.zeros_like
    @nn.compact
    def __call__(self, z : Tuple[jnp.ndarray, jnp.ndarray], inverse=False):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """

        z_masked = self.b * z
        scale = self.s(z_masked)
        trans = self.t(z_masked)

        if not inverse:
            z_ = z_masked + (1- self.b) * (z * jnp.exp(scale) + trans)
            log_det = jnp.sum((1-self.b)*scale, axis = list(tuple(range(1, scale.ndim))))
        else:
            z_ = z_masked + (1- self.b) * (z - trans) * jnp.exp(-scale)
            log_det = -jnp.sum((1-self.b)*scale, axis = list(tuple(range(1, scale.ndim))))
        return z_, log_det
    def forward(self, z):
        return self(z, inverse=False)

    def inverse(self, z):
        return self(z, inverse=True)

class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """
    flows : Sequence[nn.Module]
    param_map: Callable[[], nn.Module]
    scale_map : Literal["exp", "sigmoid", "sigmoid_inv"] = "exp"
    split_mode : Literal['channel','channel_inv','checkerboard','checkerboard_inv'] = 'channel'
    use_scale : bool =True
    @nn.compact
    def __call__(self, z):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        self.flows = []
        # Split layer
        self.flows += [Split(self.split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(self.param_map, self.use_scale, self.scale_map)]
        # Merge layer
        self.flows += [Merge(self.split_mode)]

    def forward(self, z):
        log_det_tot = jnp.zeros(z.shape[0], dtype=z.dtype)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = jnp.zeros(z.shape[0], dtype=z.dtype)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

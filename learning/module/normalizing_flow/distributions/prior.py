import numpy as np
import jax
import jax.numpy as jnp

class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        Args:
         z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        raise NotImplementedError


class TwoModes(PriorDistribution):
    def __init__(self, loc: float, scale :float):
        """Distribution 2d with two modes

        Distribution 2d with two modes at
        ```z[0] = -loc```  and ```z[0] = loc```
        following the density
        ```
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.loc = loc
        self.scale = scale

    def log_prob(self, z:jnp.ndarray)->jnp.ndarray:
        """

        ```
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        z = jnp.asarray(z)
        z0 = jnp.abs(z[:, 0])
        r = jnp.linalg.norm(z, axis=-1)
        eps = jnp.abs(jnp.asarray(self.loc))

        log_prob = (
            -0.5 * ((r- self.loc) / (2 * self.scale)) ** 2 \
            - 0.5 * ((z0 - eps) / (3 * self.scale)) ** 2  \
            + jnp.log(1 + jnp.exp(-2 * (z0 * eps) / (3 * self.scale) ** 2))
          )

        return log_prob

class Sinusoidal(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        log(p) = - 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        ```

        Args:
          scale: scale of the distribution, see formula
          period: period of the sinosoidal
        """
        self.scale = float(scale)
        self.period = float(period)

    def log_prob(self, z : jnp.ndarray)->jnp.ndarray:
        """

        ```
        log(p) = - 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        z = jnp.asarray(z)


        z0, z1 = z[..., 0], z[..., 1]
        w1 = jnp.sin(2.0 * jnp.pi / self.period * z0)
        w1 = jnp.sin(2.0 * jnp.pi / self.period * z0)

        term = -0.5 * ((z1 - w1) / (2.0 * self.scale)) ** 2
        envelope = -0.5 * (jnp.linalg.norm(z, axis=-1, ord=4) / (20.0 * self.scale)) ** 4
        return term + envelope

class Sinusoidal_gap(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density with gap
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        w_2(z) = 3 * exp(-0.5 * ((z[0] - 1) / 0.6) ** 2)
        log(p) = -log(exp(-0.5 * ((z[1] - w_1(z)) / 0.35) ** 2) + exp(-0.5 * ((z[1] - w_1(z) + w_2(z)) / 0.35) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w2_scale = 0.6
        self.w2_amp = 3.0
        self.w2_mu = 1.0

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        z0, z1 = z[..., 0], z[..., 1]

        w1 = jnp.sin(2.0 * jnp.pi / self.period * z0)
        w2 = self.w2_amp * jnp.exp(-0.5 * ((z0 - self.w2_mu) / self.w2_scale) ** 2)

        a = -0.5 * ((z1 - w1) / self.scale) ** 2
        b = -0.5 * ((z1 - w1 + w2) / self.scale) ** 2
        mix = -jax.nn.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)

        envelope = -0.5 * (jnp.linalg.norm(z, axis=-1, ord=4) / (20.0 * self.scale)) ** 4
        return mix + envelope


class Sinusoidal_split(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density with split
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        w_3(z) = 3 * sigmoid((z[0] - 1) / 0.3)
        log(p) = -log(exp(-0.5 * ((z[1] - w_1(z)) / 0.4) ** 2) + exp(-0.5 * ((z[1] - w_1(z) + w_3(z)) / 0.35) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w3_scale = 0.3
        self.w3_amp = 3.0
        self.w3_mu = 1.0


    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        z0, z1 = z[..., 0], z[..., 1]

        w1 = jnp.sin(2.0 * jnp.pi / self.period * z0)
        w3 = self.w3_amp * jax.nn.sigmoid((z0 - self.w3_mu) / self.w3_scale)

        a = -0.5 * ((z1 - w1) / self.scale) ** 2
        b = -0.5 * ((z1 - w1 + w3) / self.scale) ** 2
        mix = -jax.nn.logsumexp(jnp.stack([a, b], axis=-1), axis=-1)

        envelope = -0.5 * (jnp.linalg.norm(z, axis=-1, ord=4) / (20.0 * self.scale)) ** 4
        return mix + envelope



class Smiley(PriorDistribution):
    def __init__(self, scale):
        """Distribution 2d of a smiley :)

        Args:
          scale: scale of the smiley
        """
        self.scale = scale
        self.loc = 2.0

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        z1 = z[..., 1]
        ring = -0.5 * ((jnp.linalg.norm(z, axis=-1) - self.loc) / (2.0 * self.scale)) ** 2
        eyes = -0.5 * ((jnp.abs(z1 + 0.8) - 1.2) / (2.0 * self.scale)) ** 2
        return ring + eyes
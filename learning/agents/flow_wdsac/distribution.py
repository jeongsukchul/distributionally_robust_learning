# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX-based normalizing flow distributions for adversarial dynamics parameters."""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax

from brax.training import types
from brax.training.types import PRNGKey


class AffineCouplingFlow(nn.Module):
    """Affine coupling layer for normalizing flows."""
    
    features: int
    hidden_features: int = 64
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affine coupling transformation."""
        batch_size = x.shape[0]
        half_features = self.features // 2
        
        # Split input into two halves
        x1, x2 = x[:, :half_features], x[:, half_features:]
        
        # Create scale and shift networks for x2
        scale_shift = nn.Sequential([
            nn.Dense(self.hidden_features),
            nn.relu,
            nn.Dense(self.hidden_features),
            nn.relu,
            nn.Dense(half_features * 2)  # scale and shift
        ])(x1)
        
        scale, shift = scale_shift[:, :half_features], scale_shift[:, half_features:]
        
        if reverse:
            # Inverse transformation
            x2 = (x2 - shift) * jnp.exp(-scale)
            log_det = -jnp.sum(scale, axis=-1)
        else:
            # Forward transformation
            x2 = x2 * jnp.exp(scale) + shift
            log_det = jnp.sum(scale, axis=-1)
        
        # Reconstruct full tensor
        x_transformed = jnp.concatenate([x1, x2], axis=-1)
        
        return x_transformed, log_det


class PlanarFlow(nn.Module):
    """Planar flow layer for normalizing flows."""
    
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply planar flow transformation."""
        batch_size = x.shape[0]
        
        # Learnable parameters
        w = self.param('w', nn.initializers.normal(0.02), (self.features,))
        u = self.param('u', nn.initializers.normal(0.02), (self.features,))
        b = self.param('b', nn.initializers.zeros, ())
        
        # Ensure invertibility
        wu = jnp.dot(w, u)
        m_wu = -1 + jax.nn.softplus(wu)
        u_hat = u + (m_wu - wu) * w / (jnp.dot(w, w) + 1e-8)
        
        # Apply transformation
        wx = jnp.dot(x, w)
        if reverse:
            # Inverse transformation (approximate)
            x = x - u_hat * jax.nn.tanh(wx + b)
            log_det = -jnp.log(1 + jax.nn.tanh(wx + b) ** 2)
        else:
            # Forward transformation
            x = x + u_hat * jax.nn.tanh(wx + b)
            log_det = jnp.log(1 + jax.nn.tanh(wx + b) ** 2)
        
        return x, log_det


class NormalizingFlow(nn.Module):
    """JAX-based normalizing flow for adversarial dynamics parameters."""
    
    features: int
    num_flows: int = 4
    hidden_features: int = 128
    flow_type: str = "affine"  # "affine" or "planar"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply normalizing flow transformation."""
        batch_size = x.shape[0]
        log_det_sum = jnp.zeros(batch_size)
        
        if reverse:
            # Apply flows in reverse order
            for i in range(self.num_flows - 1, -1, -1):
                if self.flow_type == "affine":
                    x, log_det = AffineCouplingFlow(
                        features=self.features,
                        hidden_features=self.hidden_features
                    )(x, reverse=True)
                else:  # planar
                    x, log_det = PlanarFlow(features=self.features)(x, reverse=True)
                log_det_sum += log_det
        else:
            # Apply flows in forward order
            for i in range(self.num_flows):
                if self.flow_type == "affine":
                    x, log_det = AffineCouplingFlow(
                        features=self.features,
                        hidden_features=self.hidden_features
                    )(x, reverse=False)
                else:  # planar
                    x, log_det = PlanarFlow(features=self.features)(x, reverse=False)
                log_det_sum += log_det
        
        return x, log_det_sum


class FlowDistribution:
    """JAX-based flow distribution for adversarial dynamics parameters."""
    
    def __init__(
        self,
        flow_network: NormalizingFlow,
        dynamics_config: Dict[str, Any],
        flow_params: Optional[types.Params] = None
    ):
        """Initialize flow distribution.
        
        Args:
            flow_network: The normalizing flow network
            dynamics_config: Configuration for dynamics parameters (required)
            flow_params: Pre-trained flow parameters (optional)
        """
        self.flow_network = flow_network
        self.flow_params = flow_params
        self.dynamics_config = dynamics_config
        
        # Calculate the total number of dynamics parameters     #
        self.features = (1 + dynamics_config['n_dof_friction'] + 
                        3 + dynamics_config['n_body_mass'])
    
    def sample(self, rng: PRNGKey, batch_size: int = 1) -> jnp.ndarray:
        """Sample adversarial dynamics parameters from the flow."""
        # Start with uniform distributions like in cheetah.py
        if rng.ndim == 0:
            rng = rng.reshape(1)
        
        # Split RNG for different parameter groups
        rngs = jax.random.split(rng, 4)
        
        # 1. Floor friction (1 parameter) - uniform distribution
        floor_friction = jax.random.uniform(
            rngs[0], 
            shape=(batch_size, 1),
            minval=self.dynamics_config['floor_friction_range'][0],
            maxval=self.dynamics_config['floor_friction_range'][1]
        )
        
        # 2. DOF friction (n_dof-6 parameters) - uniform distribution
        n_dof_friction = self.dynamics_config['n_dof_friction']
        dof_friction = jax.random.uniform(
            rngs[1],
            shape=(batch_size, n_dof_friction),
            minval=self.dynamics_config['dof_friction_range'][0],
            maxval=self.dynamics_config['dof_friction_range'][1]
        )
        
        # 3. COM offset (3 parameters) - uniform distribution
        com_offset = jax.random.uniform(
            rngs[2],
            shape=(batch_size, 3),
            minval=self.dynamics_config['com_offset_range'][0],
            maxval=self.dynamics_config['com_offset_range'][1]
        )
        
        # 4. Body mass multipliers (n_body-1 parameters) - uniform distribution
        n_body_mass = self.dynamics_config['n_body_mass']
        body_mass = jax.random.uniform(
            rngs[3],
            shape=(batch_size, n_body_mass),
            minval=self.dynamics_config['body_mass_range'][0],
            maxval=self.dynamics_config['body_mass_range'][1]
        )
        
        # Concatenate all parameters
        uniform_params = jnp.concatenate([
            floor_friction,    # shape: (batch_size, 1)
            dof_friction,      # shape: (batch_size, n_dof_friction)
            com_offset,        # shape: (batch_size, 3)
            body_mass          # shape: (batch_size, n_body_mass)
        ], axis=-1)
        
        # Verify the total parameter size matches our network
        total_params = 1 + n_dof_friction + 3 + n_body_mass
        assert uniform_params.shape[-1] == total_params, f"Expected {total_params} parameters, got {uniform_params.shape[-1]}"
        
        # Apply the normalizing flow to transform uniform parameters
        if self.flow_params is not None:
            transformed_params, _ = self.flow_network.apply(
                self.flow_params, uniform_params, reverse=False
            )
        else:
            # If no trained params, return uniform params
            transformed_params = uniform_params
        
        return transformed_params
    
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability of parameters under the flow."""
        if self.flow_params is None:
            # If no trained params, return uniform log prob
            return jnp.zeros(x.shape[0])
        
        # Transform back to uniform space
        uniform_x, log_det = self.flow_network.apply(
            self.flow_params, x, reverse=True
        )
        
        # Compute uniform log probability
        uniform_log_prob = self._uniform_log_prob(uniform_x)
        
        # Add log determinant correction
        return uniform_log_prob + log_det
    
    def _uniform_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability under uniform distribution."""
        batch_size = x.shape[0]
        log_prob = jnp.zeros(batch_size)
        
        # Floor friction
        floor_friction = x[:, 0:1]
        min_val, max_val = self.dynamics_config['floor_friction_range']
        log_prob += jnp.where(
            (floor_friction >= min_val) & (floor_friction <= max_val),
            -jnp.log(max_val - min_val),
            -jnp.inf
        ).sum(axis=-1)
        
        # DOF friction
        n_dof_friction = self.dynamics_config['n_dof_friction']
        dof_friction = x[:, 1:1+n_dof_friction]
        min_val, max_val = self.dynamics_config['dof_friction_range']
        log_prob += jnp.where(
            (dof_friction >= min_val) & (dof_friction <= max_val),
            -jnp.log(max_val - min_val),
            -jnp.inf
        ).sum(axis=-1)
        
        # COM offset
        com_offset = x[:, 1+n_dof_friction:1+n_dof_friction+3]
        min_val, max_val = self.dynamics_config['com_offset_range']
        log_prob += jnp.where(
            (com_offset >= min_val) & (com_offset <= max_val),
            -jnp.log(max_val - min_val),
            -jnp.inf
        ).sum(axis=-1)
        
        # Body mass
        n_body_mass = self.dynamics_config['n_body_mass']
        body_mass = x[:, 1+n_dof_friction+3:1+n_dof_friction+3+n_body_mass]
        min_val, max_val = self.dynamics_config['body_mass_range']
        log_prob += jnp.where(
            (body_mass >= min_val) & (body_mass <= max_val),
            -jnp.log(max_val - min_val),
            -jnp.inf
        ).sum(axis=-1)
        
        return log_prob


def make_flow_distribution(
    flow_network: NormalizingFlow,
    dynamics_config: Dict[str, Any],
    flow_params: Optional[types.Params] = None
) -> FlowDistribution:
    """Create a flow distribution for adversarial dynamics parameters.
    
    Args:
        flow_network: The normalizing flow network
        dynamics_config: Configuration for dynamics parameters (required)
        flow_params: Pre-trained flow parameters (optional)
    
    Returns:
        FlowDistribution instance
    """
    return FlowDistribution(
        flow_network=flow_network,
        dynamics_config=dynamics_config,
        flow_params=flow_params
    )


def create_flow_network(
    features: int,
    num_flows: int = 4,
    hidden_features: int = 128,
    flow_type: str = "affine"
) -> NormalizingFlow:
    """Create a normalizing flow network.
    
    Args:
        features: Number of input/output features
        num_flows: Number of flow layers
        hidden_features: Number of hidden features in coupling layers
        flow_type: Type of flow ("affine" or "planar")
    
    Returns:
        NormalizingFlow instance
    """
    return NormalizingFlow(
        features=features,
        num_flows=num_flows,
        hidden_features=hidden_features,
        flow_type=flow_type
    )
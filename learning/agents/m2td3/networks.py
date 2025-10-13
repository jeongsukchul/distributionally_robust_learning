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

"""m2td3 networks."""

from typing import Mapping, Sequence, Tuple

from brax.training import distribution
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax
@flax.struct.dataclass
class M2TD3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork


def make_inference_fn(m2td3_networks: M2TD3Networks):
  """Creates params and inference function for the m2td3 agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False, std_min: float = 0.05, std_max: float = 0.8 
  ) -> types.Policy:

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      return m2td3_networks.policy_network.apply(*params, observations), None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):  
        act = m2td3_networks.policy_network.apply(*params, observations)
        noise = jax.random.normal(key, shape=act.shape) * noise_scales[..., None]
        return act + noise, None
    if deterministic:
        return deterministic_policy
    else:
        return stochastic_policy

  return make_policy
def make_m2td3_q_network(
    obs_size: types.ObservationSize,
    action_size: int,
    param_size : int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
    obs_key: str = 'state',
):
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""

    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray,params:jnp.ndarray):
      hidden = jnp.concatenate([obs, actions, params], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = networks.MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            layer_norm=layer_norm,
        )(hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions, params):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[obs_key], networks.normalizer_select(processor_params, obs_key)
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions, params)
  obs_size = networks._get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  dummy_params = jnp.zeros((1,param_size))
  return networks.FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action, dummy_params), apply=apply
  )

def make_m2td3_networks(
    observation_size: int,
    action_size: int,
    param_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> M2TD3Networks:
  """Make m2td3 networks."""

  policy_network = networks.make_deterministic_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key = policy_obs_key
  )
  q_network = make_m2td3_q_network(
      observation_size,
      action_size,
      param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key = value_obs_key,
  )

  return M2TD3Networks(
      policy_network=policy_network,
      q_network=q_network,
      
  )
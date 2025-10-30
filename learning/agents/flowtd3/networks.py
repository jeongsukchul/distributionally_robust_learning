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

"""flowtd3 networks."""
from typing import Any, Sequence, Tuple

from brax.training import distribution
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax
from learning.module.normalizing_flow.simple_flow import make_realnvp_flow_networks
from learning.module.sampling.point_is_valid import default_point_is_valid_fn
from learning.module.sampling.hmc import build_blackjax_hmc
from learning.module.sampling.smc import build_smc
from omegaconf import OmegaConf

# hmc_cfg = OmegaConf({
#     'n_outer_steps': 1,
#     'n_inner_steps': 10,
#     'init_step_size': 0.1,
#     'target_p_accept': 0.8,
#     'tune_step_size': True,
# }
# )
# smc_cfg = OmegaConf({
#     'use_resampling': False,
#     'n_intermediate_distributions': 12,
#     'spacing_type': 'linear',
#     'transition_operator': 'hmc',
#     'point_is_valid_fn': default_point_is_valid_fn,
# })

@flax.struct.dataclass
class FlowTd3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  flow_network: networks.FeedForwardNetwork

@flax.struct.dataclass
class FabTd3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  flow_network: networks.FeedForwardNetwork
  smc: Any
def make_inference_fn(flowtd3_networks: FlowTd3Networks):
  """Creates params and inference function for the td3 agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False,
  ) -> types.Policy:

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      return flowtd3_networks.policy_network.apply(*params, observations), None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):  
        act = flowtd3_networks.policy_network.apply(*params, observations)
        noise = jax.random.normal(key, shape=act.shape) * noise_scales[..., None]
        return act + noise, None
    if deterministic:
        return deterministic_policy
    else:
        return stochastic_policy

  return make_policy
def make_flowtd3_networks(
    observation_size: int,
    action_size: int,
    dynamics_param_size : int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    fab_online : bool = False,
    distributional_q : bool = False,
    v_min : bool = False,
    num_atoms: int = 101,
    v_max : bool = False,
) -> FlowTd3Networks:
  """Make td3 networks."""

  policy_network = networks.make_deterministic_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=policy_network_layer_norm,
      obs_key = policy_obs_key
  )
  q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=q_network_layer_norm,
      obs_key = value_obs_key,
  )
  flow_network = make_realnvp_flow_networks(
    in_channels=dynamics_param_size)
  

  # if fab_online:
  #   transition_operator = build_blackjax_hmc(
  #       dim=dynamics_param_size,
  #       n_outer_steps=hmc_cfg.n_outer_steps,
  #       init_step_size=hmc_cfg.init_step_size,
  #       target_p_accept=hmc_cfg.target_p_accept,
  #       adapt_step_size=hmc_cfg.tune_step_size,
  #       n_inner_steps=hmc_cfg.n_inner_steps)
  #   smc =build_smc(transition_operator=transition_operator,
  #           n_intermediate_distributions=smc_cfg.n_intermediate_distributions,
  #           spacing_type=smc_cfg.spacing_type, alpha=2.0,
  #           use_resampling=smc_cfg.use_resampling, point_is_valid_fn=smc_cfg.point_is_valid_fn)
  #   return FabTd3Networks(
  #     policy_network=policy_network,
  #     q_network=q_network,
  #     flow_network=flow_network,
  #     smc=smc,
  # )

  return FlowTd3Networks(
      policy_network=policy_network,
      q_network=q_network,
      flow_network=flow_network,
  )
  
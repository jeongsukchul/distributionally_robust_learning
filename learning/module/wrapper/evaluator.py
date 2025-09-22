from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import functools
import time
import scipy
import numpy as np
from flax import struct
from brax import envs
@struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

  Attributes:
      episode_metrics: Aggregated episode metrics since the beginning of the
        episode.
      active_episodes: Boolean vector tracking which episodes are not done yet.
      episode_steps: Integer vector tracking the number of steps in the episode.
  """

  episode_metrics: Dict[str, jax.Array]
  active_episodes: jax.Array
  episode_steps: jax.Array


from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics, Transition
class AdvEvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jax.Array) -> State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(
            jnp.zeros_like, reset_state.metrics
        ),
        active_episodes=jnp.ones_like(reset_state.reward),
        episode_steps=jnp.zeros_like(reset_state.reward),
    )
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: State, action: jax.Array, params: jax.Array) -> State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}'
      )
    del state.info['eval_metrics']
    nstate = self.env.step(state, action, params)
    nstate.metrics['reward'] = nstate.reward
    episode_steps = jnp.where(
        state_metrics.active_episodes,
        nstate.info['steps'],
        state_metrics.episode_steps,
    )
    episode_metrics = jax.tree_util.tree_map(
        lambda a, b: a + b * state_metrics.active_episodes,
        state_metrics.episode_metrics,
        nstate.metrics,
    )
    active_episodes = state_metrics.active_episodes * (1 - nstate.done)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps,
    )
    nstate.info['eval_metrics'] = eval_metrics
    return nstate


class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: Dict[str, Any] = {}

def adv_step(
  env: Env,
  env_state: State,
  dynamics_params: jnp.ndarray,
  policy: Policy,
  key: PRNGKey,
  extra_fields: Sequence[str] = (),
):
  
  actions, policy_extras = policy(env_state.obs, key)

  nstate = env.step(env_state, actions, dynamics_params)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      dynamics_params=dynamics_params,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation= nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
      )

def generate_unroll(
    env: Env,
    env_state: State,
    dynamics_params :jnp.ndarray,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = adv_step(
        env, state, dynamics_params, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length
  )
  return final_state, data

# TODO: Consider moving this to its own file.
class Evaluator:
  def __init__(
      self,
      eval_env: Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0

    eval_env = envs.training.EvalWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll) #jax.jit
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      dynamics_params : jnp.ndarray,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)
    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, dynamics_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    iqm_fn = functools.partial(scipy.stats.trim_mean, proportiontocut=0.25, axis=None) 
    for fn in [np.mean, np.std, iqm_fn]:
      suffix = '_std' if fn == np.std else '_iqm' if fn == iqm_fn else ''
      metrics.update({
          f'eval/episode_{name}{suffix}': (
              fn(value) if aggregate_episodes else value
          )
          for name, value in eval_metrics.episode_metrics.items()
      })
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

class AdvEvaluator:
  """Class to run evaluations."""

  def __init__(
      self,
      eval_env: Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0

    eval_env = AdvEvalWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, eval_dynamics_params, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_dynamics_params,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll) #jax.jit
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      dynamics_params : jnp.ndarray,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)
    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, dynamics_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    iqm_fn = functools.partial(scipy.stats.trim_mean, proportiontocut=0.25, axis=None) 
    for fn in [np.mean, np.std, iqm_fn]:
      suffix = '_std' if fn == np.std else '_iqm' if fn == iqm_fn else ''
      metrics.update({
          f'eval/episode_{name}{suffix}': (
              fn(value) if aggregate_episodes else value
          )
          for name, value in eval_metrics.episode_metrics.items()
      })
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

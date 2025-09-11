from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import numpy as np
import functools
import time

class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: Dict[str, Any] = {}

def wrap_for_adv_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  env = AdVmapWrapper(env, randomization_fn)
  env = EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  return env
class AdVmapWrapper(Wrapper):
  """Wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[System], Tuple[System, System]],
  ):
    super().__init__(env)
    self.rand_fn = functools.partial(randomization_fn, model=self.mjx_model, rng=None)
  def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    state = jax.vmap(self.env.reset)(rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> State:
    #params shape is [n_envs, params,]
    # print("params in  adv wrapper", params)
    mjx_model_v, in_axes = self.rand_fn(params=params)
    # print("mjx_model_v shape", mjx_model_v)
    # print("in_axes shape", in_axes)
    def step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[in_axes, 0, 0])(
        mjx_model_v, state, action
    )
    return res
class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jnp.zeros(rng.shape[:-1])
    state.info['truncation'] = jnp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array, params: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action, params)
      return nstate, nstate.reward

    
    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    rewards = jnp.sum(rewards,axis=0)
    state = state.replace(reward=rewards)
    steps = state.info['steps'] + self.action_repeat
    one = jnp.ones_like(state.done)
    zero = jnp.zeros_like(state.done)
    episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
    done = jnp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jnp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps
    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += rewards
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)

class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array, params: jax.Array) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jnp.zeros_like(state.done))
    state = self.env.step(state, action, params)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jnp.where(done, x, y)

    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)
from flax import struct
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
class EvalWrapper(Wrapper):
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

    eval_env = EvalWrapper(eval_env)

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
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
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

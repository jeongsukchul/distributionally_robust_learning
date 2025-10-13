from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import functools

def wrap_for_hard_dr_training(
    env: mjx_env.MjxEnv,
    n_nominals : int,
    n_envs : int,
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
  env = RandomVmapWrapper(env, randomization_fn, n_nominals, n_envs)
  env = EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env)
  return env
class RandomVmapWrapper(Wrapper):
  """Wrapper for domain randomization."""
  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[System], Tuple[System, System]],
      n_nominals : int,
      n_envs : int,
  ):
    super().__init__(env)
    self.rand_fn = functools.partial(randomization_fn,model=self.mjx_model)
    self.n_nominals= n_nominals
    self.n_envs = n_envs
  def _env_fn(self,  mjx_model: mjx.Model) -> mjx_env.MjxEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    state = jax.vmap(self.env.reset)(rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array, key:jax.random.PRNGKey) -> State:
    keys = jax.random.split(key, self.n_nominals* self.n_envs)
    mjx_model_v, in_axes = self.rand_fn(rng=keys)
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

  def step(self, state: State, action: jax.Array, rng: jax.random.PRNGKey) -> State:
    dr_state= jax.tree_util.tree_map(lambda x : jnp.repeat(x, self.n_nominals, axis=0), state)
    dr_action = jnp.repeat(action, self.n_nominals, axis=0)
    def f(state, _):
      nstate = self.env.step(state, dr_action, rng)
      return nstate, nstate.reward

    
    state, rewards = jax.lax.scan(f, dr_state, (), self.action_repeat)
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

  def reset(self, rng: jax.random.PRNGKey) -> mjx_env.State:
    state = self.env.reset(rng)
    state.info['first_state'] = state.data
    state.info['first_obs'] = state.obs
    return state

  def step(self, state: mjx_env.State, action: jax.Array, rng: jax.random.PRNGKey=jax.random.PRNGKey(0)) -> mjx_env.State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jnp.zeros_like(state.done))
    state = self.env.step(state, action, rng)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jnp.where(done, x, y)

    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    return state.replace(data=data, obs=obs)

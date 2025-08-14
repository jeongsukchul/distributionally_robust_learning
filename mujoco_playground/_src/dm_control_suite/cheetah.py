# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Cheetah environment."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
from omegaconf import OmegaConf

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cheetah.xml"
# Running speed above which reward is 1.
_RUN_SPEED = 10


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
  )


class Run(mjx_env.MjxEnv):
  """Cheetah running environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

  def _post_init(self) -> None:
    self._lowers = self._mj_model.jnt_range[3:, 0]
    self._uppers = self._mj_model.jnt_range[3:, 1]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1 = jax.random.split(rng, 2)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[3:].set(
        jax.random.uniform(
            rng1,
            (self.mjx_model.nq - 3,),
            minval=self._lowers,
            maxval=self._uppers,
        )
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos)

    # Stabilize.
    data = mjx_env.step(self.mjx_model, data, jp.zeros(self.mjx_model.nu), 200)
    data = data.replace(time=0.0)

    metrics = {}
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    return jp.concatenate([
        data.qpos[1:],
        data.qvel,
    ])

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info, metrics  # Unused.
    speed = mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")[
        0
    ]  # x-axis only.
    return reward.tolerance(
        speed,
        bounds=(_RUN_SPEED, float("inf")),
        margin=_RUN_SPEED,
        value_at_margin=0,
        sigmoid="linear",
    )

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
  
  @property
  def dr_range(self) -> dict:

    low = jp.array(
        [0.5] +                             #floor_friction_min 
        [0.9] * (self.mjx_model.nv - 6) +   # dof_friction_min
        [-0.1] * 3 +                          #com_offset_min
        [0.8] * (self.mjx_model.nbody - 1)) #body_mass_min
    high = jp.array(
        [1.2] +                             #floor_friction_max
        [1.1] * (self.mjx_model.nv - 6) +   #dof_friction_max
        [0.1] * 3 +                          #com_offset_max
        [1.2] * (self.mjx_model.nbody - 1)) #body_mass_max
    return low, high
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1

def domain_randomize(model: mjx.Model, params, rng:jax.Array=None, deterministic=False):
  deterministic_cfg = None
  if not deterministic:
    dr_low, dr_high = params
    dist = [functools.partial(jax.random.uniform,minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))] 

  @jax.vmap
  def shift_dynamics(params):
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(params[idx:idx+ model.nv-6])
    idx += model.nv-6
    offset = jp.array(params[idx], params[idx+1], params[idx+2])
    idx += 3
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + offset
      )
    body_mass = jp.ones((model.nbody,))
    for i in range(1, model.nbody):
      body_mass = body_mass.at[i].set(model.body_mass[i] * params[idx])
      idx+=1
    assert idx == len(params)
    return (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    )
  @jax.vmap
  def rand_dynamics(rng):
    # floor friction
    rng, key = jax.random.split(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
      dist[idx](key=key)
    )
    idx+=1
    # static friction
    dof_frictionloss = jp.zeros((model.nv-6,))
    for i in range(model.nv-6):
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[6+i] * dist[idx](key=key)
      dof_frictionloss = model.dof_frictionloss.at[6+i].set(frictionloss)
      idx+=1
    # com pos offset
    dpos = jp.zeros((3,))
    for i in range(3):
      rng, key = jax.random.split(rng)
      dpos = dpos.at[idx].set(dist[idx](key=key))
      idx+=1
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )
    # link mass 
    body_mass = jp.ones((model.nbody,))
    for i in range(1, model.nbody):
      rng, key = jax.random.split(rng)
      dmass = dist[idx](key)
      body_mass = body_mass.at[i].set(model.body_mass[i] * dmass)
      idx+=1
    assert idx == len(dr_low)
    return (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    )
  
  if deterministic:
    (geom_friction, body_ipos, body_mass, dof_frictionloss) = shift_dynamics(params)
  else:
    (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "dof_frictionloss": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "dof_frictionloss": dof_frictionloss,
  })

  return model, in_axes

def domain_randomize_eval(model: mjx.Model, params, rng:jax.Array=None, deterministic=False):
  deterministic_cfg = None
  if not deterministic:
    dr_low, dr_high = params
    dist = [functools.partial(jax.random.uniform,minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))] 
  def shift_dynamics(params):
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(params[idx:idx+ model.nv-6])
    idx += model.nv-6
    offset = jp.array(params[idx], params[idx+1], params[idx+2])
    idx += 3
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + offset
      )
    body_mass = jp.ones((model.nbody,))
    for i in range(1, model.nbody):
      body_mass = body_mass.at[i].set(model.body_mass[i] * params[idx])
      idx+=1
    assert idx == len(params)
    return (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    )
  def rand_dynamics(rng):
    # floor friction
    rng, key = jax.random.split(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
      dist[idx](key=key)
    )
    idx+=1
    # static friction
    dof_frictionloss = jp.zeros((model.nv-6,))
    for i in range(model.nv-6):
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[6+i] * dist[idx](key=key)
      dof_frictionloss = model.dof_frictionloss.at[6+i].set(frictionloss)
      idx+=1
    # com pos offset
    dpos = jp.zeros((3,))
    for i in range(3):
      rng, key = jax.random.split(rng)
      dpos = dpos.at[idx].set(dist[idx](key=key))
      idx+=1
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )
    # link mass 
    body_mass = jp.ones((model.nbody,))
    for i in range(1, model.nbody):
      rng, key = jax.random.split(rng)
      dmass = dist[idx](key)
      body_mass = body_mass.at[i].set(model.body_mass[i] * dmass)
      idx+=1
    assert idx == len(dr_low)
    return (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    )
  
  if deterministic_cfg is not None:
    # If deterministic_cfg is provided, use it to shift the dynamics.
    (geom_friction, body_ipos, body_mass, dof_frictionloss) = shift_dynamics(params)
  else:
    (
      geom_friction,
      body_ipos,
      body_mass,
      dof_frictionloss,
    ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "dof_frictionloss": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "dof_frictionloss": dof_frictionloss,
  })

  return model, in_axes
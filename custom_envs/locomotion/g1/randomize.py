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
"""Utilities for randomization."""
import jax
from mujoco import mjx
import functools
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16


def domain_randomize(model: mjx.Model, dr_range, rng: jax.Array, params=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
  @jax.vmap
  def shift_dynamics(params):
    idx=0
    pair_friction = model.pair_friction.at[0:2, 0:2].set(params[idx])
    idx+=1

    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] * params[idx:idx+model.nv-6])
    idx+=model.nv-6

    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:]*params[idx:idx+model.nv-6])
    idx+=model.nv-6

    body_mass = model.body_mass.at[:].set(model.body_mass * params[idx:idx+model.nbody])
    idx+=model.nbody

    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + params[idx]
    )
    idx+=1

    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    return (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )

  @jax.vmap
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx=0
    pair_friction = model.pair_friction.at[0:2, 0:2].set(rng_params[idx])
    idx+=1

    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] * rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6

    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:]*rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6

    body_mass = model.body_mass.at[:].set(model.body_mass * rng_params[idx:idx+model.nbody])
    idx+=model.nbody

    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + rng_params[idx]
    )
    idx+=1

    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + rng_params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    return (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )
  if params is not None and rng is None:
    (
        pair_friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = shift_dynamics(params)
  elif params is None and rng is not None:
    (
        pair_friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)
  else:
    raise ValueError('rng or param should be not None!')

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  return model, in_axes
def domain_randomize_eval(model: mjx.Model, dr_range, rng: jax.Array, params=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
  def shift_dynamics(params):
    idx=0
    pair_friction = model.pair_friction.at[0:2, 0:2].set(params[idx])
    idx+=1

    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] * params[idx:idx+model.nv-6])
    idx+=model.nv-6

    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:]*params[idx:idx+model.nv-6])
    idx+=model.nv-6

    body_mass = model.body_mass.at[:].set(model.body_mass * params[idx:idx+model.nbody])
    idx+=model.nbody

    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + params[idx]
    )
    idx+=1

    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    return (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )

  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx=0
    pair_friction = model.pair_friction.at[0:2, 0:2].set(rng_params[idx])
    idx+=1

    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] * rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6

    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:]*rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6

    body_mass = model.body_mass.at[:].set(model.body_mass * rng_params[idx:idx+model.nbody])
    idx+=model.nbody

    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + rng_params[idx]
    )
    idx+=1

    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + rng_params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    return (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )
  if params is not None and rng is None:
    (
        pair_friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = shift_dynamics(params)
  elif params is None and rng is not None:
    (
        pair_friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)
  else:
    raise ValueError('rng or param should be not None!')

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  return model, in_axes

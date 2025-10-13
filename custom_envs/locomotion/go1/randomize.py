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
"""Domain randomization for the Go1 environment."""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1

import functools
def domain_randomize(model: mjx.Model, dr_range, rng: jax.Array, params=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
  @jax.vmap
  def shift_dynamics(params):
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] *params[idx:idx+ model.nv-6])
    idx+=model.nv-6
    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:] * params[idx:idx+model.nv-6])
    idx+=model.nv-6
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + params[idx:idx+3]
    )
    idx+=3
    body_mass = model.body_mass.at[:].set(model.body_mass * params[idx:idx+model.nbody])
    idx+=model.nbody
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + params[idx]
    )
    idx+=1
    # Jitter qpos0: +U(-0.05, 0.05).
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    assert idx == len(params)
    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )
  @jax.vmap
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] *rng_params[idx:idx+ model.nv-6])
    idx+=model.nv-6
    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:] * rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + rng_params[idx:idx+3]
    )
    idx+=3
    body_mass = model.body_mass.at[:].set(model.body_mass * rng_params[idx:idx+model.nbody])
    idx+=model.nbody
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + rng_params[idx]
    )
    idx+=1
    # Jitter qpos0: +U(-0.05, 0.05).
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + rng_params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    assert idx == len(rng_params)
    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )
  if rng is None and params is not None:
    (geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss,dof_armature) = shift_dynamics(params)
  elif rng is not None and params is None:

    (
      geom_friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
    ) = rand_dynamics(rng)
  else:
    raise ValueError(f"only one of the rng or params should be None but rng={rng}, params={params}")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes
def domain_randomize_eval(model: mjx.Model, dr_range, rng: jax.Array, params=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
  def shift_dynamics(params):
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] *params[idx:idx+ model.nv-6])
    idx+=model.nv-6
    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:] * params[idx:idx+model.nv-6])
    idx+=model.nv-6
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + params[idx:idx+3]
    )
    idx+=3
    body_mass = model.body_mass.at[:].set(model.body_mass * params[idx:idx+model.nbody])
    idx+=model.nbody
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + params[idx]
    )
    idx+=1
    # Jitter qpos0: +U(-0.05, 0.05).
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    assert idx == len(params)
    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(model.dof_frictionloss[6:] *rng_params[idx:idx+ model.nv-6])
    idx+=model.nv-6
    dof_armature = model.dof_armature.at[6:].set(model.dof_armature[6:] * rng_params[idx:idx+model.nv-6])
    idx+=model.nv-6
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + rng_params[idx:idx+3]
    )
    idx+=3
    body_mass = model.body_mass.at[:].set(model.body_mass * rng_params[idx:idx+model.nbody])
    idx+=model.nbody
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + rng_params[idx]
    )
    idx+=1
    # Jitter qpos0: +U(-0.05, 0.05).
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + rng_params[idx:idx+model.nv-6]
    )
    idx+=model.nv-6
    assert idx == len(rng_params)
    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )
  if rng is None and params is not None:
    (geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss,dof_armature) = shift_dynamics(params)
  elif rng is not None and params is None:

    (
      geom_friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
    ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes
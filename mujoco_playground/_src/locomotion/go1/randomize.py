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

import functools
import jax
from mujoco import mjx
import jax.numpy as jp
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, params, rng: jax.Array, deterministic: bool = False):
  deterministic_cfg = None
  if not deterministic:
    dr_low, dr_high = params
    dist = [functools.partial(jax.random.uniform,minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))] 
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        # jax.random.uniform(key, minval=0.4, maxval=1.0)
        dist[idx](key)
    )
    idx+=1

    # Scale static friction: *U(0.9, 1.1).
    for i in range(model.nv-6):
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[6+i] * dist[idx](key=key)
      idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 12)
    dof_armature_params = jp.array([dist[idx+i](key=keys[i]) for i in range(12)])
    idx+=12
    armature = model.dof_armature[6:] * dof_armature_params
    # jax.random.uniform(
    #     key, shape=(12,), minval=1.0, maxval=1.05
    # )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Jitter center of mass positiion: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 3)
    dpos = [dist[idx+i](key=keys[i]) for i in range(3)]
    idx+=3
    dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + jp.array(dpos)
    )

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, model.nbody)
    dmass = jp.array([dist[idx+i](key=keys[i]) for i in range(model.nbody)])
    idx += model.nbody
    # dmass = jax.random.uniform(
    #     key, shape=(model.nbody,), minval=0.9, maxval=1.1
    # )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = dist[idx](key=key)
    idx+=1
    # dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )


    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 12)
    qpos0 = model.qpos0
    dqpos = [dist[idx+i](key=keys[i]) for i in range(12)]
    idx +=12
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        # + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
        + jp.array(dqpos)
    )
    assert idx == len(dist), "Index mismatch, check the distribution list."

    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )

  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
  ) = rand_dynamics(rng)

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
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes

def domain_randomize_eval(model: mjx.Model, params, rng: jax.Array, deterministic: bool = False):
  deterministic_cfg = None
  if not deterministic:
    dr_low, dr_high = params
    dist = [functools.partial(jax.random.uniform,minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))] 
  def rand_dynamics(rng):
    # Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    idx=0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        # jax.random.uniform(key, minval=0.4, maxval=1.0)
        dist[idx](key)
    )
    idx+=1

    # Scale static friction: *U(0.9, 1.1).
    for i in range(model.nv-6):
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[6+i] * dist[idx](key=key)
      idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 12)
    dof_armature_params = jp.array([dist[idx+i](key=keys[i]) for i in range(12)])
    idx+=12
    armature = model.dof_armature[6:] * dof_armature_params
    # jax.random.uniform(
    #     key, shape=(12,), minval=1.0, maxval=1.05
    # )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Jitter center of mass positiion: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 3)
    dpos = [dist[idx+i](key=keys[i]) for i in range(3)]
    idx+=3
    dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + jp.array(dpos)
    )

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, model.nbody)
    dmass = jp.array([dist[idx+i](key=keys[i]) for i in range(model.nbody)])
    idx += model.nbody
    # dmass = jax.random.uniform(
    #     key, shape=(model.nbody,), minval=0.9, maxval=1.1
    # )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = dist[idx](key=key)
    idx+=1
    # dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )


    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 12)
    qpos0 = model.qpos0
    dqpos = [dist[idx+i](key=keys[i]) for i in range(12)]
    idx +=12
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        # + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
        + jp.array(dqpos)
    )
    assert idx == len(dist), "Index mismatch, check the distribution list."

    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )

  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
  ) = rand_dynamics(rng)

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
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes


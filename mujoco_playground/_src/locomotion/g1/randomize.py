import functools
import jax
from mujoco import mjx
import jax.numpy as jp

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16

def domain_randomize(model: mjx.Model, params, rng: jax.Array, deterministic: bool = False):
    
    if not deterministic:
        dr_low, dr_high = params
        print('dr_low:', dr_low, 'dr_high:', dr_high)
        print('len', len(dr_low))
        dist = [functools.partial(jax.random.uniform, minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))]
    else:
        ValueError("Deterministic mode not supported for domain randomization.")

    @jax.vmap
    def rand_dynamics(rng):
        idx = 0
        # Floor friction
        rng, key = jax.random.split(rng)
        frictionloss = dist[idx](key=key)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(frictionloss)
        idx += 1

        # Static friction loss
        for i in range(29):
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6+i] * dist[idx](key)
            idx += 1
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Armature
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, 29)
        dof_armature_params = jp.array([dist[idx+i](keys[i]) for i in range(29)])
        idx += 29
        armature = model.dof_armature[6:] * dof_armature_params
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Link masses
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, model.nbody)
        dmass = jp.array([dist[idx+i](keys[i]) for i in range(model.nbody)])
        idx += model.nbody
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Torso mass
        rng, key = jax.random.split(rng)
        dmass = dist[idx](key)
        idx += 1
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, 29)
        dqpos = [dist[idx+i](keys[i]) for i in range(29)]
        idx += 29
        qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + jp.array(dqpos))

        assert idx == len(dist), "Index mismatch, check the distribution list."

        return (
            pair_friction,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        )

    (
        pair_friction,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0,
        "body_mass": 0,
        "qpos0": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
    })

    model = model.tree_replace({
        "pair_friction": pair_friction,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
    })

    return model, in_axes

def domain_randomize_eval(model: mjx.Model, params, rng: jax.Array, deterministic: bool = True):
   
    if not deterministic:
        dr_low, dr_high = params
        dist = [functools.partial(jax.random.uniform, minval=dr_low[i], maxval=dr_high[i]) for i in range(len(dr_low))]
    else:
        dist = [lambda key: 1.0] * len(params[0])  # deterministic: no randomization

    def rand_dynamics(rng):
        idx = 0

        # Floor friction
        rng, key = jax.random.split(rng)
        friction = dist[idx](key=key)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
        idx += 1

        # Static friction loss
        for i in range(29):
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6+i] * dist[idx](key)
            idx += 1
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Armature
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, 29)
        dof_armature_params = jp.array([dist[idx+i](keys[i]) for i in range(29)])
        idx += 29
        armature = model.dof_armature[6:] * dof_armature_params
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Link masses
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, model.nbody)
        dmass = jp.array([dist[idx+i](keys[i]) for i in range(model.nbody)])
        idx += model.nbody
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Torso mass
        rng, key = jax.random.split(rng)
        dmass = dist[idx](key)
        idx += 1
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, 29)
        dqpos = [dist[idx+i](keys[i]) for i in range(29)]
        idx += 29
        qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + jp.array(dqpos))

        assert idx == len(dist), "Index mismatch, check the distribution list."

        return (
            pair_friction,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        )

    (
        pair_friction,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0,
        "body_mass": 0,
        "qpos0": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
    })

    model = model.tree_replace({
        "pair_friction": pair_friction,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
    })

    return model, in_axes
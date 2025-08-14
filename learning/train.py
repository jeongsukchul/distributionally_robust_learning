import os

from learning.agents.wdsac.dr_wrapper import wrap_for_dr_training
from omegaconf import OmegaConf
os.environ['MUJOCO_GL'] = 'glfw'

import mediapy as media
import matplotlib.pyplot as plt

import mujoco
from absl import logging

import imageio
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from agents.rambo import networks as rambo_networks
from agents.rambo import train as rambo
from agents.wdsac import train as wdsac
from agents.wdsac import networks as wdsac_networks
from agents.flowsac import train as flowsac
from agents.flowsac import networks as flowsac_networks
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import wandb
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from configs.training_config import brax_rambo_config, brax_wdsac_config, brax_flowsac_config
import mujoco_playground
from mujoco_playground import wrapper
import hydra
from mujoco_playground import registry
from helper import parse_cfg
from helper import Logger
from helper import make_dir
import warnings
import pickle
import shutil
from mujoco_playground._src.wrapper import Wrapper, wrap_for_brax_training
from mujoco_playground._src import mjx_env
from utils import save_configs_to_wandb_and_local



#egl로 바꾸는 게 왜인지 모르겠지만 RAM을 적게 먹는다.
# # Ignore the info logs from brax
# logging.set_verbosity(logging.WARNING)

# warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# # Suppress DeprecationWarnings from JAX
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# # Suppress UserWarnings from absl (used by JAX and TensorFlow)
# warnings.filterwarnings("ignore", category=UserWarning, module="absl")

env_name = "FishSwim"  # @param ["AcrobotSwingup", "AcrobotSwingupSparse", "BallInCup", "CartpoleBalance", "CartpoleBalanceSparse", "CartpoleSwingup", "CartpoleSwingupSparse", "CheetahRun", "FingerSpin", "FingerTurnEasy", "FingerTurnHard", "FishSwim", "HopperHop", "HopperStand", "HumanoidStand", "HumanoidWalk", "HumanoidRun", "PendulumSwingup", "PointMass", "ReacherEasy", "ReacherHard", "SwimmerSwimmer6", "WalkerRun", "WalkerStand", "WalkerWalk"]
CAMERAS = {
    "AcrobotSwingup": "fixed",
    "AcrobotSwingupSparse": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerWalk": "side",
    "WalkerStand": "side",
}
camera_name = CAMERAS[env_name]

class BraxDomainRandomizationWrapper(Wrapper):
  """Brax wrapper for domain randomization."""
  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model, self._in_axes = randomization_fn(self.env.mjx_model)
    self.env.unwrapped._mjx_model = self._mjx_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    res = self.env.step(state, action)
    return res

def policy_params_fn(current_step, make_policy, params, ckpt_path: epath.Path):
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f"{current_step}"
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)
def progress_fn(num_steps, metrics, use_wandb=True):
    if use_wandb:
        wandb.log(metrics, step=num_steps)
    print("-------------------------------------------------------------------")
    print(f"num_steps: {num_steps}")
    
    for k,v in metrics.items():
        print(f" {k} :  {v}")
    print("-------------------------------------------------------------------")


def train_ppo(cfg:dict, randomization_fn, env, eval_env):

    print("training with ppo")
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        ppo_params = dm_control_suite_params.brax_ppo_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        ppo_params = locomotion_params.brax_ppo_config(cfg.task)
    print("shift_dynamics:", cfg.shift_dynamics)
    if cfg.shift_dynamics:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.{cfg.shift_dynamics_type}.hard_dr={cfg.custom_wrapper}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}"
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=functools.partial(policy_params_fn, ckpt_path=cfg.work_dir / "models" ),
        randomization_fn=randomization_fn,
    )
    if cfg.custom_wrapper and cfg.shift_dynamics:
        wrap_fn = functools.partial(wrap_for_dr_training, n_nominals=1,n_envs=ppo_params.num_envs)
    else:
        wrap_fn = wrap_for_brax_training
    wrap_eval_fn = wrap_for_brax_training
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env = eval_env,
        wrap_env_fn=wrap_fn,
        wrap_eval_env_fn= wrap_eval_fn
    )
    return make_inference_fn, params, metrics

def train_sac(cfg:dict, randomization_fn, env, eval_env):
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        sac_params = dm_control_suite_params.brax_sac_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        sac_params = locomotion_params.brax_sac_config(cfg.task)
    sac_training_params = dict(sac_params)
    if cfg.shift_dynamics:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.{cfg.shift_dynamics_type}.hard_dr={cfg.custom_wrapper}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}"
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
    )
    if cfg.custom_wrapper and cfg.shift_dynamics:
        wrap_fn = functools.partial(wrap_for_dr_training, n_nominals=1,n_envs=sac_params.num_envs)
    else:
        wrap_fn = wrap_for_brax_training
    wrap_eval_fn = wrap_for_brax_training
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = eval_env,
        wrap_env_fn=wrap_fn,
        wrap_eval_env_fn=wrap_eval_fn,
    )
    return make_inference_fn, params, metrics

def train_rambo(cfg:dict, randomization_fn, env, eval_env):
    times = [datetime.now()]
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        rambo_params = brax_rambo_config(cfg.task)
    # elif cfg.task in mujoco_playground._src.locomotion._envs:
    #     sac_params = locomotion_params.brax_sac_config(cfg.task)

    for param in rambo_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            rambo_params[param] = getattr(cfg, param)
    if cfg.use_wandb:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.({rambo_params.rollout_length},\
              {rambo_params.adv_weight}, {rambo_params.real_ratio})_{rambo_params.batch_size}_{rambo_params.rollout_batch_size}"
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    rambo_training_params = dict(rambo_params)
    network_factory = rambo_networks.make_rambo_networks
    if "network_factory" in rambo_params:
        del rambo_training_params["network_factory"]
        network_factory = functools.partial(
            rambo_networks.make_rambo_networks,
            **rambo_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        rambo.train, **dict(rambo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    return make_inference_fn, params, metrics

def train_wdsac(cfg:dict, randomization_fn, env, eval_env):
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        wdsac_params = brax_wdsac_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        wdsac_params = locomotion_params.brax_sac_config(cfg.task)
    for param in wdsac_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            wdsac_params[param] = getattr(cfg, param)
    if cfg.shift_dynamics:
        wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.{cfg.shift_dynamics_type}.delta={wdsac_params.delta}\
            .nominals={wdsac_params.n_nominals}.single_lambda={wdsac_params.single_lambda}.\
                distance_type={wdsac_params.distance_type}.length={wdsac_params.lambda_update_steps}\
                    .lmbda_lr={wdsac_params.lmbda_lr}.init_lmbda={wdsac_params.init_lmbda}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}"
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    network_factory = wdsac_networks.make_wdsac_networks
    wdsac_training_params = dict(wdsac_params)
    if "network_factory" in wdsac_params:
        del wdsac_training_params["network_factory"]
        network_factory = functools.partial(
            wdsac_networks.make_wdsac_networks,
            **wdsac_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        wdsac.train, **dict(wdsac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    return make_inference_fn, params, metrics

def train_flowsac(cfg:dict, randomization_fn, env, eval_env):
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        flowsac_params = brax_flowsac_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        flowsac_params = locomotion_params.brax_sac_config(cfg.task)
    for param in flowsac_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            flowsac_params[param] = getattr(cfg, param)
    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.{cfg.shift_dynamics_type}.delta={flowsac_params.delta}\
            .single_lambda={flowsac_params.single_lambda}\
            .length={flowsac_params.lambda_update_steps}.lmbda_lr={flowsac_params.lmbda_lr}.init_lmbda={flowsac_params.init_lmbda}.flow_lr={flowsac_params.flow_lr}"
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    network_factory = flowsac_networks.make_flowsac_networks
    flowsac_training_params = dict(flowsac_params)
    if "network_factory" in flowsac_params:
        del flowsac_training_params["network_factory"]
        network_factory = functools.partial(
            flowsac_networks.make_flowsac_networks,
            **flowsac_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        flowsac.train, **dict(flowsac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
    )
    if cfg.custom_wrapper and cfg.shift_dynamics:
        wrap_fn = functools.partial(wrap_for_dr_training, n_nominals=1,n_envs=flowsac_params.num_envs)
    else:
        wrap_fn = wrap_for_brax_training
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = eval_env,
        wrap_env_fn=wrap_fn,
    )
    return make_inference_fn, params, metrics

@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    rng = jax.random.PRNGKey(cfg.seed)
    
    # if cfg.shift_dynamics:
    path = epath.Path(".").resolve()
    cfg_dir = make_dir(cfg.work_dir / "cfg")
    shutil.copy('config.yaml', os.path.join(cfg_dir, 'config.yaml'))
    env_cfg = registry.get_default_config(cfg.task)
    env = registry.load(cfg.task, config=env_cfg)

    if cfg.shift_dynamics_type == "stochastic":
        randomizer = registry.get_domain_randomizer(cfg.task)
        randomizer = functools.partial(randomizer, params=env.dr_range)

    else:
        raise ValueError(f"Unknown dynamics shift type: {cfg.shift_dynamics_type}")
    if cfg.shift_dynamics:
        randomization_fn = randomizer
    else:
        randomization_fn = None 
    print("shift_dynamics", cfg.shift_dynamics)
    print("randomization_fn:", randomization_fn)

    if cfg.policy == "sac":
        make_inference_fn, params, metrics = train_sac(cfg, randomization_fn, env, None)
    elif cfg.policy == "ppo":
        make_inference_fn, params, metrics = train_ppo(cfg, randomization_fn, env, None)
    elif cfg.policy == "rambo":
        make_inference_fn, params, metrics = train_rambo(cfg, randomization_fn, env, None)
    elif cfg.policy == "wdsac":
        make_inference_fn, params, metrics = train_wdsac(cfg, randomization_fn, env, None)
    elif cfg.policy == "flowsac":
        make_inference_fn, params, metrics = train_flowsac(cfg, randomization_fn, env, None)
    else:
        print("no policy!")


    save_dir = make_dir(cfg.work_dir / "models")
    print(f"Saving parameters to {save_dir}")
    with open(os.path.join(save_dir, f"{cfg.policy}_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(params, f)
    latest_path = os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(params, f)

    # Save config.yaml and shift_dynamics config to wandb and local directory
    save_configs_to_wandb_and_local(cfg, cfg.work_dir)
    if cfg.eval_randomization:
        eval_rng, rng = jax.random.split(rng)
        randomizer_eval = registry.get_domain_randomizer_eval(cfg.task)
        if cfg.shift_dynamics_type == "stochastic":
            randomizer_eval = functools.partial(randomizer_eval, rng=eval_rng, params=env.dr_range)
        # elif cfg.shift_dynamics_type == "deterministic":
        #     randomizer_eval = functools.partial(randomizer_eval, rng=eval_rng,deterministic_cfg=deterministic_cfg, stochastic_cfg=None)
        else:
            raise ValueError(f"Unknown dynamics shift type: {cfg.shift_dynamics_type}")
        eval_env = BraxDomainRandomizationWrapper(
            registry.load(cfg.task),
            randomization_fn=randomizer_eval,
        )
    else:
        eval_env = registry.load(cfg.task)
    if cfg.save_video and cfg.use_wandb:
        jit_inference_fn = jax.jit(make_inference_fn(params,deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        total_reward = 0.0

        state = jit_reset(jax.random.PRNGKey(0))
        rollout = [state]
        for _ in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            action, info = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            rollout.append(state)
            total_reward += state.reward
            
        frames = eval_env.render(rollout, camera=CAMERAS[cfg.task],)
        frames = np.stack(frames).transpose(0, 3, 1, 2)
        fps=1.0 / env.dt
        wandb.log({'eval_video': wandb.Video(frames, fps=fps, format='mp4')})
        wandb.log({'final_eval_reward' : total_reward})

   
if __name__ == "__main__":
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    train()
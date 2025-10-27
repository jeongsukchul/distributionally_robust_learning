import os

from omegaconf import OmegaConf
os.environ['MUJOCO_GL'] = 'egl'

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
from agents.ppo import networks as ppo_networks
from agents.ppo import train as ppo
from agents.sac import networks as sac_networks
from agents.sac import train as sac
from agents.td3 import networks as td3_networks
from agents.td3 import train as td3
from agents.rambo import networks as rambo_networks
from agents.rambo import train as rambo
from agents.wdsac import train as wdsac
from agents.wdsac import networks as wdsac_networks
from agents.wdtd3 import train as wdtd3
from agents.wdtd3 import networks as wdtd3_networks
from agents.flowsac import train as flowsac
from agents.flowsac import networks as flowsac_networks
from agents.flowtd3 import train as flowtd3
from agents.flowtd3 import networks as flowtd3_networks
from agents.gmmtd3 import train as gmmtd3
from agents.gmmtd3 import networks as gmmtd3_networks
from agents.m2td3 import train as m2td3
from agents.m2td3 import networks as m2td3_networks
from agents.tdmpc import train as tdmpc
from agents.tdmpc import networks as tdmpc_networks
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
from learning.configs.dm_control_training_config import brax_ppo_config, brax_sac_config, brax_td3_config, brax_tdmpc_config
from learning.configs.locomotion_training_config import locomotion_ppo_config, locomotion_sac_config, locomotion_td3_config
from learning.configs.manipulation_training_config import manipulation_ppo_config, manipulation_td3_config
import hydra
from custom_envs import registry, dm_control_suite, locomotion, manipulation
from brax import envs
from helper import parse_cfg
from helper import Logger
from helper import make_dir
import warnings
import pickle
import shutil
from learning.module.wrapper.wrapper import Wrapper, wrap_for_brax_training
from custom_envs import mjx_env
from utils import save_configs_to_wandb_and_local
from learning.module.wrapper.wrapper import Wrapper, wrap_for_brax_training
import scipy


import jax.numpy as jnp
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
    "Go1Handstand": "side",
    "Go1JoystickRoughTerrain": "track",
    "G1InplaceGaitTracking" : "track",
    "G1JoystickGaitTracking" : "track",
    "T1JoystickFlatTerrain" :"track",
    "T1JoystickRoughTerrain" :"track",
    "LeapCubeRotateZAxis" :"side",
    "LeapCubeReorient" :"side",
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


def train_ppo(cfg:dict, randomization_fn, env, eval_env=None):

    print("training with ppo")
    if cfg.task in dm_control_suite._envs:
        ppo_params = brax_ppo_config(cfg.task)
    elif cfg.task in locomotion._envs:
        ppo_params = locomotion_ppo_config(cfg.task)
    elif cfg.task in manipulation._envs:
        ppo_params = manipulation_ppo_config(cfg.task)
    if cfg.randomization:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.domain_randomized"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.eval_rand={cfg.eval_randomization}"
    wandb_name += cfg.comment
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
        if not cfg.asymmetric_critic:
            ppo_params.network_factory.value_obs_key = "state"
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
        seed=cfg.seed,
        # custom_wrapper = cfg.custom_wrapper
    )
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
    )
    return make_inference_fn, params, metrics

def train_sac(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        sac_params = brax_sac_config(cfg.task)
    elif cfg.task in locomotion._envs:
        sac_params = locomotion_sac_config(cfg.task)
    sac_training_params = dict(sac_params)
    if cfg.randomization:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.\
            hard_dr={cfg.custom_wrapper}.adv_wrapper={cfg.adv_wrapper}"#.dr_train_ratio={cfg.dr_train_ratio}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.eval_rand={cfg.eval_randomization}"
    wandb_name += f"simba={cfg.simba}"
    wandb_name += cfg.comment
    wandb_name 
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    if cfg.simba:
        network_factory = sac_networks.make_simba_sac_networks
    else:
        network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            sac_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            sac_networks.make_simba_sac_networks if cfg.simba else sac_networks.make_sac_networks,
            **sac_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        custom_wrapper=cfg.custom_wrapper,
        seed=cfg.seed,
        adv_wrapper = cfg.adv_wrapper
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_td3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        td3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        td3_params = locomotion_td3_config(cfg.task)
    elif cfg.task in manipulation._envs:
        td3_params = manipulation_td3_config(cfg.task)
    td3_training_params = dict(td3_params)
    if cfg.randomization:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.\
            hard_dr={cfg.custom_wrapper}"
        if cfg.custom_wrapper and cfg.adv_wrapper:
            wandb_name+=f".adv_wrapper={cfg.adv_wrapper}"#dr_train_ratio={cfg.dr_train_ratio}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.eval_rand={cfg.eval_randomization}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})

    network_factory = td3_networks.make_td3_networks
    if "network_factory" in td3_params:
        del td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            td3_networks.make_td3_networks,
            **td3_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        td3.train, **dict(td3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        custom_wrapper = cfg.custom_wrapper,
        seed=cfg.seed,
        adv_wrapper = cfg.adv_wrapper
    )
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_m2td3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        m2td3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        m2td3_params = locomotion_td3_config(cfg.task)
    m2td3_params.omega_distance_threshold = 0.1
    for param in m2td3_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            m2td3_params[param] = getattr(cfg, param)
    print("omega_distance_threshold:", m2td3_params.omega_distance_threshold)
    m2td3_training_params = dict(m2td3_params)
    wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.dist={m2td3_params.omega_distance_threshold}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name, 
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})

    network_factory = m2td3_networks.make_m2td3_networks
    if "network_factory" in m2td3_params:
        del m2td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            m2td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            m2td3_networks.make_m2td3_networks,
            **m2td3_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        m2td3.train, **dict(m2td3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_flowsac(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        flowsac_params = brax_sac_config(cfg.task)
    elif cfg.task in locomotion._envs:
        flowsac_params = locomotion_sac_config(cfg.task)
    flowsac_params.delta = 0.01
    flowsac_params.lambda_update_steps = 10
    flowsac_params.lmbda_lr = 3e-4
    flowsac_params.flow_lr = 3e-4
    flowsac_params.init_lmbda = 0.

    for param in flowsac_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            flowsac_params[param] = getattr(cfg, param)

    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.dr_train_ratio={cfg.dr_train_ratio}\
                .init_lmbda={flowsac_params.init_lmbda}.flow_lr={flowsac_params.flow_lr}.dr_flow={cfg.dr_flow}.simba={cfg.simba}\
                    .eval_param={cfg.eval_with_training_env}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    flowsac_training_params = dict(flowsac_params)
    if "network_factory" in flowsac_params:
        if not cfg.asymmetric_critic:
            flowsac_params.network_factory.value_obs_key = "state"
        del flowsac_training_params["network_factory"]
        network_factory = functools.partial(
            flowsac_networks.make_flowsac_networks,
            simba=cfg.simba,
            **flowsac_params.network_factory,
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(  
        flowsac.train, **dict(flowsac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        use_wandb=cfg.use_wandb,
        dr_flow = cfg.dr_flow,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
        eval_with_training_env = cfg.eval_with_training_env,
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = eval_env,
        wrap_eval_env_fn = wrap_for_brax_training,
    )
    return make_inference_fn, params, metrics
def train_flowtd3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        flowtd3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        flowtd3_params = locomotion_td3_config(cfg.task)
    flowtd3_params.flow_lr = 3e-4
    flowtd3_params.init_lmbda = 0.
    for param in flowtd3_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            flowtd3_params[param] = getattr(cfg, param)

    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.dr_train_ratio={cfg.dr_train_ratio}\
                .init_lmbda={flowtd3_params.init_lmbda}.flow_lr={flowtd3_params.flow_lr} \
                    .eval_param={cfg.eval_with_training_env}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    flowtd3_training_params = dict(flowtd3_params)
    if "network_factory" in flowtd3_params:
        if not cfg.asymmetric_critic:
            flowtd3_params.network_factory.value_obs_key = "state"
        del flowtd3_training_params["network_factory"]
        network_factory = functools.partial(
            flowtd3_networks.make_flowtd3_networks,
            **flowtd3_params.network_factory,
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(  
        flowtd3.train, **dict(flowtd3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        use_wandb=cfg.use_wandb,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
        eval_with_training_env = cfg.eval_with_training_env,
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_gmmtd3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        gmmtd3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        gmmtd3_params = locomotion_td3_config(cfg.task)
    for param in gmmtd3_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            gmmtd3_params[param] = getattr(cfg, param)
    gmmtd3_params['num_evals'] = 1000
    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.dr_train_ratio={cfg.dr_train_ratio}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    gmmtd3_training_params = dict(gmmtd3_params)
    if "network_factory" in gmmtd3_params:
        if not cfg.asymmetric_critic:
            gmmtd3_params.network_factory.value_obs_key = "state"
        del gmmtd3_training_params["network_factory"]
        network_factory = functools.partial(
            gmmtd3_networks.make_gmmtd3_networks,
            **gmmtd3_params.network_factory,
        )
    randomizer = registry.get_domain_randomizer_eval(cfg.task)
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(  
        gmmtd3.train, **dict(gmmtd3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        eval_randomization_fn=randomization_fn,
        randomization_fn=randomizer,
        use_wandb=cfg.use_wandb,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
        eval_with_training_env = cfg.eval_with_training_env,
        value_obs_key = gmmtd3_params.network_factory.value_obs_key,
    )
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_wdtd3(cfg:dict, randomization_fn, env):
    if cfg.task in dm_control_suite._envs:
        wdtd3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        wdtd3_params = locomotion_td3_config(cfg.task)
    wdtd3_params.n_nominals = 10# added
    wdtd3_params.delta = 0.01    #added
    wdtd3_params.lambda_update_steps = 100  #added: number of lambda optimization steps
    wdtd3_params.single_lambda = False #added
    wdtd3_params.distance_type = "wass" #added
    wdtd3_params.lmbda_lr = 3e-4 #added
    wdtd3_params.init_lmbda = 0. #added
    for param in wdtd3_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            wdtd3_params[param] = getattr(cfg, param)
    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.delta={wdtd3_params.delta}\
        .nominals={wdtd3_params.n_nominals}.single_lambda={wdtd3_params.single_lambda}.asym={cfg.asymmetric_critic}\
            distance_type={wdtd3_params.distance_type}.length={wdtd3_params.lambda_update_steps}\
                .lmbda_lr={wdtd3_params.lmbda_lr}.init_lmbda={wdtd3_params.init_lmbda}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    network_factory = wdtd3_networks.make_wdtd3_networks
    wdtd3_training_params = dict(wdtd3_params)
    if "network_factory" in wdtd3_params:
        if not cfg.asymmetric_critic:
            wdtd3_params.network_factory.value_obs_key = "state"
        del wdtd3_training_params["network_factory"]
        network_factory = functools.partial(
            wdtd3_networks.make_wdtd3_networks,
            **wdtd3_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        wdtd3.train, **dict(wdtd3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        seed=cfg.seed,
        randomization_fn=randomization_fn,
    )

    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_tdmpc(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        tdmpc_params = brax_tdmpc_config(cfg.task)
    # elif cfg.task in locomotion._envs:
        # tdmpc_params = locomotion_tdmpc_config(cfg.task)
    for param in tdmpc_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            tdmpc_params[param] = getattr(cfg, param)
    tdmpc_training_params = dict(tdmpc_params)
    if cfg.randomization:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.\
            hard_dr={cfg.custom_wrapper}"
        if cfg.custom_wrapper and cfg.adv_wrapper:
            wandb_name+=f".adv_wrapper={cfg.adv_wrapper}"#dr_train_ratio={cfg.dr_train_ratio}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.eval_rand={cfg.eval_randomization}"
    wandb_name += cfg.comment

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})

    network_factory = tdmpc_networks.make_tdmpc_networks
    if "network_factory" in tdmpc_params:
        del tdmpc_training_params["network_factory"]
        network_factory = functools.partial(
            tdmpc_networks.make_tdmpc_networks,
            **tdmpc_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        tdmpc.train, **dict(tdmpc_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        custom_wrapper = cfg.custom_wrapper,
        seed=cfg.seed,
        adv_wrapper = cfg.adv_wrapper
    )
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics
@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    rng = jax.random.PRNGKey(cfg.seed)
    
    path = epath.Path(".").resolve()
    cfg_dir = make_dir(cfg.work_dir / "cfg")
    shutil.copy('config.yaml', os.path.join(cfg_dir, 'config.yaml'))
    env_cfg = registry.get_default_config(cfg.task)
    env_cfg['impl'] = cfg.impl
    if cfg.policy == "td3" :
        if "stand" in cfg.task:
            env_cfg.reward_config.scales.energy = -5e-5
            env_cfg.reward_config.scales.action_rate = -1e-1
            env_cfg.reward_config.scales.torques = -1e-3
        elif "T1" in cfg.task or "G1" in cfg.task:
            env_cfg.reward_config.scales.energy = -5e-5
            env_cfg.reward_config.scales.action_rate = -1e-1
            env_cfg.reward_config.scales.torques = -1e-3
            env_cfg.reward_config.scales.pose = -1.0
            env_cfg.reward_config.scales.tracking_ang_vel = 1.25
            env_cfg.reward_config.scales.tracking_lin_vel = 1.25
            env_cfg.reward_config.scales.feet_phase = 1.0
            env_cfg.reward_config.scales.ang_vel_xy = -0.3
            env_cfg.reward_config.scales.orientation = -5.0
    
    env = registry.load(cfg.task, config=env_cfg)

    if cfg.randomization:
        randomizer = registry.get_domain_randomizer(cfg.task)
        randomization_fn = randomizer
    else:
        randomization_fn = None 

    print("randomization_fn:", randomization_fn)
    if cfg.policy == "sac":
        make_inference_fn, params, metrics = train_sac(cfg, randomization_fn, env)
    if cfg.policy == "td3":
        make_inference_fn, params, metrics = train_td3(cfg, randomization_fn, env)
    elif cfg.policy == "ppo":
        make_inference_fn, params, metrics = train_ppo(cfg, randomization_fn, env)
    elif cfg.policy == "flowsac":
        make_inference_fn, params, metrics = train_flowsac(cfg, randomization_fn, env)
    elif cfg.policy == "flowtd3":
        make_inference_fn, params, metrics = train_flowtd3(cfg, randomization_fn, env)
    elif cfg.policy == "gmmtd3":
        make_inference_fn, params, metrics = train_gmmtd3(cfg, randomization_fn, env)
    elif cfg.policy == "wdtd3":
        make_inference_fn, params, metrics = train_wdtd3(cfg, randomization_fn, env)
    elif cfg.policy == "m2td3":
        make_inference_fn, params, metrics = train_m2td3(cfg, randomization_fn, env)
    elif cfg.policy == "tdmpc":
        make_inference_fn, params, metrics = train_tdmpc(cfg, randomization_fn, env)
    else:
        print("no policy!")


    save_dir = make_dir(cfg.work_dir / "models")
    print(f"Saving parameters to {save_dir}")
    with open(os.path.join(save_dir, f"{cfg.policy}_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(params, f)
    latest_path = os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(params, f)

    # Save config.yaml and randomization config to wandb and local directory
    save_configs_to_wandb_and_local(cfg, cfg.work_dir)
    print("eval_randomization", cfg.eval_randomization)
    if cfg.eval_randomization:
        eval_rng, rng = jax.random.split(rng)
        randomizer_eval = registry.get_domain_randomizer_eval(cfg.task)
        randomizer_eval = functools.partial(randomizer_eval, rng=eval_rng, dr_range=env.dr_range)
        eval_env = BraxDomainRandomizationWrapper(
            registry.load(cfg.task, config=env_cfg),
            randomization_fn=randomizer_eval,
        )
    else:
        eval_env = registry.load(cfg.task, config=env_cfg)
    if cfg.save_video and cfg.use_wandb:
        n_episodes = 100
        jit_inference_fn = jax.jit(make_inference_fn(params,deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        reward_list = []
        rollout = []
        rng, eval_rng = jax.random.split(rng)
        rngs = jax.random.split(eval_rng, n_episodes)
        for i in range(n_episodes): #10 episodes
            state = jit_reset(rngs[i])
            rollout = [state]
            rewards = 0
            for _ in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                action, info = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                rollout.append(state)
                rewards += state.reward
            reward_list.append(rewards)
            
        frames = eval_env.render(rollout, camera=CAMERAS[cfg.task],)
        frames = np.stack(frames).transpose(0, 3, 1, 2)
        fps=1.0 / env.dt
        rewards = jnp.asarray(reward_list)
        wandb.log({'final_eval_reward' : rewards.mean()})
        wandb.log({'final_eval_reward_iqm' : scipy.stats.trim_mean(rewards, proportiontocut=0.25, axis=None) })
        wandb.log({'final_eval_reward_std' :rewards.std() })
        wandb.log({'eval_video': wandb.Video(frames, fps=fps, format='mp4')})

   
if __name__ == "__main__":
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    train()
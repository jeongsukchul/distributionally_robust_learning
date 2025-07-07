import os

from omegaconf import OmegaConf
os.environ['MUJOCO_GL'] = 'glfw'
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

import mediapy as media
import matplotlib.pyplot as plt

import mujoco
from absl import logging


# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
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
import mujoco_playground
from mujoco_playground import wrapper
import hydra
from mujoco_playground import registry
from helper import parse_cfg
from helper import Logger
from helper import make_dir
import warnings
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

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
import pickle
import shutil
def policy_params_fn(current_step, make_policy, params, ckpt_path: epath.Path):
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f"{current_step}"
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)



    # if cfg.save_video:
    #         logger.video.save(num_steps, key='results/video')


def train_ppo(cfg:dict):
    times = [datetime.now()]

    print("training with ppo")
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        ppo_params = dm_control_suite_params.brax_ppo_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        ppo_params = locomotion_params.brax_ppo_config(cfg.task)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    if cfg.dynamics_shift:
        path = epath.Path(".").resolve()
        if cfg.dynamics_shift_type == "stochastic":
            dynamics_path = os.path.join(path, "dynamics_shift", "stochastic", f"{cfg.task}.yaml")
            shutil.copy(dynamics_path, cfg.work_dir / "dynamics_shift" / "stochastic")
            stochastic_cfg = OmegaConf.load(dynamics_path)
            randomization_fn = registry.get_domain_randomizer(cfg.task)
            randomization_fn = functools.partial(randomization_fn,deterministic_cfg=None, stochastic_cfg=stochastic_cfg)

        elif cfg.dynamics_shift_type == "deterministic":
            dynamics_path = os.path.join(path, "dynamics_shift", "deterministic", f"{cfg.task}.yaml")
            make_dir(cfg.work_dir / "dynamics_shift" / "deterministic")
            shutil.copy(dynamics_path, cfg.work_dir / "dynamics_shift" / "deterministic")
            deterministic_cfg = OmegaConf.load(dynamics_path)
            randomization_fn = registry.get_domain_randomizer(cfg.task)
            randomization_fn = functools.partial(randomization_fn, deterministic_cfg=deterministic_cfg, stochastic_cfg=None)
        else:
            raise ValueError(f"Unknown dynamics shift type: {cfg.dynamics_shift_type}")
    else:
        randomization_fn = None
    def progress(num_steps, metrics, use_wandb=True):
        times.append(datetime.now())
        if use_wandb:
            wandb_metrics={}
            wandb_metrics["eval/episode_reward"] = metrics["eval/episode_reward"]
            wandb_metrics["eval/episode_length"] = metrics["eval/avg_episode_length"]
            wandb.log(wandb_metrics, step=num_steps)
        print("-------------------------------------------------------------------")
        print(f"num_steps: {num_steps}")
        
        for k,v in metrics.items():
            print(f" {k} :  {v}")
        print("-------------------------------------------------------------------")
        
    progress = functools.partial(progress, use_wandb=cfg.use_wandb)

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=functools.partial(policy_params_fn, ckpt_path=cfg.work_dir / "models" ),
        randomization_fn=randomization_fn,
    )

    env_cfg = registry.get_default_config(cfg.task)
    env = registry.load(cfg.task, config=env_cfg)
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env = registry.load(cfg.task, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, params, metrics

def train_sac(cfg:dict):
    times = [datetime.now()]
    if cfg.task in mujoco_playground._src.dm_control_suite._envs:
        sac_params = dm_control_suite_params.brax_sac_config(cfg.task)
    elif cfg.task in mujoco_playground._src.locomotion._envs:
        sac_params = locomotion_params.brax_sac_config(cfg.task)
    sac_training_params = dict(sac_params)
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_params.network_factory
        )

    if cfg.dynamics_shift:
        path = epath.Path(".").resolve()
        if cfg.dynamics_shift_type == "stochastic":
            dynamics_path = os.path.join(path, "dynamics_shift", "stochastic", f"{cfg.task}.yaml")
            wandb.save(dynamics_path)
            stochastic_cfg = OmegaConf.load(dynamics_path)
            randomization_fn = registry.get_domain_randomizer(cfg.task)
            randomization_fn = functools.partial(randomization_fn, deterministic_cfg=None, stochastic_cfg=stochastic_cfg)

        elif cfg.dynamics_shift_type == "deterministic":
            dynamics_path = os.path.join(path, "dynamics_shift", "deterministic", f"{cfg.task}.yaml")
            deterministic_cfg = OmegaConf.load(dynamics_path)
            randomization_fn = registry.get_domain_randomizer(cfg.task)
            randomization_fn = functools.partial(randomization_fn, deterministic_cfg=deterministic_cfg, stochastic_cfg=None)
        else:
            raise ValueError(f"Unknown dynamics shift type: {cfg.dynamics_shift_type}")
    else:
        randomization_fn = None
    def progress(num_steps, metrics, use_wandb=True):
        times.append(datetime.now())
        

        if use_wandb:
            wandb_metrics={}
            wandb_metrics["eval/episode_reward"] = metrics["eval/episode_reward"]
            wandb_metrics["eval/episode_length"] = metrics["eval/avg_episode_length"]
            wandb.log(wandb_metrics, step=num_steps)
            for k,v in metrics.items():
                print(f" {k} :  {v}")
        else:
            for k,v in metrics.items():
                print(f" {k} :  {v}")
        
    progress = functools.partial(progress, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
    )

    env_cfg = registry.get_default_config(cfg.task)
    env = registry.load(cfg.task, config=env_cfg)
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
        eval_env = registry.load(cfg.task, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, params, metrics

@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)

    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    wandb.init(
        project=cfg.wandb_project, 
        entity=cfg.wandb_entity, 
        name=f"{cfg.task}.{cfg.policy}.{cfg.seed}",
        dir=make_dir(cfg.work_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.config.update({"env_name": cfg.task})
    if cfg.policy == "sac":
        make_inference_fn, params, metrics = train_sac(cfg)
    elif cfg.policy == "ppo":
        make_inference_fn, params, metrics = train_ppo(cfg)
    # elif cfg.policy == "td-mpc":
    #     train_tdmpc(cfg)
    else:
        print("no policy!")
    save_dir = make_dir(cfg.work_dir / "models")
    print(f"Saving parameters to {save_dir}")
    with open(os.path.join(save_dir, f"{cfg.policy}_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(params, f)
    latest_path = os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(params, f)

   
if __name__ == "__main__":
    train()
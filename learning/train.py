import os

from omegaconf import OmegaConf
os.environ['MUJOCO_GL'] = 'glfw'
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

import mediapy as media
import matplotlib.pyplot as plt

import mujoco


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
from mujoco_playground import wrapper
import hydra
from mujoco_playground import registry
from helper import parse_cfg
from helper import Logger
from helper import make_dir
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



    # if cfg.save_video:
    #         logger.video.save(num_steps, key='results/video')


def train_ppo(cfg:dict):
    print("training with ppo")
    ppo_params = dm_control_suite_params.brax_ppo_config(cfg.task)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    env = registry.load(cfg.task)
    times = [datetime.now()]
    
    def progress(num_steps, metrics, use_wandb=True):
        times.append(datetime.now())
        wandb_metrics={}
        wandb_metrics["eval/episode_reward"] = metrics["eval/episode_reward"]
        wandb_metrics["eval/episode_length"] = metrics["eval/avg_episode_length"]
        for k,v in metrics.items():
            if not "eval" in k:
                wandb_metrics[k] = v
        if use_wandb:
            wandb.log(wandb_metrics, step=num_steps)
            for k,v in metrics.items():
                print(f" {k} :  {v}")
        else:
            for k,v in metrics.items():
                print(f" {k} :  {v}")
        
    progress = functools.partial(progress, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress
    )


    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, params, metrics

def train_sac(cfg:dict):
    sac_params = dm_control_suite_params.brax_sac_config(env_name)
    sac_training_params = dict(sac_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
    network_factory = functools.partial(
        sac_networks.make_ppo_networks,
        **sac_params.network_factory
    )

    env = registry.load(cfg.task)
    times = [datetime.now()]

    progress = functools.partial(progress, wandb=cfg.use_wandb)
    def progress(num_steps, metrics, use_wandb=True):
        times.append(datetime.now())
        wandb_metrics={}
        wandb_metrics["eval/episode_reward"] = metrics["eval/episode_reward"]
        wandb_metrics["eval/episode_length"] = metrics["eval/avg_episode_length"]
        for k,v in metrics.items():
            if not "eval" in k:
                wandb_metrics[k] = v
        if use_wandb:
            wandb.log(wandb_metrics, step=num_steps)
        else:
            for k,v in metrics:
                print(f" {k} :  {v}")
    progress = functools.partial(progress, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=progress
    )


    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, params, metrics

@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)
    m = mujoco.MjModel.from_xml_string('<mujoco/>')
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    # More legible printing from numpy.
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    wandb.init(
        project=cfg.wandb_project, 
        entity=cfg.wandb_entity, 
        name=f"{cfg.task}.{cfg.exp_name}.{cfg.seed}",
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
    save_dir = make_dir(cfg.wrok_dir / "models")

    with open(os.path.join(save_dir, f"{cfg.policy}_params.pkl"), "wb") as f:
        pickle.dump(params, f)
   
if __name__ == "__main__":
    train()
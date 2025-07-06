import os
import pickle
import wandb
import jax
from datetime import datetime
from mujoco_playground import registry, wrapper
from helper import parse_cfg
from omegaconf import OmegaConf
from flax.training import checkpoints
from brax.training.agents.ppo import networks as ppo_networks

import hydra

def rollout(env, env_cfg, policy, jit_step, jit_reset):
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state]

    f = 0.5
    for i in range(env_cfg.episode_length):
        
        action = policy(state)
        state = jit_step(state, action)
        rollout.append(state)

    frames = env.render(rollout)
    media.show_video(frames, fps=1.0 / env.dt)


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


@hydra.main(config_name="config", config_path=".", version_base=None)
def evaluate(cfg):
    cfg = parse_cfg(cfg)
    print("cfg:", cfg)

    # Load environment
    env = registry.load(cfg.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=cfg.episode_length)

    # Restore PPO networks
    obs_size = env.observation_size
    act_size = env.action_size
    network = ppo_networks.make_ppo_networks(obs_size, act_size)

    make_policy_fn = ppo_networks.make_inference_fn(network)

    # Load saved parameters
    save_dir = os.path.join(cfg.work_dir, "models")
    with open(os.path.join(save_dir, "ppo_params.pkl"), "rb") as f:
        normalizer_params, policy_params, value_params = pickle.load(f)

    policy = make_policy_fn((normalizer_params, policy_params, value_params))

    # Evaluation loop
    key = jax.random.PRNGKey(cfg.seed)
    state = env.reset(key)
    total_reward = 0.0

    for _ in range(cfg.episode_length):
        action = policy(state.obs)
        state = env.step(state, action)
        total_reward += state.reward

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    evaluate()

from collections import defaultdict
import functools
import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from algos.sac import SAC, SACTrainState
import distrax
import flax
import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from utils.termination_fns import get_termination_fn
import wandb
from flax.training.train_state import TrainState
from modules.dynamics import EnsembleDynamics, EnsembleDynamicsModel
from utils.data import Transition
from modules.common import MLP, DoubleCritic, TanhGaussianActor, VectorQ, ExpModule
from omegaconf import OmegaConf
from pydantic import BaseModel
from utils.data import load_d4rl_dataset, load_minari_dataset
from utils.logger import Logger
from utils.flax_utils import update_by_loss_grad
from algos.test_sac import EnsembleCritic, Actor, Alpha
from gymnasium.wrappers import NormalizeReward
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
os.environ["WANDB_SILENT"] = "true"
class RAMBOConfig(BaseModel):
    algo: str = "RAMBO"
    project: str = "q_learning"
    entity: str = "tjrcjf410-seoul-national-university"
    env_name: str = "HalfCheetah-v5"
    data_name : str = "halfcheetah-medium-v2"
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 10000
    eval_interval: int = 10000
    batch_size: int = 256
    max_steps: int = int(3e6)
    n_jitted_updates: int = 8
    use_wandb : bool = True
    log_dir: str = "./logs"
    debug : bool = False
    # DATASET
    d4rl : bool = True
    minari : bool = False
    custom : bool = False
    comment : str = ""
    
    # NETWORK
    actor_hidden_dims: List[int] = [256, 256]
    critic_hidden_dims: List[int] = [256, 256]
    n_critics : int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr : float =3e-4
    target_entropy : float = 0.
    tau: float = 0.005
    discount: float = 0.99
    alpha: float = 0.2

    # DYNAMICS NETWORK
    dynamics_hidden_dims : List[int] = [256, 256, 256, 256]
    dynamics_weight_decay :float = 2.5e-5
    dynamics_lr: float = 3e-4
    n_ensemble : int = 7
    n_elites : int = 5
    rollout_interval : int = 256
    dynamics_update_interval : int = 1e3
    adv_batch_size : int = 256
    rollout_batch_size : int = 50000
    rollout_length : int = 2
    adv_weight : float = 3e-4
    model_retain_steps : int = 5e4
    real_ratio : float = 0.5
    pretrain_epochs: int = 400
    lmbda: float = 3e-4
    dynamics_path : str = "RAMBO_2_halfcheetah-medium-v2_(0.0003, 2, 0.5)_d4rl_buf100_dynamics.msgpack"

    bc_path : str= None
    bc_pretrain : bool = False
    bc_epochs : int = 50
    bc_batch_size : int = 256


    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
cfg = RAMBOConfig(**conf_dict)
class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(self.params, self.target_params, tau)
        return self.replace(target_params=new_target_params)
    
class RAMBOTrainState(NamedTuple):
    dynamics: TrainState
    critic: CriticTrainState
    actor : TrainState
    alpha : TrainState

class RAMBO(SAC):
    def __init__(self, lmbda:float, tau: float, discount: float, target_entropy : float):
        self.lmbda = lmbda
        self.tau = tau
        self.discount = discount
        self.target_entropy = target_entropy
    def update_dynamics(
        self,
        train_state: RAMBOTrainState,
        dataset : Transition,
        rng: jax.random.PRNGKey,
        elite_idxs : jnp.ndarray,
        adv_batch_size : int,
        adv_rollout_length : int,
        termination_fn,
        n_jitted_updates,
    ) -> Tuple["RAMBOTrainState", jnp.ndarray]:
        supervised_losses = []
        adv_losses = []
        advs = []
        log_probs = []
        for _ in range(n_jitted_updates):
            rng, init_rng = jax.random.split(rng)
            init_indicies = jax.random.randint(
                    init_rng, (adv_batch_size,), 0, len(dataset.observations)
                )
            init_batch = jax.tree_util.tree_map(lambda x: x[init_indicies], dataset)
            observations = init_batch.observations
            s_loss = 0.
            a_loss = 0.
            adv = 0.
            log_prob = 0.
            new_dynamics = train_state.dynamics
            for __ in range(adv_rollout_length):
                rng, batch_rng, act_rng, loss_rng = jax.random.split(rng,4)
                batch_indices = jax.random.randint(
                        batch_rng, (adv_batch_size,), 0, len(dataset.observations)
                    )
                batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

                dist= train_state.actor.apply_fn(train_state.actor.params, observations, )
                actions, _ = dist.sample_and_log_prob(seed=act_rng)

                def get_supervised_loss(dynamics_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
                    means, logvar = train_state.dynamics.apply_fn(dynamics_params, jnp.concatenate([batch.observations, batch.actions], axis=1))
                    target = jnp.concatenate([batch.next_observations - batch.observations, batch.rewards.reshape(-1, 1)], axis=-1)
                    mse_loss = (((means - target) ** 2) * jnp.exp(-logvar)).mean(axis=(1,2))
                    # elite_idxs = jnp.argsort(mse_loss)[: n_elites]
                    var_loss = logvar.sum(0).mean()
                    max_logvar = dynamics_params["params"]["max_logvar"]
                    min_logvar = dynamics_params["params"]["min_logvar"]
                    logvar_diff = (max_logvar - min_logvar).sum()
                    supervised_loss = mse_loss.mean() + var_loss + 0.001 * logvar_diff
                    return supervised_loss

                def get_adv_loss(dynamics_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
                    index_rng, dynamics_rng, next_act_rng = jax.random.split(loss_rng,3 )
                    means, logvar = train_state.dynamics.apply_fn(dynamics_params, jnp.concatenate([observations, actions], axis=1))
                    #compute model next_ob, next_act
                    delta_obs = means[...,:-1]
                    rewards = means[...,-1:]
                    means = jnp.concatenate([delta_obs+observations, rewards], axis=-1)
                    dist = distrax.Normal(means,jnp.exp(0.5*logvar))
                    samples, _ = dist.sample_and_log_prob(seed=dynamics_rng)
                    elite_indices = jax.random.choice(index_rng,jnp.array(elite_idxs), (adv_batch_size,))
                    elite_samples = samples[elite_indices, jnp.arange(adv_batch_size)]
                    next_observations = elite_samples[..., :-1]
                    rewards = elite_samples[..., -1]
                    terminals = termination_fn(observations, actions, next_observations)

                    # log_prob = dist.log_prob(elite_samples.astype(jnp.float64)).sum(-1)
                    log_prob = dist.log_prob(elite_samples).sum(-1)

                    log_prob = log_prob[elite_idxs, :]
                    # log_prob = jax.nn.logsumexp(log_prob, axis=0) - jnp.log(len(elite_samples))
                    max_log_prob = jnp.max(log_prob, axis=0, keepdims=True)
                    log_prob = jax.nn.logsumexp(log_prob - max_log_prob, axis=0) + max_log_prob.squeeze() - jnp.log(len(elite_samples))
                    # log_prob = log_prob.astype(jnp.float32) 
                    #compute advantage
                    actor_dist = train_state.actor.apply_fn(
                        train_state.actor.params, next_observations
                    )
                    next_actions, _ = actor_dist.sample_and_log_prob(seed=next_act_rng)
                    qns = train_state.critic.apply_fn(
                        train_state.critic.params, next_observations, next_actions,
                    ).min(0)
                    value = rewards + (1-terminals) * self.discount * qns
                    qs = train_state.critic.apply_fn(
                        train_state.critic.params, observations, actions,
                    ).min(0)

                    advantage = value - qs
                    advantage = (advantage - advantage.mean()) / jnp.maximum(advantage.std(), 1e-6)#normalizing advantages
                    advantage = jax.lax.stop_gradient(advantage)
                    adv_loss = (log_prob * advantage).mean()
                    
                    return adv_loss, (jax.lax.stop_gradient(next_observations), advantage, jax.lax.stop_gradient(log_prob))
                def get_total_loss(dynamics_params: flax.core.FrozenDict[str, Any]):
                    supervised_loss = get_supervised_loss(dynamics_params)
                    adv_loss,(next_observations, advantage, log_prob)  \
                        = get_adv_loss(dynamics_params)
                    loss = supervised_loss + self.lmbda* adv_loss
                    return loss, (next_observations, advantage, log_prob, supervised_loss, adv_loss) 



                # new_dynamics, supervised_loss = \
                #     update_by_loss_grad(new_dynamics, get_supervised_loss)
                # adv_loss_fn = lambda dynamics_params: get_adv_loss(dynamics_params)
                # new_dynamics, adv_loss, (next_observations, advantage, log_prob) = update_by_loss_grad(new_dynamics, adv_loss_fn, has_aux=True)
                new_dynamics, loss, (next_observations, advantage, log_prob, supervised_loss, adv_loss) =\
                      update_by_loss_grad(new_dynamics, get_total_loss, has_aux=True)
                observations = next_observations
                s_loss += supervised_loss
                a_loss += adv_loss
                adv += advantage.mean()
                log_prob += log_prob.mean()
            supervised_losses.append(s_loss/adv_rollout_length)
            adv_losses.append(a_loss/adv_rollout_length)
            advs.append(adv/adv_rollout_length)
            log_probs.append(log_prob/adv_rollout_length)
            train_state = train_state._replace(dynamics = new_dynamics)
        s_losses = jnp.array(supervised_losses)
        a_losses = jnp.array(adv_losses)
        advantages = jnp.array(advs)
        l_probs = jnp.array(log_probs)
        loss_info = {
            "dynamics_loss" : (s_losses + a_losses).mean(),
            "dynamics_bc_loss" : s_losses.mean(),
            "dynamics_adv_loss" : a_losses.mean(),
            "dynamics_adv_advantage" : advantages.mean(),
            "dynamics_adv_log_prob" : l_probs.mean(),
        }
        return train_state, loss_info
    # def update_n_times(
    #     self,
    #     train_state: RAMBOTrainState,
    #     rng: jax.random.PRNGKey,
    #     dataset: Transition,
    #     rollout_buffer : Transition,
    #     batch_size : int = 256 ,
    #     real_ratio : float = 0.5,
    #     n_jitted : int = 8,
    # ) -> Tuple["RAMBOTrainState", Dict]:
    #     for _ in range(n_jitted):
    #         # rng, batch_rng1, batch_rng2, actor_rng, critic_rng, alpha_rng  = jax.random.split(rng, 6)
    #         # real_batch_size = int(batch_size * real_ratio)
    #         # fake_batch_size = batch_size - real_batch_size
    #         # real_batch_indices = jax.random.randint(
    #         #     batch_rng1, (real_batch_size,), 0, len(dataset.observations)
    #         # )
    #         # real_batch = jax.tree_util.tree_map(lambda x: x[real_batch_indices], dataset)
    #         # fake_batch_indices = jax.random.randint(
    #         #     batch_rng2, (fake_batch_size,), 0, len(rollout_buffer.observations)
    #         # )
    #         # fake_batch = jax.tree_util.tree_map(lambda x: x[fake_batch_indices], rollout_buffer)
    #         # batch = jax.tree_util.tree_map(
    #         #         lambda x,y: jnp.concatenate([x,y],axis=0), real_batch, fake_batch)
    #         rng, batch_rng, actor_rng, critic_rng, alpha_rng  = jax.random.split(rng, 5)
    #         batch_indices = jax.random.randint(
    #             batch_rng, (batch_size,), 0, len(dataset.observations)
    #         )
    #         batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)
    #         train_state, actor_loss, q_min, batch_entropy = self.update_actor(
    #             train_state, batch, actor_rng
    #         )
            
    #         train_state, alpha_loss = self.update_alpha(
    #             train_state, batch, alpha_rng
    #         )
    #         train_state, critic_loss, q_target = self.update_critic(
    #             train_state, batch, critic_rng
    #         )

    #     return train_state, {
    #         "critic_loss": critic_loss,
    #         "actor_loss": actor_loss,
    #         "alpha_loss" : alpha_loss,
    #         "alpha" : train_state.alpha.apply_fn(train_state.alpha.params),
    #         "q_min" : q_min.mean(),
    #         "batch_entropy": batch_entropy.mean(),
    #         "q_target": q_target.mean(),
    #     }
    def explore_action(
        self,
        train_state: RAMBOTrainState,
        observations: np.ndarray,
        rng : jax.random.PRNGKey,
    ):
        dist = train_state.actor.apply_fn(
            train_state.actor.params, 
            observations,
        )
        samples, log_probs = dist.sample_and_log_prob(seed=rng)
        return samples
    def update_actor_bc(
        self,
        train_state: RAMBOTrainState,
        batch : Transition,
        rng: jax.random.PRNGKey,
    ) -> Tuple["RAMBOTrainState", jnp.ndarray]:
        def get_bc_loss(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(
                actor_params, batch.observations,
            )
            actions = dist.mean()
            loss = jnp.mean(jnp.square(batch.actions - actions))

            return loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, get_bc_loss)
        return train_state._replace(actor=new_actor), actor_loss
    def behavior_cloning(
        self,
        train_state: RAMBOTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        n_jitted_updates,
        batch_size,
    ):
        for _ in range(n_jitted_updates):
            rng, bc_rng, batch_rng  = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(
                batch_rng, (batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, bc_loss = self.update_actor_bc(
                train_state, batch, bc_rng
            )
        return train_state, {
            "bc_loss": bc_loss,
        }
    def update_dynamics_pretrain(
        self,
        train_state: RAMBOTrainState,
        batch : Transition,
        rng: jax.random.PRNGKey,
    ):
        def get_pretrain_loss(dynamics_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            means, logvar = train_state.dynamics.apply_fn(dynamics_params, jnp.concatenate([batch.observations, batch.actions], axis=1))
            target = jnp.concatenate([batch.next_observations - batch.observations, batch.rewards.reshape(-1, 1)], axis=-1)
            mse_loss = ((means - target) ** 2) * jnp.exp(-logvar)
            mse_loss = mse_loss.sum(0).mean()
            var_loss = logvar.sum(0).mean()
            max_logvar = dynamics_params["params"]["max_logvar"]
            min_logvar = dynamics_params["params"]["min_logvar"]
            logvar_diff = (max_logvar - min_logvar).sum()
            loss = mse_loss + var_loss + 0.001 * logvar_diff
            return loss

        new_dynamics, dynamics_loss = update_by_loss_grad(train_state.dynamics, get_pretrain_loss)
        return train_state._replace(dynamics=new_dynamics), dynamics_loss
    def dynamics_pretraining(
        self,
        train_state: RAMBOTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        n_jitted_updates,
        batch_size,
        num_elites,
    ):
        def get_eval_loss(dynamics_params):
            means, logvar = train_state.dynamics.apply_fn(dynamics_params, jnp.concatenate([batch.observations, batch.actions], axis=1))
            target = jnp.concatenate((batch.next_observations - batch.observations, batch.rewards.reshape(-1, 1)), axis=-1)
            mse_loss = ((means - target) ** 2).mean(axis=(1,2))
            elite_idxs = jnp.argsort(mse_loss)[: num_elites]
            var_loss = logvar.sum(0).mean()

            return mse_loss.mean(), var_loss, elite_idxs
        for _ in range(n_jitted_updates):
            rng, batch_rng, dynamics_rng  = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(
                batch_rng, (batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

            train_state, dynamics_loss = self.update_dynamics_pretrain(
                train_state, batch, dynamics_rng
            )
        val_loss, var_loss, elite_idxs = get_eval_loss(train_state.dynamics.params)

        return train_state, elite_idxs, {
            "dynamics_pretrain_loss": dynamics_loss,
            "dynamics_pretrain_val_loss": val_loss,
            "dynamics_pretrain_var_loss": var_loss,
        }
def create_rambo_train_state(
    rng: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    cfg: RAMBOConfig,
) -> Tuple["RAMBOTrainState", nn.Module] :
    rng, actor_rng, critic_rng1, critic_rng2, dynamics_rng, alpha_rng = jax.random.split(rng, 6)
    # initialize actor
    action_dim = actions.shape[-1]
    state_dim = observations.shape[-1]
    dynamics_net = EnsembleDynamicsModel(
        obs_dim=state_dim,
        action_dim=action_dim,
        num_ensemble=cfg.n_ensemble,
        hidden_dims = cfg.dynamics_hidden_dims,
    )
    dynamics = TrainState.create(
        apply_fn = dynamics_net.apply,
        params=dynamics_net.init(dynamics_rng, jnp.concatenate([observations, actions])),
        tx=optax.adamw(learning_rate=cfg.dynamics_lr, eps=1e-5, weight_decay=cfg.dynamics_weight_decay),
    )
    action_dim = actions.shape[-1]
    actor_model = TanhGaussianActor(
        hidden_dim=256,
        action_dim=action_dim,
    )
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=optax.adam(learning_rate=cfg.actor_lr),
    )
    # initialize critic
    # critic_model = DoubleCritic(config.critic_hidden_dims)
    critic_model = VectorQ(num_critics=int(cfg.n_critics), hidden_dim=256, )
    # critic_model = VectorQ(config.n_critics, config.critic_hidden_dims, )
    critic = CriticTrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng1, observations, actions),
        target_params=critic_model.init(critic_rng2, observations, actions),
        tx=optax.adam(learning_rate=cfg.critic_lr),
    )
    alpha_module = ExpModule()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_rng),
        tx=optax.adam(learning_rate=cfg.alpha_lr)
    )
    return RAMBOTrainState(
        dynamics = dynamics,
        critic=critic,
        actor=actor,
        alpha=alpha,
    ), dynamics_net
    
 
def evaluate(
    policy_fn: Callable,
    env: gymnasium.Env,
    num_episodes: int,
    seed: int
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy_fn(observations=observation)
            observation, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            episode_return += reward
        episode_returns.append(episode_return)
    return np.mean(episode_returns) 

from utils.data import load_d4rl_dataset, load_minari_dataset
from utils.logger import Logger
from gymnasium.wrappers import NormalizeReward



if __name__ == "__main__":
    if cfg.d4rl:
        data_path= os.path.join("./d4rl_dataset", f"{cfg.data_name}.pkl")
        save_name = cfg.data_name.replace("/", "-")
        save_path = f'RAMBO_{cfg.n_critics}_{save_name}_({cfg.lmbda}, {cfg.rollout_length}, {cfg.real_ratio})_d4rl'
        save_path += f'_{cfg.comment}'
        dataset, env = load_d4rl_dataset(cfg.env_name, data_path)
    elif cfg.minari:
        save_name = cfg.data_name.replace("/", "-")
        save_path = f'RAMBO_{save_name}_minari'
        save_path += f'_{cfg.comment}'
        dataset, env = load_minari_dataset(cfg.data_name)
    elif cfg.custom:
        data_path = os.path.join("./custom_dataset", f"{cfg.env_name}_{cfg.data_policy}")
        # save_name = cfg.data_name.replace("/", "-")
        save_path = f'RAMBO_{cfg.env_name}_custom'
        save_path += f'_{cfg.comment}'
        dataset, env = load_d3rlpy_dataset(cfg.env_name, data_path)
    obs_mean = dataset.observations.mean(0, keepdims=True)
    obs_std = dataset.observations.std(0, keepdims=True) +1e-6
    dataset = dataset._replace(
        observations=(dataset.observations - obs_mean)/obs_std,
        next_observations = (dataset.next_observations - obs_mean)/obs_std
    )

    eval_env = NormalizeReward(env, cfg.discount, epsilon=1e-8)
    state_dim= env.observation_space.shape[0]
    action_dim= env.action_space.shape[0]
    if cfg.debug:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_disable_jit", True)
    rng = jax.random.PRNGKey(cfg.seed)

    rng, subkey = jax.random.split(rng)

    train_state, dynamics_net  = create_rambo_train_state(
        subkey,
        dataset.observations[0],
        dataset.actions[0],
        cfg,
    )
    cfg.target_entropy = -np.prod(env.action_space.shape)
    algo = RAMBO(lmbda= cfg.lmbda, tau=cfg.tau, discount=cfg.discount, target_entropy=cfg.target_entropy)
    bc_fn = jax.jit(algo.behavior_cloning, static_argnums=(3,4))
    dynamics_pretraining_fn = jax.jit(algo.dynamics_pretraining, static_argnums=(3,4,5))
    dynamics_fn = jax.jit(algo.update_dynamics, static_argnums=(4,5,6,7))
    # update_fn = jax.jit(algo.update_n_times, static_argnums=(4,5,6))
    update_fn = jax.jit(algo.update_n_times, static_argnums=(3,4))
    act_fn = jax.jit(algo.get_action)
    explore_fn = jax.jit(algo.explore_action)
    rollout_interval = int(cfg.rollout_interval // cfg.n_jitted_updates)
    eval_interval = int(cfg.eval_interval // cfg.n_jitted_updates)
    log_interval = int(cfg.log_interval // cfg.n_jitted_updates)
    dynamics_update_interval = int(cfg.dynamics_update_interval // cfg.n_jitted_updates)
    dataset_size = dataset.observations.shape[0]
    bc_steps = cfg.bc_epochs * dataset_size//cfg.batch_size//cfg.n_jitted_updates
    dynamics_steps = cfg.pretrain_epochs * dataset_size//cfg.batch_size//cfg.n_jitted_updates
    num_steps = cfg.max_steps//cfg.n_jitted_updates

    
    termination_fn = get_termination_fn(cfg.data_name)
    dynamics_model = EnsembleDynamics(
        dynamics_net, train_state.dynamics.params, termination_fn
    )
    rollout_fn = dynamics_model.make_rollout_fn(
        dataset = dataset,
        batch_size=cfg.rollout_batch_size,
        rollout_length=cfg.rollout_length,
    )
    max_buffer_size = 100 *cfg.rollout_batch_size * cfg.rollout_length
    rollout_buffer = jax.tree_util.tree_map(
        lambda x: jnp.zeros((max_buffer_size, *x.shape[1:])),
        dataset,
    )

    logger = Logger(save_path, cfg)
    # if cfg.use_wandb:
    #     wandb.define_metric("training/*", step_metric="step")
    #     wandb.define_metric("eval/*", step_metric="eval_step")
    #     wandb.define_metric("behavior_cloning/*", step_metric="bc_step")
    #     wandb.define_metric("pretraining/*", step_metric="pretrain_step")
    #     wandb.define_metric("dynamics_training/*", step_metric="dynamics_step")
    #     wandb.define_metric("rollout_info/*", step_metric="rollout_step")
    # bc updates
    if cfg.bc_pretrain:
        if cfg.bc_path is None:
            print("BC policy updates...")
            for i in tqdm.tqdm(range(1, bc_steps + 1), smoothing=0.1, dynamic_ncols=True):
                rng, bc_rng = jax.random.split(rng)
                train_state, update_info = bc_fn(
                    train_state,
                    dataset,
                    bc_rng,
                    cfg.n_jitted_updates,
                    cfg.batch_size
                )
                if i % log_interval == 0:
                    train_metrics = {f"behavior_cloning/{k}": v for k, v in update_info.items()}
                    # logger.log({**train_metrics, "bc_step":i * cfg.n_jitted_updates})
                    logger.log(train_metrics, step=i*cfg.n_jitted_updates)
            logger.save_policy(train_state.actor)
        else:
            train_state =  train_state._replace(actor = logger.load_policy(train_state.actor, cfg.bc_path))
    elite_idxs = jnp.arange(cfg.n_elites)
    # dynamics learning
    if cfg.dynamics_path is None:
        print("Dynamics updates...")
        for i in tqdm.tqdm(range(1, dynamics_steps + 1), smoothing=0.1, dynamic_ncols=True):
            rng, d_rng = jax.random.split(rng)
            train_state, elite_idxs, update_info = dynamics_pretraining_fn(
                train_state,
                dataset,
                d_rng,
                cfg.n_jitted_updates,
                cfg.batch_size,
                cfg.n_elites,
            )
            if i % log_interval == 0:
                train_metrics = {f"pretraining/{k}": v for k, v in update_info.items()}
                logger.log(train_metrics, step=i*cfg.n_jitted_updates)
                # logger.log({**train_metrics, "pretrain_step":i * cfg.n_jitted_updates})
        logger.save_dynamics(train_state.dynamics)
    else:
        rng, d_rng = jax.random.split(rng)
        train_state = train_state._replace(dynamics= logger.load_dynamics(train_state.dynamics, cfg.dynamics_path))
        _, elite_idxs, _ = dynamics_pretraining_fn(
                train_state,
                dataset,
                d_rng,
                cfg.n_jitted_updates,
                cfg.batch_size,
                cfg.n_elites,
            )
    # RAMBO updates
    print("RAMBO updates...")
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, step_rng = jax.random.split(rng)
        if i % rollout_interval == 0:
            step_rng, exp_rng, rng_buffer = jax.random.split(step_rng, 3)
            policy_fn = lambda obs, rng: explore_fn(train_state, obs, exp_rng)
            rollout_buffer, rollout_info = rollout_fn(rng_buffer, train_state.dynamics.params, elite_idxs, policy_fn, rollout_buffer)
            logger.log(rollout_info,step=i * cfg.n_jitted_updates)
            # logger.log({**rollout_info, "rollout_step":i * cfg.n_jitted_updates})
        if i % dynamics_update_interval ==0:
            step_rng, dynamics_rng = jax.random.split(step_rng, 2)
            for j in range(1, dynamics_update_interval + 1):
                train_state, dynamics_info = dynamics_fn(
                    train_state,
                    dataset,
                    dynamics_rng,
                    elite_idxs,
                    cfg.adv_batch_size,
                    cfg.rollout_length,
                    termination_fn,
                    cfg.n_jitted_updates,
                )
            train_metrics = {f"dynamics_training/{k}": v for k,v in dynamics_info.items()}
            logger.log(train_metrics, step=i*cfg.n_jitted_updates)
        step_rng, train_rng = jax.random.split(step_rng, 2)
        train_state, update_info = update_fn(
            train_state,
            train_rng,
            dataset,
            # rollout_buffer,
            cfg.batch_size,
            # cfg.real_ratio,
            cfg.n_jitted_updates,
        )

        if i % log_interval == 0:
            train_metrics = {f"training/{k}": v.mean() for k, v in update_info.items()}
            
            # logger.log({**train_metrics, "step":i*cfg.n_jitted_updates})
            logger.log(train_metrics, step=i*cfg.n_jitted_updates)
        if i % eval_interval == 0:
            step_rng, eval_rng = jax.random.split(step_rng, 2)

            policy_fn = partial(
                act_fn,
                rng=eval_rng,
                train_state=train_state,
            )
            normalized_score = evaluate(
                policy_fn, eval_env, cfg.eval_episodes, seed=cfg.seed,
            )
            print(i*cfg.n_jitted_updates, normalized_score)
            eval_metrics = {f"eval/normalized_score": normalized_score}
            # logger.log({**eval_metrics, "eval_step":i*cfg.n_jitted_updates})
            logger.log(eval_metrics, step=i*cfg.n_jitted_updates)

        
    # final evaluation
    rng, eval_rng = jax.random.split(rng)
    policy_fn = partial(
        act_fn,
        rng = eval_rng,
        train_state=train_state,
    )
    normalized_score = evaluate(policy_fn, eval_env, cfg.eval_episodes, seed=cfg.seed)
    print("Final evaluation score", normalized_score)
    logger.save_dynamics(train_state.dynamics, "_adv")
    logger.save_policy(train_state.actor)
    logger.finish()

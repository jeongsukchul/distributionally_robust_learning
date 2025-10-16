import os
from typing import Callable, NamedTuple
import hydra
import jax
import chex
import functools
import jax.numpy as jnp
from omegaconf import OmegaConf
from omegaconf import DictConfig


# now pass cfg to your factory
from learning.module.gmmvi.component_adaptation import ComponentAdaptationState, setup_vips_component_adaptation
from learning.module.gmmvi.configs import get_default_algorithm_config
from learning.module.gmmvi.gmm_setup import GMMWrapper, GMMWrapperState, setup_full_cov_gmm, setup_gmm_wrapper
from learning.module.gmmvi.ng_update import get_ng_update_fns
from learning.module.gmmvi.sample_db import SampleDB, SampleDBState, setup_sampledb
from learning.module.gmmvi.sample_selector import setup_fixed_sample_selector
from learning.module.gmmvi.stepsize_update import WeightStepsizeAdaptationState, update_component_stepsize_adaptive
from learning.module.gmmvi.weight_update import setup_weight_update_fn
from pathlib import Path
yaml_path = (Path(__file__).resolve().parent / "gmmvi.yaml")  # file-relative
cfg = OmegaConf.load(str(yaml_path))

class GMMTrainingState(NamedTuple):
    temperature: float
    num_updates: chex.Array
    model_state: GMMWrapperState
    sample_db_state: SampleDBState
    component_adaptation_state: ComponentAdaptationState
    weight_stepsize: int
    
class GMMNetwork(NamedTuple):
    model : GMMWrapper
    sample_db : SampleDB
    ng_estimator : Callable
    component_adapter : Callable
    component_updater : Callable
    sample_selector : Callable
    component_stepsize_fn : Callable
    weight_updater : Callable

def create_gmm_network_and_state(
    dim : int,
    batch_size : int,
    key : jax.random.PRNGKey
):  
    
    gmm = setup_full_cov_gmm(dim)
    
    gmm_state = gmm.init_gmm_state(key,
                                cfg.num_initial_components,
                                cfg.prior_mean,
                                cfg.prior_scale,
                                cfg.use_diagonal_convs,
                                cfg.init_std)
    model = setup_gmm_wrapper(gmm,
                            cfg.initial_stepsize,
                            cfg.initial_l2_regularizer,
                            10000)
    model_state = model.init_gmm_wrapper_state(gmm_state)
    sample_db = setup_sampledb(dim,
                            cfg.use_sample_database,
                            cfg.max_database_size,
                            cfg.use_diagonal_convs,
                            batch_size)
    sample_db_state = sample_db.init_sampleDB_state()

    # 'S' ; Stein Estimator +  'T' : Trust Rgeion Component Updater
    ng_update_fn, hass_grad_fn = get_ng_update_fns(model,dim,
                                            cfg.use_diagonal_convs,
                                            cfg.use_self_normalized_importance_weights,
                                            cfg.temperature,
                                            cfg.initial_l2_regularizer,
                                            )
    # 'A' vips component adaptation component increasing.
    component_adapter_state, component_adapter = setup_vips_component_adaptation(sample_db,
                        model,
                        dim,
                        cfg.prior_mean,
                        cfg.init_std ** 2,
                        cfg.use_diagonal_convs,
                        cfg.del_iters,
                        cfg.add_iters,
                        cfg.max_components,
                        cfg.thresholds_for_add_heuristic,
                        cfg.min_weight_for_del_heuristic,
                        cfg.num_database_samples)
    # 'Q' Fixed Sample Selector we can't use other because target log prob is expensive
    sample_selector = setup_fixed_sample_selector(sample_db,
                                                    model,
                                                    batch_size,
                                                    cfg.ratio_reused_samples_to_desired,
                                                    cfg.max_components)
    # 'R' Adaptive Component Stepsize update
    component_stepsize_fn = functools.partial(update_component_stepsize_adaptive, 
                                                MIN_STEPSIZE=cfg.min_stepsize,               
                                                MAX_STEPSIZE=cfg.max_stepsize,
                                                STEPSIZE_INC_FACTOR=cfg.stepsize_inc_factor,
                                                STEPSIZE_DEC_FACTOR=cfg.stepsize_dec_factor,)
    # 'U' Direct Update
    weight_update_fn = setup_weight_update_fn(model,
                                            cfg.temperature,
                                            cfg.use_self_normalized_importance_weights)
    # 'X' Weight Stepsize Adaptation
    # Fxied so not implemented
    initial_train_state = GMMTrainingState(temperature=cfg.temperature,
                                    num_updates=jnp.array([0]),
                                    model_state=model_state,
                                    sample_db_state=sample_db_state,
                                    component_adaptation_state=component_adapter_state,
                                    weight_stepsize=cfg.initial_stepsize)
    gmm_network = GMMNetwork(model = model,
            sample_db=sample_db,
            ng_estimator=hass_grad_fn, 
            component_adapter=component_adapter,
            component_updater=ng_update_fn,
            sample_selector=sample_selector,
            component_stepsize_fn=component_stepsize_fn,
            weight_updater=weight_update_fn)
    return initial_train_state, gmm_network
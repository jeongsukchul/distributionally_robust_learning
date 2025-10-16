from typing import NamedTuple, Callable
from jax import lax
from learning.module.gmmvi.utils import reduce_weighted_logsumexp
from learning.module.gmmvi.gmm_setup import GMMWrapperState, GMMWrapper
import chex
import jax.numpy as jnp
import jax



def setup_weight_update_fn(gmm_wrapper: GMMWrapper, TEMPERATURE, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS):

    # gmm_wrapper.log_densities_also_individual gmm_wrapper.store_rewards
    def _update_weights_from_expected_log_ratios(gmm_wrapper_state: GMMWrapperState,
                                                 expected_log_ratios: chex.Array, stepsize: jnp.float32):
        def true_fn(gmm_wrapper_state, stepsize, expected_log_ratios):
            unnormalized_weights = gmm_wrapper_state.gmm_state.log_weights + stepsize / TEMPERATURE * expected_log_ratios
            new_log_probs = unnormalized_weights - jax.nn.logsumexp(unnormalized_weights)
            new_log_probs = jnp.maximum(new_log_probs, -69.07)  # lower bound weights to 1e-30
            new_log_probs -= jax.nn.logsumexp(new_log_probs)
            return gmm_wrapper.replace_weights(gmm_wrapper_state, new_log_probs)

        return jax.lax.cond(gmm_wrapper_state.gmm_state.num_components > 1,
                            true_fn,
                            lambda gmm_wrapper_state, stepsize, expected_log_ratios: gmm_wrapper_state,
                            gmm_wrapper_state, stepsize, expected_log_ratios)
    @jax.jit
    def get_expected_log_ratios_and_update(gmm_wrapper_state: GMMWrapperState, samples, background_mixture_densities, target_lnpdfs, stepsize):

        model_densities, component_log_densities = jax.vmap(gmm_wrapper.log_densities_also_individual, in_axes=(None, 0))(gmm_wrapper_state.gmm_state, samples)
        component_log_densities = jnp.transpose(component_log_densities)  # transpose because of switched shape
        log_ratios = target_lnpdfs - TEMPERATURE * model_densities
        if USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS:
            log_weights = component_log_densities - background_mixture_densities
            log_weights -= jax.nn.logsumexp(log_weights, axis=1, keepdims=True)
            weights = jnp.exp(log_weights)
            importance_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
            expected_log_ratios = jnp.dot(importance_weights, log_ratios)
        else:
            n = jnp.array(jnp.shape(samples)[0], jnp.float32)
            log_importance_weights = component_log_densities - background_mixture_densities
            lswe, signs = reduce_weighted_logsumexp(
                log_importance_weights + jnp.log(jnp.abs(log_ratios)),
                w=jnp.sign(log_ratios), axis=1, return_sign=True)
            expected_log_ratios = 1 / n * signs * jnp.exp(lswe)

        component_rewards = TEMPERATURE * gmm_wrapper_state.gmm_state.log_weights + expected_log_ratios
        gmm_wrapper_state = gmm_wrapper.store_rewards(gmm_wrapper_state, component_rewards)
        gmm_wrapper_state = _update_weights_from_expected_log_ratios(gmm_wrapper_state, expected_log_ratios, stepsize)
        return gmm_wrapper_state
        # return gmm_wrapper_state, expected_log_ratios
    return get_expected_log_ratios_and_update



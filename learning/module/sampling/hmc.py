from typing import Dict, NamedTuple, Optional, Callable, Tuple, Protocol, Union

import chex
import jax
from learning.module.sampling.base import IntegratorState, Point, TransitionOperator, get_grad_intermediate_log_prob, get_intermediate_log_prob


LogProbFn = Callable[[chex.Array], chex.Array]

"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Tuple, NamedTuple, Optional
from functools import partial

import chex
import jax.numpy as jnp
import jax.random

import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.trajectory as trajectory
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.mcmc.integrators import EuclideanKineticEnergy



EuclideanIntegrator = Callable[[IntegratorState, float], IntegratorState]

class HMCState(NamedTuple):
    """State of the HMC algorithm. Save log_q and log_p instead of logdensity and logdensity_grad
    """

    position: ArrayTree
    log_q: chex.Array
    log_p: chex.Array
    grad_log_q: chex.Array
    grad_log_p: chex.Array
    beta: chex.Array
    alpha: float

    @property
    def logdensity(self) -> float:
        return get_intermediate_log_prob(log_q=self.log_q, log_p=self.log_p, beta=self.beta, alpha=self.alpha)

    @property
    def logdensity_grad(self) -> ArrayTree:
        return get_grad_intermediate_log_prob(grad_log_q=self.grad_log_q, grad_log_p=self.grad_log_p,
                                              beta=self.beta, alpha=self.alpha)


def velocity_verlet(
    log_q_fn: Callable,
    log_p_fn: Callable,
    kinetic_energy_fn: EuclideanKineticEnergy,
) -> EuclideanIntegrator:
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    logdensity_q_and_grad_fn = jax.value_and_grad(log_q_fn)
    logdensity_p_and_grad_fn = jax.value_and_grad(log_p_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, logdensity_grad = state.position, state.momentum, state.logdensity_grad

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        logdensity_q, logdensity_q_grad = logdensity_q_and_grad_fn(position)
        logdensity_p, logdensity_p_grad = logdensity_p_and_grad_fn(position)
        state = IntegratorState(position=position, momentum=momentum, log_q=logdensity_q,
                                log_p=logdensity_p, grad_log_q=logdensity_q_grad, grad_log_p=logdensity_p_grad,
                                beta=state.beta, alpha=state.alpha)
        logdensity_grad = state.logdensity_grad

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return state._replace(momentum=momentum)

    return one_step


class HMCInfo(NamedTuple):
    momentum: ArrayTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: IntegratorState
    num_integration_steps: int


def init(point: Point, beta: chex.Array, alpha: float):
    return HMCState(position=point.x, log_p=point.log_p, log_q=point.log_q, beta=beta,
                    grad_log_p=point.grad_log_p, grad_log_q=point.grad_log_q, alpha=alpha)


def kernel(
    integrator: Callable = velocity_verlet,
    divergence_threshold: float = 1000,
):

    def one_step(
        rng_key: PRNGKey,
        state: HMCState,
        log_q_fn: Callable,
        log_p_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
    ) -> Tuple[HMCState, HMCInfo]:
        """Generate a new sample with the HMC kernel."""

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
        symplectic_integrator = integrator(log_q_fn, log_p_fn, kinetic_energy_fn)
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        momentum = momentum_generator(key_momentum, state.position)

        integrator_state = IntegratorState(**state._asdict(), momentum=momentum)
        proposal, info = proposal_generator(key_integrator, integrator_state)
        hmc_state_kwargs = proposal._asdict()
        del(hmc_state_kwargs['momentum'])
        proposal = HMCState(**hmc_state_kwargs)
        return proposal, info

    return one_step


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, ArrayTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    build_trajectory = trajectory.static_integration(integrator)
    init_proposal, generate_proposal = proposal.proposal_generator(
        hmc_energy(kinetic_energy), divergence_threshold
    )

    def generate(
        rng_key, state: IntegratorState
    ) -> Tuple[IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def flip_momentum(
    state: IntegratorState
) -> IntegratorState:
    flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
    return state._replace(momentum=flipped_momentum)

class HMCState(NamedTuple):
    key: chex.PRNGKey
    inverse_mass_maxtric: chex.Array
    step_size: chex.Array


def build_blackjax_hmc(
                 dim: int,
                 n_outer_steps: int = 1,
                 n_inner_steps: int = 5,
                 init_step_size: float = 1e-4,
                 adapt_step_size: bool = True,
                 target_p_accept: float = 0.65,
                 step_size_multiplier: float = 1.02,
) -> TransitionOperator:

    one_step = kernel(divergence_threshold=1000)


    def init(key: chex.PRNGKey) -> HMCState:
        inverse_mass_matrix = jnp.ones(dim)
        return HMCState(key, inverse_mass_matrix, step_size=jnp.array(init_step_size))

    def apply(point: Point,
             transition_operator_state: HMCState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, HMCState, dict]:

        chex.assert_rank(point.x, 2)
        batch_size = point.x.shape[0]

        hmc_state = jax.vmap(init, in_axes=(0, None, None))(point, beta, alpha)

        def scan_fn(body, xs):
            key = xs
            key_batch = jax.random.split(key, batch_size)
            hmc_state, transition_operator_state = body
            step_fn_partial = partial(one_step,
                                      log_q_fn=log_q_fn,
                                        log_p_fn=log_p_fn,
                                        step_size=transition_operator_state.step_size,
                                        inverse_mass_matrix=transition_operator_state.inverse_mass_maxtric,
                                        num_integration_steps=n_inner_steps)
            hmc_state, info = jax.vmap(step_fn_partial)(key_batch, hmc_state)
            if adapt_step_size:
                step_size = jax.lax.cond(jnp.mean(info.acceptance_rate) > target_p_accept,
                                         lambda step_size: step_size * step_size_multiplier,
                                         lambda step_size: step_size / step_size_multiplier,
                                         transition_operator_state.step_size)
                transition_operator_state = transition_operator_state._replace(step_size=step_size)
            return (hmc_state, transition_operator_state), info


        key, subkey = jax.random.split(transition_operator_state.key)
        (hmc_state, transition_operator_state), infos = jax.lax.scan(
            scan_fn,
            (hmc_state, transition_operator_state),
            jax.random.split(subkey, n_outer_steps))

        # Info for logging
        info = {f"mean_acceptance_rate": jnp.mean(infos.acceptance_rate)}
        info.update(step_size=transition_operator_state.step_size)

        point_kwargs = hmc_state._asdict()
        del(point_kwargs['beta']); del(point_kwargs['alpha'])
        point_kwargs["x"] = point_kwargs["position"]
        del(point_kwargs['position'])
        point = Point(**point_kwargs)

        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, apply)

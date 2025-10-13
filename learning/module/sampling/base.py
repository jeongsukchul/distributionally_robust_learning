from asyncio import Protocol
from typing import Callable, NamedTuple, Optional, Union
import chex
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
import jax.numpy as jnp

class IntegratorState(NamedTuple):
    position: ArrayTree
    momentum: ArrayTree
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

LogProbFn = Callable[[chex.Array], chex.Array]


def get_intermediate_log_prob(
        log_q: chex.Array,
        log_p: chex.Array,
        beta: chex.Array,
        alpha: Union[chex.Array, float],
        ) -> chex.Array:
    """Get log prob of point according to intermediate AIS distribution.
    Set AIS final target g=p^\alpha q^(1-\alpha).
    log_prob = (1 - beta) log_q + beta log_g.
    """
    return ((1-beta) + beta*(1-alpha)) * log_q + beta*alpha*log_p

def get_grad_intermediate_log_prob(
        grad_log_q: chex.Array,
        grad_log_p: chex.Array,
        beta: chex.Array,
        alpha: Union[chex.Array, float],
) -> chex.Array:
    """Get gradient of intermediate AIS distribution for a point.
    Set AIS final target g=p^\alpha q^(1-\alpha). log_prob = (1 - beta) log_q + beta log_g.
    """
    return ((1-beta) + beta*(1-alpha)) * grad_log_q + beta*alpha*grad_log_p


class Point(NamedTuple):
    """State of the MCMC chain, specifically designed for FAB."""
    x: chex.Array
    log_q: chex.Array
    log_p: chex.Array
    grad_log_q: Optional[chex.Array] = None
    grad_log_p: Optional[chex.Array] = None

TransitionOperatorState = chex.ArrayTree

class TransitionOperatorStep(Protocol):
    def __call__(self,
             point: Point,
             transition_operator_state: TransitionOperatorState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn) -> Tuple[Point, TransitionOperatorState, Dict]:
        """Perform MCMC step with the intermediate target given by:
            \log target = ((1-beta) + beta*(1-alpha)) * log_q + beta*alpha*log_p
        """

class TransitionOperator(NamedTuple):
    init: Callable[[chex.PRNGKey], chex.ArrayTree]
    step: TransitionOperatorStep
    # Whether the transition operator uses gradients (True for HMC, False for metropolis).
    uses_grad: bool = True

    
class AISForwardFn(Protocol):
    def __call__(self, sample_q_fn: Callable[[chex.PRNGKey], chex.Array],
                 log_q_fn: LogProbFn, log_p_fn: LogProbFn,
                 ais_state: chex.Array) -> [chex.Array, chex.Array, chex.ArrayTree, Dict]:
        """

        Args:
            sample_q_fn: Sample from base distribution.
            log_q_fn: Base log density.
            log_p_fn: Target log density (note not the same as the AIS target which is p^2/q)
            ais_state: AIS state.

        Returns:
            x: Samples from AIS.
            log_w: Unnormalized log weights from AIS.
            ais_state: Updated AIS state.
            info: Dict with additional information.
        """

class PointIsValidFn(Protocol):
    def __call__(self, point: Point) -> bool:
        """
        Determines whether a point is valid or invalid. A point may be invalid if it contains
        NaN values under the target log prob. The user can provide any criternion for this function, which allows
        for enforcement of additional useful properties, such as problem bounds.
        See `default_point_is_valid` for the default version of this function.
        """

def default_point_is_valid_fn(point: Point) -> bool:
    chex.assert_rank(point.x, 1)
    is_valid = jnp.isfinite(point.log_q) & jnp.isfinite(point.log_p) & jnp.all(jnp.isfinite(point.x))
    return is_valid


def point_is_valid_if_in_bounds_fn(point: Point,
                                   min_bounds: Union[chex.Array, float],
                                   max_bounds: [chex.Array, float]) -> bool:
    """Returns True if a point is within the provided bounds. Must be wrapped with a partial to be
    used as a `PointIsValidFn`."""
    chex.assert_rank(point.x, 1)
    if isinstance(min_bounds, chex.Array):
        chex.assert_equal_shape(point.x, min_bounds, max_bounds)
    else:
        min_bounds = jnp.ones_like(point.x) * min_bounds
        max_bounds = jnp.ones_like(point.x) * max_bounds

    is_valid = (point.x > min_bounds).all() & (point.x < max_bounds).all()
    is_valid = is_valid & default_point_is_valid_fn(point)
    return is_valid
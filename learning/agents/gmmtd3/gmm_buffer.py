from typing import NamedTuple
import jax
import flax
import chex
import jax.numpy as jnp
@flax.struct.dataclass

class BufferState(NamedTuple):
    samples: chex.Array
    means: chex.Array
    chols: chex.Array
    inv_chols: chex.Array
    target_lnpdfs: chex.Array
    target_grads: chex.Array
    mapping: chex.Array
    num_samples_written: chex.Array

# class GMMBuffer:
#     def __init__(
#         self,
#         MAX_BUFFER_SIZE: int,
#         param_size: int,
#         sample_batch_size: int,
#     ):
#         self._MAX_BUFFER_SIZE=MAX_BUFFER_SIZE
#         self._size =0
#         self._sample_batch_size = sample_batch_size

#     def init(self, key: jax.random.PRNGKey) -> BufferState:
#         return BufferState(
#             samples=jnp.zeros(self._MAX_BUFFER_SIZE, param_size)

#         )
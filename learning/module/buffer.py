from brax.training import replay_buffers
from brax.training.replay_buffers import ReplayBufferState, UniformSamplingQueue, Sample
import jax
import jax.numpy as jnp
class DynamicBatchQueue(UniformSamplingQueue):
    def sample_batch(self, buffer_state: ReplayBufferState, batch_size: int):
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(sample_key, (batch_size,), 0, buffer_state.insert_position)
        batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
        return buffer_state.replace(key=key), self._unflatten_fn(batch)

    
# Curriculum Learning for RQS Flow: Gradual Transformation from Uniform Distribution

## Overview

This implementation adds curriculum learning to the Rational Quadratic Spline (RQS) flow, enabling a **true gradual transformation** from a uniform distribution to the learned distribution over training steps.

## How It Works

### 1. **Uniform Initialization**
- At training step 0, the RQS flow starts with a uniform distribution
- All bins have equal widths and heights
- Slopes are constant (close to identity transformation)

### 2. **Gradual Transition**
- Over `curriculum_steps` training steps, the transformation gradually becomes more complex
- Uses linear interpolation between uniform parameters and learned parameters
- The interpolation factor `alpha` goes from 0.0 (uniform) to 1.0 (fully learned)

### 3. **Mathematical Implementation**
```python
# Curriculum progress (0.0 to 1.0)
curriculum_progress = (training_step - curriculum_start_step) / curriculum_steps

# Interpolate between uniform and learned parameters
alpha = curriculum_progress
bin_w = (1 - alpha) * uniform_bin_w + alpha * learned_bin_w
bin_h = (1 - alpha) * uniform_bin_h + alpha * learned_bin_h
slopes = (1 - alpha) * uniform_slopes + alpha * learned_slopes
```

## Configuration

### Enable Curriculum Learning
```python
cfg.flow_network.use_curriculum = True
cfg.flow_network.curriculum_steps = 10000      # Transition over 10k steps
cfg.flow_network.curriculum_start_step = 0     # Start from step 0
```

### Parameters
- **`use_curriculum`**: Enable/disable curriculum learning
- **`curriculum_steps`**: Number of steps for gradual transition
  - Higher values = slower, smoother transition
  - Lower values = faster transition
- **`curriculum_start_step`**: Useful for resuming training

## Usage Examples

### 1. **Basic Curriculum Learning**
```python
# Use the provided config
python -m learning.train --config=learning/configs/ant_flowsac_curriculum.py --policy=flowsac
```

### 2. **Custom Curriculum Settings**
```python
# Modify the config in your training script
cfg.flow_network.use_curriculum = True
cfg.flow_network.curriculum_steps = 20000      # Slower transition
cfg.flow_network.curriculum_start_step = 5000  # Start curriculum at step 5000
```

### 3. **Disable Curriculum Learning**
```python
cfg.flow_network.use_curriculum = False
# Falls back to normal training without gradual transformation
```

## Benefits

1. **Stable Training**: Starting with uniform distribution prevents early training instability
2. **Controlled Complexity**: Gradually increasing transformation complexity
3. **Better Convergence**: Smoother learning trajectory
4. **Reproducible**: Deterministic transition from uniform to learned

## Training Behavior

- **Steps 0-1000**: Mostly uniform distribution (alpha ≈ 0.0)
- **Steps 1000-9000**: Gradual transition (alpha = 0.1 to 0.9)
- **Steps 9000+**: Fully learned transformation (alpha ≈ 1.0)

## Monitoring

You can monitor the curriculum progress by checking the `training_step` parameter passed to the flow network. The transformation smoothly interpolates between uniform and learned parameters based on this step.

## Notes

- The curriculum only affects the **forward pass** during training
- **Sampling** always uses the fully learned transformation
- **Evaluation** uses the current transformation (respects curriculum progress)
- The uniform initialization is mathematically sound and preserves the flow's invertibility 
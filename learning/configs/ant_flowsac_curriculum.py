# Configuration for FlowSAC with curriculum learning for gradual transformation
# This config enables the RQS flow to start with a uniform distribution and gradually
# learn more complex transformations over training steps

from learning.configs.ant_flowsac import get_config

def get_config():
    cfg = get_config()
    
    # Enable curriculum learning for gradual transformation
    cfg.flow_network.use_curriculum = True
    cfg.flow_network.curriculum_steps = 10000  # Gradual transition over 10k steps
    cfg.flow_network.curriculum_start_step = 0  # Start curriculum from step 0
    
    # You can adjust these parameters:
    # - curriculum_steps: Higher values = slower transition
    # - curriculum_start_step: Useful for resuming training with curriculum
    
    return cfg 
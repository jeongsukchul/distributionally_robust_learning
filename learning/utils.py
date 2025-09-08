import os
import shutil
import wandb
from helper import make_dir


def save_configs_to_wandb_and_local(cfg, work_dir):
    """
    Save config.yaml and shift_dynamics config to both wandb (as artifacts) and local directory.
    
    Args:
        cfg: Configuration object containing task, shift_dynamics, shift_dynamics_type
        work_dir: Working directory path where configs should be saved locally
    """
    config_yaml_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    cfg_dir = make_dir(work_dir / "cfg")
    
    # Save config.yaml locally and to wandb
    _save_config_yaml(config_yaml_path, cfg_dir, cfg.use_wandb)
    
def _save_config_yaml(config_yaml_path, cfg_dir, use_wandb):
    """Save config.yaml to local directory and wandb."""
    if os.path.exists(config_yaml_path):
        # Save locally
        shutil.copy(config_yaml_path, os.path.join(cfg_dir, 'config.yaml'))
        print(f"Saved config.yaml to {cfg_dir}")
        
        # Save to wandb
        if use_wandb:
            artifact = wandb.Artifact('config', type='config')
            artifact.add_file(config_yaml_path)
            wandb.log_artifact(artifact)
            print("Uploaded config.yaml to wandb")


def _save_shift_dynamics_config(cfg, cfg_dir):
    """Save shift_dynamics config to local directory and wandb."""
    if cfg.shift_dynamics_type == "stochastic":
        dyn_path = os.path.join(os.path.dirname(__file__), '..', 'shift_dynamics', 'stochastic', f'{cfg.task}.yaml')
    elif cfg.shift_dynamics_type == "deterministic":
        dyn_path = os.path.join(os.path.dirname(__file__), '..', 'shift_dynamics', 'deterministic', f'{cfg.task}.yaml')
    else:
        print(f"Unknown dynamics shift type: {cfg.shift_dynamics_type}")
        return
    
    if dyn_path and os.path.exists(dyn_path):
        # Save locally
        local_path = os.path.join(cfg_dir, f'{cfg.shift_dynamics_type}_{cfg.task}.yaml')
        shutil.copy(dyn_path, local_path)
        print(f"Saved shift_dynamics config to {local_path}")
        
        # Save to wandb
        if cfg.use_wandb:
            dyn_artifact = wandb.Artifact('shift_dynamics', type='config')
            dyn_artifact.add_file(dyn_path)
            wandb.log_artifact(dyn_artifact)
            print("Uploaded shift_dynamics config to wandb")
    else:
        print(f"Dynamics shift config not found at: {dyn_path}") 
gpu_id=$1 
wandb_project="dr-effect"
task="CheetahRun"
seed=$2 
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task
# 

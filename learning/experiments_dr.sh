gpu_id=$1 
wandb_project="dr-effect"
task="CheetahRun"
use_wandb=false
seed=$2 
<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task eval_randomization=true use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=false use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=false use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task eval_randomization=true use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=false use_wandb=$use_wandb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=false use_wandb=$use_wandb
=======
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task eval_randomization=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project shift_dynamics=false task=$task eval_randomization=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=true
>>>>>>> 07d70da373f3a763ed5ec8423e7d15828b22024e
# 

gpu_id=$1 
wandb_project="Go1Handstand"
task="Go1Handstand"
use_wandb=true
seed=$2 
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project randomization=false task=$task eval_randomization=false use_wandb=$use_wandb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=true use_wandb=$use_wandb asymmetric_critic=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" seed=$seed wandb_project=$wandb_project randomization=false task=$task eval_randomization=false use_wandb=$use_wandb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=true use_wandb=$use_wandb asymmetric_critic=true

# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project randomizatoin=false task=$task eval_randomization=true use_wandb=$use_wandb
# # CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=true use_wandb=$use_wandb asymmetric_critic=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project randomization=false task=$task eval_randomization=false use_wandb=$use_wandb
# # CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project randomization=false task=$task eval_randomization=true use_wandb=$use_wandb
# # CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=true task=$task eval_randomization=true use_wandb=$use_wandb asymmetric_critic=true
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" seed=$seed wandb_project=$wandb_project custom_wrapper=false task=$task eval_randomization=true use_wandb=$use_wandb asymmetric_critic=true

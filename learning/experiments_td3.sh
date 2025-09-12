gpu_id=$1 
wandb_project="td3-algorithm"
seed=$2
use_wandb=true

dr_train_ratio=0.9
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                        #  wandb_project=$wandb_project randomization=false eval_randomization=false asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                         wandb_project=$wandb_project asymmetric_critic=true task="CheetahRun" seed=$seed use_wandb=$use_wandb 
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                         wandb_project=$wandb_project asymmetric_critic=true task="CheetahRun" seed=$seed use_wandb=$use_wandb 
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                         wandb_project=$wandb_project asymmetric_critic=true task="CartpoleSwingup" seed=$seed use_wandb=$use_wandb 
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                         wandb_project=$wandb_project asymmetric_critic=true task="CartpoleSwingup" seed=$seed use_wandb=$use_wandb 
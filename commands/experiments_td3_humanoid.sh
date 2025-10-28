gpu_id=$1 
wandb_project="td3-humanoid"
use_wandb=true
task="T1JoystickRoughTerrain"
dr_train_ratio=1.0
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=true  task=$task seed=$seed use_wandb=$use_wandb 
done


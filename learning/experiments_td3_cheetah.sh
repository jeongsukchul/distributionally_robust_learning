gpu_id=$1 
wandb_project="td3-custom_wrapper-test-cheetah4"
use_wandb=true
task="CheetahRun"
dr_train_ratio=1.0
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=true adv_wrapper=true task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=true adv_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done

wandb_project="sac-custom_wrapper-cheetah-test4"
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=true adv_wrapper=true task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=true adv_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true adv_wrapper=false  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb 
done
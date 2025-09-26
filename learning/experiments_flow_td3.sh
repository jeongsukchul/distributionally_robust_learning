gpu_id=$1 
wandb_project="flow-td3-cartpole7"
use_wandb=true
dr_train_ratio=1.0
task="CartpoleSwingup"
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project randomization=false eval_randomization=false task=$task seed=$seed use_wandb=$use_wandb
# # CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="flowtd3" flow_lr=1e-4  \
# #      wandb_project=$wandb_project asymmetric_critic=true init_lmbda=0.05 task=$task \
#     # seed=$seed use_wandb=$use_wandb dr_train_ratio=$dr_train_ratio 
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=true task=$task seed=$seed use_wandb=$use_wandb

CUDA_VISIBLE_DEVICES=0 python train.py policy="flowtd3" flow_lr=1e-4  \
    lambda_update_steps=10  wandb_project=$wandb_project asymmetric_critic=true init_lmbda=0.5 task="CartpoleSwingup" \
    seed=0 use_wandb=false dr_train_ratio=1.0 
for flow_lr in 1e-5 1e-4
do
    for init_lmbda in 0.5 1.0 2.5 5.
        do
        for seed in 1 2 3 
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="flowtd3" flow_lr=$flow_lr  \
                lambda_update_steps=10  wandb_project=$wandb_project asymmetric_critic=true init_lmbda=$init_lmbda task=$task \
                seed=$seed use_wandb=$use_wandb dr_train_ratio=$dr_train_ratio 
            done
        done 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=true custom_wrapper=false task=$task seed=$seed use_wandb=$use_wandb
done
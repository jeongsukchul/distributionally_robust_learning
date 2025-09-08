gpu_id=$1 
wandb_project="flow-cartpole"
seed=$2
use_wandb=false

dr_flow=true
dr_train_ratio=0.9
task="CartpoleSwingup"
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" simba=true\
                        #  wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
for flow_lr in 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
do
    for init_lmbda in 0.01 0.1 1 10
        do
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="flowsac" delta=0 flow_lr=$flow_lr single_lambda=true \
            lambda_update_steps=10 lmbda_lr=0. wandb_project=$wandb_project asymmetric_critic=true init_lmbda=$init_lmbda task=$task \
            seed=$seed use_wandb=$use_wandb dr_flow=$dr_flow dr_train_ratio=$dr_train_ratio simba=true
        done
done
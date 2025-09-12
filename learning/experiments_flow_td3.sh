gpu_id=$1 
wandb_project="flow-cartpole"
seed=$2
use_wandb=true
dr_train_ratio=0.9
task="CartpoleSwingup"
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
for flow_lr in 1e-5 5e-5 #1e-4 5e-4 1e-3 5e-3 1e-2
do
    for init_lmbda in 0.05 0.1 0.5 1.0
        do
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="flowsac"flow_lr=$flow_lr  \
            lambda_update_steps=10  wandb_project=$wandb_project asymmetric_critic=true init_lmbda=$init_lmbda task=$task \
            seed=$seed use_wandb=$use_wandb dr_flow=$dr_flow dr_train_ratio=$dr_train_ratio eval_with_training_env=true
        done
done
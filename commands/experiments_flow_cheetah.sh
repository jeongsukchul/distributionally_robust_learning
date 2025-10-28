gpu_id=$1 
wandb_project="wdtd3-panelty-cheetah6"
use_wandb=true
task="CheetahRun"
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="ppo" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
#                          wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
                        
for flow_lr in 1e-5 5e-5 #1e-4 5e-4 1e-3 5e-3 1e-2
do
    for init_lmbda in 0.01 0.05 0.1 0.5 1.0 2.5
        do
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="flowtd3" delta=0 flow_lr=$flow_lr  \
            wandb_project=$wandb_project asymmetric_critic=false init_lmbda=$init_lmbda task=$task \
             use_wandb=$use_wandb  
        done
done
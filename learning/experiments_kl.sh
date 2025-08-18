gpu_id=$1 
wandb_project="wdsac-kl"
seed=$2
use_wandb=true
task="CheetahRun"
# CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
for delta in 0.001 0.05 0.1 0.5 1
do
    for n_nominals in 5 10 15
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdsac" n_nominals=$n_nominals delta=$delta single_lambda="False" \
                lambda_update_steps=10 distance_type="kl" wandb_project=$wandb_project asymmetric_critic=true init_lmbda=10 task=$task \
                seed=$seed use_wandb=$use_wandb
        done
done
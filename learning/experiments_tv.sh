gpu_id=$1 
wandb_project="wdtd3-kl2"
use_wandb=true
task="CheetahRun"

for delta in 0.001 0.05 0.1 0.5 1
do
    for n_nominals in 5 10 15
        do
            for seed in 1 2 3
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdtd3" n_nominals=$n_nominals delta=$delta single_lambda=false \
                    lambda_update_steps=10 distance_type="tv" wandb_project=$wandb_project asymmetric_critic=false init_lmbda=10 task=$task \
                    seed=$seed use_wandb=$use_wandb
            done
        done
done
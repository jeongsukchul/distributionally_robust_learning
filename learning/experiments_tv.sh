gpu_id=$1 
wandb_project="wdsac-tv"

for delta in 0 0.001 0.05 0.1 0.5 1
do
    for n_nominals in 10 15 25 50
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdsac" n_nominals=$n_nominals delta=$delta single_lambda="False" \
                lambda_update_steps=10 distance_type="tv" wandb_project=$wandb_project  init_lmbda=0.1
        done
done
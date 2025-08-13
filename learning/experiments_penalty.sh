gpu_id=$1
wandb_project="wdsac-penalty"
CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" wandb_project=$wandb_project

for init_lmbda in 0.0001 0.0005 0.001 0.05 0.1 0.5 1
for n_nominals in 10 15 25 40 50
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdsac" n_nominals=10 delta=0 lmbda_lr=0 single_lambda="True" \
         lambda_update_steps=1 init_lmbda=$init_lmbda wandb_project=$wandb_project
done
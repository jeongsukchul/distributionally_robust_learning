# 0.01 0.1 1 5 10 20 50 100 200 500
for delta in 0 0.001 0.05 0.1 0.5 1
do
    for n_nominals in 10 15 25 50
        do
            CUDA_VISIBLE_DEVICES=5 python train.py policy="wdsac" n_nominals=$n_nominals delta=$delta single_lambda="False" \
                lambda_update_steps=100
        done
done
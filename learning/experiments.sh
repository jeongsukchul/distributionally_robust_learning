q_agg="mean"
# 0.01 0.1 1 5 10 20 50 100 200 500
for real_ratio in 0.9 0.95 1. 
do
    CUDA_VISIBLE_DEVICES=3 python train.py real_ratio=$real_ratio
done
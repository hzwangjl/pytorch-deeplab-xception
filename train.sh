CUDA_VISIBLE_DEVICES=0 python train.py \
    --backbone mobilenet \
    --lr 0.01 \
    --workers 8 \
    --base-size 480 \
    --crop-size 480 \
    --epochs 150 \
    --batch-size 8 \
    --checkname deeplab-mobilenet \
    --eval-interval 1 \
    --dataset mydataset

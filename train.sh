CUDA_VISIBLE_DEVICES=0 python train.py \
    --backbone mobilenet \
    --lr 0.01 \
    --workers 12 \
    --out-stride 16 \
    --base-size 240 \
    --crop-size 240 \
    --epochs 200 \
    --batch-size 8 \
    --checkname deeplab-mobilenet \
    --eval-interval 1 \
    --dataset mydataset \
    2>&1| tee ${LOG_FILE} &

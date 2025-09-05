# NOTE: if you want to checkout the logs, just run 'tensorboard --logdir ./logs'

CUDA_VISIBLE_DEVICES=0 \
python MAE/run.py \
    --dataset_name cifar10 \
    --output_dir ./vit-mae-demo-per_image-5ep_test \
    --logging_dir ./logs \
    --report_to tensorboard \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --bt_variant none \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --bt_variant per_batch # none, per_batch or per_image
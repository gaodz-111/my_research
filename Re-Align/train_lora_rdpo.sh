lr=1e-6
beta=0.1

deepspeed --include=localhost:5 --master_port 60000 train_rdpo.py \
    --model_name_or_path /data2/gaodz/llava-v1.6-vicuna-7b \
    --data_path "./preference_data/pref_data.json" \
    --deepspeed "./deepspeed/zero2.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --lora_enable True \
    --beta $beta \
    --output_dir "/data2/gaodz/llava-vicuna-7b-rdpo-lora-$lr-beta-$beta" \

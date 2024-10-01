export TOKENIZERS_PARALLELISM=false

lr_B=${1:-0.0001}
lr_A=${2:-0.0001}
seed=${3:-0}
export CUDA_VISIBLE_DEVICES=${4:-0}

task=mnli
exp_name=llama-3.1-8b_lora_${task}_seed_${seed}
# target=q_proj,k_proj,v_proj
target=q_proj,v_proj
lr_ratio=$(awk -v lrB="$lr_B" -v lrA="$lr_A" 'BEGIN{ print lrB / lrA }')


python /.../LightweightAtt/run.py \
  --model_name_or_path  /share/shared_models/llama-3.1-8b \
  --cache_dir /share/shared_models \
  --task_name $task \
  --use_lora \
  --target_modules $target \
  --lr_scheduler_type constant \
  --optim adamw_torch \
  --learning_rate $lr_A \
  --lr_ratio $lr_ratio \
  --adam_beta2 0.999 \
  --weight_decay 0.0 \
  --max_seq_length 128 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 2 \
  --lora_rank 16 \
  --num_train_epochs 1 \
  --max_steps 800 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --logging_steps 10 \
  --logging_strategy steps \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --dataloader_num_workers 8 \
  --gradient_checkpointing \
  --fp16 \
  --report_to tensorboard\
  --keep_checkpoints eval \
  --ignore_mismatched_sizes \
  --seed $seed  \
  --use_local True \
  --data_dir /.../LightweightAtt/data  \
  --output_dir output/$exp_name/lrB_${lr_B}_lrA_${lr_A}_${target} \
  --logging_dir output/$exp_name/lr-${lr_A}_ratio-${lr_ratio}/logs/ \
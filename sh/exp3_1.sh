export TOKENIZERS_PARALLELISM=false

lr_B=${1:-0.0001}
task=${2:-mnli}
seed=${3:-0}
export CUDA_VISIBLE_DEVICES=${4:-0}

lr_A=1e-4 #used in Section 5.
lora_rank=16
use_lora=True
exp_name=roberta_${use_lora}_${task}_seed_${seed}_${lora_rank}
# target=query,value,key
target=query,value
lr_ratio=$(awk -v lrB="$lr_B" -v lrA="$lr_A" 'BEGIN{ print lrB / lrA }')

python /.../LightweightAtt/run.py \
  --model_name_or_path roberta-base \
  --task_name $task \
  --use_lora $use_lora\
  --target_modules $target \
  --do_train  True\
  --do_predict False \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 128 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --num_train_epochs 10 \
  --learning_rate $lr_A \
  --lr_ratio $lr_ratio \
  --lora_rank $lora_rank \
  --lora_alpha 8 \
  --lr_scheduler_type 'linear' \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --output_dir output2/$exp_name/lrB_${lr_B}_lrA_${lr_A}_${target}  \
  --logging_dir output2/$exp_name/lr-${lr_A}_ratio-${lr_ratio}/logs/ \
  --evaluation_strategy steps \
  --save_strategy steps \
  --report_to tensorboard \
  --keep_checkpoints eval \
  --overwrite_output_dir \
  --ignore_mismatched_sizes \
  --save_total_limit 1 \
  --use_local True \
  --data_dir /.../LightweightAtt/data \
  --fp16 \
  --seed $seed  \
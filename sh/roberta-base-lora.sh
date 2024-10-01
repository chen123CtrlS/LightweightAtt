#!/bin/bash

# Environment setup  # roberta-base-lora.sh
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
# lr_Q/K = [1e-5,2e-5,5e-5,1e-4,2e-4,4e-4,8e-4]
# lr_V = [1e-4,2e-4,5e-4,1e-3,2e-3,4e-3]
#[10,20,50,100,200,400]
# [5,10,25,50,100,200]
# Experiment configuration
task=mnli
exp_name=roberta_lora_$task
lr=4e-4
lr_ratio=2.5
target=query,value,key
#query,value,key

# Execute command  
python /.../LightweightAtt/run.py \
  --model_name_or_path roberta-base \
  --task_name $task \
  --use_lora True\
  --target_modules $target \
  --do_train  True\
  --do_predict False \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 128 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --num_train_epochs 30 \
  --learning_rate $lr \
  --lr_ratio $lr_ratio \
  --lora_rank 8 \
  --lora_alpha 8 \
  --lr_scheduler_type 'linear' \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --output_dir output2/$exp_name/lr-${lr}_ratio-${lr_ratio}-${target} \
  --logging_dir output2/$exp_name/lr-${lr}_ratio-${lr_ratio}/logs/ \
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
import warnings
warnings.filterwarnings("ignore")

import os


from Arguments import ModelArguments, DataTrainingArguments, ATT_TrainingArguments,task_to_keys,DEFAULT_PAD_TOKEN
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EvalPrediction, HfArgumentParser, LlamaTokenizer,
                          PretrainedConfig, default_data_collator, set_seed)
from transformers.utils import send_example_telemetry
from datasets import load_dataset, load_from_disk

from utils import train_model,cleanup_checkpoints,smart_tokenizer_and_embedding_resize
import random
import sys
import datasets
# import evaluate
from peft import LoraConfig, get_peft_model, AdaLoraConfig
import numpy as np
import torch
import evaluate
from transformers.trainer_utils import get_last_checkpoint
from MyTrainer import MyTrainer


if __name__ == "__main__":
# TODO: Arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, ATT_TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Print each process the small summary:
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    print(f"Training/evaluation parameters {training_args}")

    print(f'data_args {data_args}')

# TODO: Load the dataset
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.use_local:
            raw_datasets = load_from_disk(
                os.path.join(data_args.data_dir, "glue", data_args.task_name)
            )
            # datasets.set_caching_enabled(False)
            datasets.disable_caching()
        else:
            raw_datasets = load_dataset(
                "glue",
                data_args.task_name,
                cache_dir=model_args.cache_dir,
                token=True if model_args.token else None,
            )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=True if model_args.token else None,
        )
    else:
        raise ValueError(
                    "See more about loading any type of standard or custom dataset at https://huggingface.co/docs/datasets/loading_datasets.html."
                )
#TODO: Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        raise ValueError(
                    "No task_name"
                )

#TODO: Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )

    torch_dtype = (
        torch.float32
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if "roberta" in config.name_or_path:
        device_map = None
    else:
        device_map= 'auto' #muti gpus load
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path), #Load from TensorFlow checkpoint
        torch_dtype=torch_dtype,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        device_map=device_map, 
    )
    # for name, param in model.named_parameters():
    #     print(f"{name} is on {param.device}")
    # exit()
    
    
    if "llama" in config.name_or_path or isinstance(tokenizer, LlamaTokenizer):
        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is None, `<unk>` token is id 0.
        print("Adding special tokens.")
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": "<unk>",
            }
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.use_lora:
        target_modules = training_args.target_modules
        assert target_modules is not None
        target_modules = target_modules.split(",")
        target_modules = [target_module.strip() for target_module in target_modules]
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            fan_in_fan_out=True,
            target_modules=target_modules,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=["classifier", "score"],
        )

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    else:
        target_modules = training_args.target_modules
        assert target_modules is not None
        target_modules = target_modules.split(",")
        target_modules = [target_module.strip() for target_module in target_modules]
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ttt in target_modules:
                if ttt in name or "classifier" in name:
                    param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")
    
    # Manually freeze dense layer in RoBERTa
    # if "roberta" in config.name_or_path:
    #     for name, param in model.named_parameters():
    #         if "classifier" in name and "dense" in name:
    #             param.requires_grad = False

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False


    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))


        # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    

#TODO: Training
    if training_args.do_train:
        train_model(
            trainer,
            training_args,
            data_args,
            train_dataset,
            last_checkpoint=last_checkpoint,
        )


    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(valid_mm_dataset), data_args.max_eval_samples
                )
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )

    if training_args.do_predict:
        print("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    print(f"***** Predict results {task} *****")
                    writer.write(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # cleanup checkpoints
    cleanup_checkpoints(trainer, training_args)


import sys
import traceback
import shutil
from typing import Dict

import transformers




def attempt_train(trainer, checkpoint):
    """
    Attempts to start or resume training from a given checkpoint.

    Args:
        trainer: The Hugging Face Trainer instance.
        checkpoint: The checkpoint path to resume from, or None to start afresh.

    Returns:
        The training result if successful, None otherwise.
    """
    try:
        return trainer.train(resume_from_checkpoint=checkpoint), True
    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error resuming from checkpoint {checkpoint}: {e}")
        print("Full Traceback:")
        print(traceback.format_exc())
        return None, False


def find_valid_checkpoint(trainer, training_args):
    """
    Finds a valid checkpoint to resume from if the initial checkpoint fails.

    Args:
        trainer: The Hugging Face Trainer instance.
        training_args: TrainingArguments used for training configuration.

    Returns:
        The training result after resuming from a valid checkpoint, or starts afresh if none found.
    """
    checkpoint_dirs = trainer._sorted_checkpoints(
        use_mtime=False, output_dir=training_args.output_dir
    )
    for checkpoint in reversed(
        checkpoint_dirs[1:]
    ):  # Skip the most recent as it's already attempted
        print(f"Attempting to resume from {checkpoint}")
        train_result, loaded = attempt_train(trainer, checkpoint)
        if loaded:
            return train_result
    print("No valid checkpoint found, starting from scratch.")
    return trainer.train(resume_from_checkpoint=None)


def train_model(trainer, training_args, data_args, train_dataset, last_checkpoint=None):
    """
    Manages the training process, handling resumption from checkpoints and errors.

    Args:
        trainer: The Hugging Face Trainer instance.
        training_args: TrainingArguments for training configuration.
        data_args: DataTrainingArguments for data configuration.
        train_dataset: The dataset used for training.
    """
    checkpoint = (
        training_args.resume_from_checkpoint or last_checkpoint
    )  # Simplified conditional assignment
    train_result, loaded = attempt_train(trainer, checkpoint)

    if not loaded:
        train_result = find_valid_checkpoint(trainer, training_args)

    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples or len(
        train_dataset
    )  # Use or for default value
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Save the model and tokenizer
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()



def delete_checkpoint(checkpoint_dir):
    """
    Deletes a specified checkpoint directory.

    Args:
        checkpoint_dir: The path to the checkpoint directory to delete.
    """
    try:
        print(f"Deleting checkpoint {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        print("Deletion successful.")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_dir} does not exist.")

def cleanup_checkpoints(trainer, training_args):
    """
    Cleans up checkpoints based on training arguments.

    Args:
        trainer: The Hugging Face Trainer instance.
        training_args: TrainingArguments used for training configuration.
    """
    print("Cleaning up checkpoints!")
    output_dir = training_args.output_dir
    checkpoints_sorted = trainer._sorted_checkpoints(
        use_mtime=False, output_dir=output_dir
    )
    print(f"Current checkpoints: {checkpoints_sorted}")

    if training_args.keep_checkpoints == "none":
        # Delete the entire output directory
        try:
            print(f"Deleting all checkpoints in {output_dir}")
            shutil.rmtree(output_dir)
            print("Deletion successful.")
        except FileNotFoundError:
            print(f"Directory {output_dir} does not exist.")
    elif training_args.keep_checkpoints == "eval":
        # Delete individual checkpoints but preserve evaluation results
        # Assumes checkpoints and evaluation results are stored separately
        for checkpoint in checkpoints_sorted:
            delete_checkpoint(checkpoint)

    print("Successful completion.")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings_data = output_embeddings.weight.data
            output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
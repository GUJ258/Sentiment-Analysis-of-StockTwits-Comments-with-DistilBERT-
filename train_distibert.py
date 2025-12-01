import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from config import *
from data_utils import load_full_dataset, split_model_and_perf, prepare_model_df
from hf_dataset import build_hf_dataset
from models import load_tokenizer, load_model

# Select device: MPS (Apple GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
    )


def main():
    # Load and prepare dataset
    df = load_full_dataset()
    model_df, _ = split_model_and_perf(df, ratio=0.8)
    model_df, label_encoder = prepare_model_df(model_df)

    dataset = build_hf_dataset(model_df, test_size=0.2, seed=SEED)

    # Load tokenizer and model
    tokenizer = load_tokenizer()
    model = load_model(len(label_encoder.classes_))
    model.to(device)  # Move model to MPS

    # Tokenize dataset
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    # HuggingFace Trainer arguments (with MPS enabled)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOG_DIR,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_mps_device=True,  # <-- IMPORTANT: enable Apple Silicon GPU
    )

    # Compute accuracy
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished.")


if __name__ == "__main__":
    main()

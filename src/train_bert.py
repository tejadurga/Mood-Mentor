import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.bert_model import load_bert_model, load_bert_tokenizer
from src.bert_training import create_emotion_dataset


TRAIN_DATA_PATH = "data/emotion_train.csv"
VALIDATION_DATA_PATH = "data/emotion_validation.csv"

MODEL_OUTPUT_DIR = "models/bert_emotion"

BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
EPOCHS = 1


def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    total_loss = 0.0

    for step, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(
                f"Step {step + 1}/{len(dataloader)} "
                f"- Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(dataloader)

    return average_loss


def evaluate_loss(model, dataloader, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():

    print("=" * 60)
    print("MOOD MENTOR - BERT FINE-TUNING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    print("\nLoading training data...")

    train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    validation_dataframe = pd.read_csv(VALIDATION_DATA_PATH)

    print(f"Training examples   : {len(train_dataframe)}")
    print(f"Validation examples : {len(validation_dataframe)}")

    print("\nLoading BERT tokenizer...")

    tokenizer = load_bert_tokenizer()

    print("✓ Tokenizer loaded")

    print("\nCreating datasets...")

    train_dataset = create_emotion_dataset(
        dataframe=train_dataframe,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    validation_dataset = create_emotion_dataset(
        dataframe=validation_dataframe,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    print("✓ Datasets created")

    print("\nCreating DataLoaders...")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("✓ DataLoaders created")

    print("\nLoading BERT model...")

    model = load_bert_model()

    model.to(device)

    print("✓ BERT model loaded")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nTraining configuration:")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Max length    : {MAX_LENGTH}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Epochs        : {EPOCHS}")

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            device
        )

        validation_loss = evaluate_loss(
            model,
            validation_dataloader,
            device
        )

        print(f"\nTraining loss   : {train_loss:.4f}")
        print(f"Validation loss : {validation_loss:.4f}")

    print("\n" + "-" * 60)
    print("Training completed.")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print(f"\nSaving model to: {MODEL_OUTPUT_DIR}")

    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("✓ BERT model saved successfully")

    print("\nSaved files:")

    for filename in os.listdir(MODEL_OUTPUT_DIR):
        print(f"  - {filename}")

    print("\n" + "=" * 60)
    print("BERT FINE-TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
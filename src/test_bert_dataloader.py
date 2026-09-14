import pandas as pd
from torch.utils.data import DataLoader

from src.bert_model import load_bert_tokenizer
from src.bert_training import create_emotion_dataset


TRAIN_DATA_PATH = "data/emotion_train.csv"


def main():

    print("Loading training data...")

    dataframe = pd.read_csv(TRAIN_DATA_PATH)

    print(f"Training rows: {len(dataframe)}")

    print("\nLoading BERT tokenizer...")

    tokenizer = load_bert_tokenizer()

    print("✓ Tokenizer loaded")

    print("\nCreating PyTorch dataset...")

    dataset = create_emotion_dataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_length=128
    )

    print(f"Dataset size: {len(dataset)}")

    print("\nCreating DataLoader...")

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    print("✓ DataLoader created")

    print("\nChecking first batch...")

    batch = next(iter(dataloader))

    print("\nBatch contents:")

    print(f"input_ids shape      : {batch['input_ids'].shape}")
    print(f"attention_mask shape : {batch['attention_mask'].shape}")
    print(f"labels shape         : {batch['labels'].shape}")

    print("\nFirst example labels:")

    print(batch["labels"][0])

    print("\nExpected shapes:")

    print("input_ids      : [8, 128]")
    print("attention_mask : [8, 128]")
    print("labels         : [8, 6]")

    print("\n✓ BERT DataLoader validation completed successfully.")


if __name__ == "__main__":
    main()
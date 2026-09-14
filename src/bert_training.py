import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.emotion_config import EMOTION_LABELS


class EmotionDataset(Dataset):
    """
    PyTorch dataset for Mood Mentor's
    multi-label emotion classification.
    """

    def __init__(self, dataframe, tokenizer, max_length=128):

        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        text = str(row["text"])

        # Tokenize natural text for BERT.
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = np.array(
            [
                row[emotion]
                for emotion in EMOTION_LABELS
            ],
            dtype=np.float32
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels)
        }

        return item


def load_training_data(data_path):

    return pd.read_csv(data_path)


def create_emotion_dataset(
    dataframe,
    tokenizer,
    max_length=128
):

    return EmotionDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_length=max_length
    )
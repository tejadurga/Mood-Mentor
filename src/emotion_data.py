from datasets import load_dataset
import pandas as pd

from src.emotion_config import EMOTION_LABELS


DATASET_NAME = "google-research-datasets/go_emotions"


def load_goemotions():
    """
    Load the GoEmotions dataset from Hugging Face.
    """

    return load_dataset(DATASET_NAME)


def get_label_names(dataset):
    """
    Get the original GoEmotions label names.
    """

    label_feature = dataset["train"].features["labels"].feature

    return label_feature.names


def filter_target_emotions(dataset_split, label_names):
    """
    Keep only examples containing at least one
    of Mood Mentor's six target emotions.

    Convert emotion labels into multi-label binary columns.
    """

    target_label_ids = {
        label_names.index(emotion): emotion
        for emotion in EMOTION_LABELS
    }

    rows = []

    for example in dataset_split:

        detected_emotions = set()

        for label_id in example["labels"]:

            if label_id in target_label_ids:
                detected_emotions.add(
                    target_label_ids[label_id]
                )

        # Ignore examples that contain none of
        # our six target emotions.
        if not detected_emotions:
            continue

        row = {
            "text": example["text"]
        }

        for emotion in EMOTION_LABELS:
            row[emotion] = int(
                emotion in detected_emotions
            )

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_emotion_dataset():

    dataset = load_goemotions()

    label_names = get_label_names(dataset)

    train_df = filter_target_emotions(
        dataset["train"],
        label_names
    )

    validation_df = filter_target_emotions(
        dataset["validation"],
        label_names
    )

    test_df = filter_target_emotions(
        dataset["test"],
        label_names
    )

    return train_df, validation_df, test_df
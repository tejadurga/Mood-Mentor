import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.emotion_config import EMOTION_LABELS


MODEL_PATH = "models/bert_emotion"

ISEAR_SUPPORTED_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
]

ISEAR_LABEL_MAP = {
    1: "joy",
    2: "fear",
    3: "anger",
    4: "sadness",
    5: "disgust",
    6: "shame",
    7: "guilt",
}

BATCH_SIZE = 16
MAX_LENGTH = 128
THRESHOLD = 0.50


def load_isear_dataset():

    print("\nLoading ISEAR dataset...")

    dataset = load_dataset(
        "savalera/isear-from-original",
        "original"
    )

    print("✓ ISEAR dataset loaded")

    print(
        f"Available splits: {list(dataset.keys())}"
    )

    # Use the held-out test split.
    dataframe = dataset["test"].to_pandas()

    print(
        f"ISEAR test rows loaded: {len(dataframe)}"
    )

    return dataframe


def prepare_isear_dataframe(dataframe):

    required_columns = {"SIT", "EMOT"}

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "ISEAR dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe = dataframe[
        ["SIT", "EMOT"]
    ].copy()

    dataframe = dataframe.rename(
        columns={
            "SIT": "text",
            "EMOT": "emotion_id",
        }
    )

    dataframe["emotion_id"] = pd.to_numeric(
        dataframe["emotion_id"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=["text", "emotion_id"]
    )

    dataframe["emotion"] = (
        dataframe["emotion_id"]
        .astype(int)
        .map(ISEAR_LABEL_MAP)
    )

    dataframe = dataframe.dropna(
        subset=["emotion"]
    )

    # Keep only emotions supported by Mood Mentor.
    dataframe = dataframe[
        dataframe["emotion"].isin(
            ISEAR_SUPPORTED_EMOTIONS
        )
    ]

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


def load_trained_model():

    print("\nLoading trained BERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    print("✓ Trained BERT model loaded")

    return tokenizer, model


def generate_predictions(
    texts,
    tokenizer,
    model
):

    all_probabilities = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )

        probabilities = torch.sigmoid(
            outputs.logits
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    return np.vstack(
        all_probabilities
    )


def calculate_primary_predictions(
    dataframe,
    probabilities
):

    primary_indices = np.argmax(
        probabilities,
        axis=1
    )

    primary_predictions = [
        EMOTION_LABELS[index]
        for index in primary_indices
    ]

    true_emotions = dataframe[
        "emotion"
    ].tolist()

    primary_accuracy = accuracy_score(
        true_emotions,
        primary_predictions
    )

    return (
        primary_predictions,
        primary_indices,
        primary_accuracy
    )


def calculate_supported_metrics(
    dataframe,
    probabilities
):

    true_binary = []

    predicted_binary = []

    for row_index, row in dataframe.iterrows():

        true_emotion = row["emotion"]

        true_row = []

        predicted_row = []

        for emotion in ISEAR_SUPPORTED_EMOTIONS:

            emotion_index = EMOTION_LABELS.index(
                emotion
            )

            true_row.append(
                int(true_emotion == emotion)
            )

            predicted_row.append(
                int(
                    probabilities[row_index][emotion_index]
                    >= THRESHOLD
                )
            )

        true_binary.append(true_row)

        predicted_binary.append(
            predicted_row
        )

    true_binary = np.array(
        true_binary
    )

    predicted_binary = np.array(
        predicted_binary
    )

    macro_precision = precision_score(
        true_binary,
        predicted_binary,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        true_binary,
        predicted_binary,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        true_binary,
        predicted_binary,
        average="macro",
        zero_division=0
    )

    return (
        true_binary,
        predicted_binary,
        macro_precision,
        macro_recall,
        macro_f1
    )


def print_emotion_wise_results(
    dataframe,
    probabilities
):

    print("\n" + "=" * 75)
    print("ISEAR EMOTION-WISE PERFORMANCE")
    print("=" * 75)

    print(
        f"\n{'Emotion':<12}"
        f"{'Correct':<12}"
        f"{'Total':<12}"
        f"{'Accuracy':<15}"
        f"{'Avg True Prob':<18}"
    )

    print("-" * 75)

    for emotion in ISEAR_SUPPORTED_EMOTIONS:

        emotion_index = EMOTION_LABELS.index(
            emotion
        )

        emotion_rows = dataframe[
            dataframe["emotion"] == emotion
        ]

        indices = emotion_rows.index.tolist()

        correct = 0

        true_probabilities = []

        for row_index in indices:

            primary_index = int(
                np.argmax(
                    probabilities[row_index]
                )
            )

            if (
                EMOTION_LABELS[primary_index]
                == emotion
            ):

                correct += 1

            true_probabilities.append(
                probabilities[
                    row_index,
                    emotion_index
                ]
            )

        total = len(indices)

        emotion_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        average_probability = (
            np.mean(true_probabilities)
            if true_probabilities
            else 0
        )

        print(
            f"{emotion:<12}"
            f"{correct:<12}"
            f"{total:<12}"
            f"{emotion_accuracy:<15.4f}"
            f"{average_probability:<18.4f}"
        )


def print_confidence_analysis(
    dataframe,
    probabilities,
    primary_indices
):

    primary_confidences = (
        probabilities[
            np.arange(len(probabilities)),
            primary_indices
        ]
    )

    true_emotion_confidences = []

    for row_index, true_emotion in enumerate(
        dataframe["emotion"]
    ):

        emotion_index = EMOTION_LABELS.index(
            true_emotion
        )

        true_emotion_confidences.append(
            probabilities[
                row_index,
                emotion_index
            ]
        )

    true_emotion_confidences = np.array(
        true_emotion_confidences
    )

    print("\n" + "=" * 75)
    print("ISEAR CONFIDENCE ANALYSIS")
    print("=" * 75)

    print(
        f"\nAverage primary prediction confidence: "
        f"{primary_confidences.mean():.4f}"
    )

    print(
        f"Average confidence for true emotion: "
        f"{true_emotion_confidences.mean():.4f}"
    )

    print(
        f"Minimum true-emotion confidence: "
        f"{true_emotion_confidences.min():.4f}"
    )

    print(
        f"Maximum true-emotion confidence: "
        f"{true_emotion_confidences.max():.4f}"
    )


def show_incorrect_predictions(
    dataframe,
    probabilities,
    primary_predictions,
    number_of_examples=15
):

    incorrect_indices = [
        index
        for index, row in dataframe.iterrows()
        if primary_predictions[index]
        != row["emotion"]
    ]

    print("\n" + "=" * 75)
    print("INCORRECT ISEAR PREDICTIONS")
    print("=" * 75)

    print(
        f"\nIncorrect predictions: "
        f"{len(incorrect_indices)}"
    )

    print(
        f"Showing up to "
        f"{number_of_examples} examples."
    )

    for index in incorrect_indices[
        :number_of_examples
    ]:

        probability_row = probabilities[index]

        primary_index = int(
            np.argmax(probability_row)
        )

        true_emotion_index = EMOTION_LABELS.index(
            dataframe.iloc[index]["emotion"]
        )

        print("\n" + "-" * 75)

        print(
            f"Text: "
            f"{dataframe.iloc[index]['text']}"
        )

        print(
            f"Expected emotion: "
            f"{dataframe.iloc[index]['emotion']}"
        )

        print(
            f"Predicted emotion: "
            f"{primary_predictions[index]}"
        )

        print(
            f"Prediction confidence: "
            f"{probability_row[primary_index]:.4f}"
        )

        print(
            f"Expected-emotion probability: "
            f"{probability_row[true_emotion_index]:.4f}"
        )


def print_confusion_matrix_results(
    dataframe,
    primary_predictions
):

    matrix_labels = [
        "joy",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    ]

    true_emotions = dataframe[
        "emotion"
    ].tolist()

    matrix = confusion_matrix(
        true_emotions,
        primary_predictions,
        labels=matrix_labels
    )

    print("\n" + "=" * 75)
    print("ISEAR CONFUSION MATRIX")
    print("=" * 75)

    print(
        "\nRows = expected emotion"
    )

    print(
        "Columns = predicted emotion"
    )

    print()

    print(
        f"{'':<12}"
        + "".join(
            f"{label:<12}"
            for label in matrix_labels
        )
    )

    print("-" * 75)

    for row_index, emotion in enumerate(
        matrix_labels
    ):

        print(
            f"{emotion:<12}"
            + "".join(
                f"{value:<12}"
                for value in matrix[row_index]
            )
        )


def main():

    print("=" * 75)
    print("MOOD MENTOR - ISEAR BENCHMARK VALIDATION")
    print("=" * 75)

    dataframe = load_isear_dataset()

    dataframe = prepare_isear_dataframe(
        dataframe
    )

    print(
        f"\nPrepared ISEAR test examples "
        f"(supported emotions only): "
        f"{len(dataframe)}"
    )

    print("\nISEAR emotion distribution:")

    print(
        dataframe["emotion"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nNote: ISEAR does not contain "
        "the Surprise emotion."
    )

    print(
        "The benchmark therefore evaluates "
        "the five emotions shared with Mood Mentor."
    )

    tokenizer, model = load_trained_model()

    print(
        "\nGenerating BERT predictions..."
    )

    probabilities = generate_predictions(
        dataframe["text"].tolist(),
        tokenizer,
        model
    )

    print("✓ Predictions generated")

    print(
        f"Probability matrix shape: "
        f"{probabilities.shape}"
    )

    (
        primary_predictions,
        primary_indices,
        primary_accuracy
    ) = calculate_primary_predictions(
        dataframe,
        probabilities
    )

    print("\n" + "=" * 75)
    print("ISEAR PRIMARY PREDICTION ACCURACY")
    print("=" * 75)

    print(
        f"\nAccuracy: "
        f"{primary_accuracy:.4f}"
    )

    (
        _,
        _,
        macro_precision,
        macro_recall,
        macro_f1
    ) = calculate_supported_metrics(
        dataframe,
        probabilities
    )

    print("\n" + "=" * 75)
    print("ISEAR SUPPORTED-EMOTION METRICS")
    print("=" * 75)

    print(
        f"\nThreshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    print_emotion_wise_results(
        dataframe,
        probabilities
    )

    print_confidence_analysis(
        dataframe,
        probabilities,
        primary_indices
    )

    show_incorrect_predictions(
        dataframe,
        probabilities,
        primary_predictions
    )

    print_confusion_matrix_results(
        dataframe,
        primary_predictions
    )

    print("\n" + "=" * 75)
    print("ISEAR BENCHMARK VALIDATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
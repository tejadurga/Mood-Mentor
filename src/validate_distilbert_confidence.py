import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.emotion_config import EMOTION_LABELS


# --------------------------------------------------
# Confidence Validation Configuration
# --------------------------------------------------

MODEL_PATH = "models/distilbert_emotion"
VALIDATION_DATA_PATH = "data/emotion_validation.csv"

BATCH_SIZE = 8
MAX_LENGTH = 128

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

DEFAULT_THRESHOLD = 0.50


def load_trained_model():
    """
    Load the fine-tuned DistilBERT tokenizer and model.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


def generate_probabilities(
    dataframe,
    tokenizer,
    model,
):
    """
    Generate six-emotion probabilities for the
    validation dataset.
    """

    texts = (
        dataframe["text"]
        .astype(str)
        .tolist()
    )

    all_probabilities = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():

            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
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


def calculate_primary_accuracy(
    probabilities,
    dataframe,
):
    """
    Calculate how often the highest-probability
    emotion matches at least one true emotion.
    """

    correct = 0

    for index, row in dataframe.iterrows():

        primary_prediction = int(
            np.argmax(
                probabilities[index]
            )
        )

        true_labels = [
            int(row[emotion])
            for emotion in EMOTION_LABELS
        ]

        if true_labels[
            primary_prediction
        ] == 1:

            correct += 1

    return correct / len(dataframe)


def analyze_emotion_confidence(
    probabilities,
    dataframe,
):
    """
    Compare average probabilities when each emotion
    is present versus absent.
    """

    print("\n" + "=" * 70)
    print("EMOTION CONFIDENCE ANALYSIS")
    print("=" * 70)

    print(
        f"\n{'Emotion':<12}"
        f"{'True Avg Prob':<18}"
        f"{'False Avg Prob':<18}"
        f"{'Max Probability':<18}"
    )

    print("-" * 70)

    for index, emotion in enumerate(
        EMOTION_LABELS
    ):

        true_mask = (
            dataframe[emotion].values == 1
        )

        false_mask = (
            dataframe[emotion].values == 0
        )

        true_probabilities = (
            probabilities[
                true_mask,
                index,
            ]
        )

        false_probabilities = (
            probabilities[
                false_mask,
                index,
            ]
        )

        true_average = (
            true_probabilities.mean()
            if len(true_probabilities) > 0
            else 0
        )

        false_average = (
            false_probabilities.mean()
            if len(false_probabilities) > 0
            else 0
        )

        maximum_probability = (
            probabilities[
                :,
                index,
            ].max()
        )

        print(
            f"{emotion:<12}"
            f"{true_average:<18.4f}"
            f"{false_average:<18.4f}"
            f"{maximum_probability:<18.4f}"
        )


def analyze_thresholds(
    probabilities,
):
    """
    Show how the detection threshold affects
    the number of predicted emotions.
    """

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"\n{'Threshold':<12}"
        f"{'Avg emotions/text':<22}"
        f"{'Texts with >=1 emotion':<25}"
    )

    print("-" * 70)

    total_texts = len(
        probabilities
    )

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        )

        emotion_count_per_text = (
            predictions.sum(axis=1)
        )

        average_emotions = (
            emotion_count_per_text.mean()
        )

        texts_with_prediction = (
            (emotion_count_per_text >= 1).sum()
        )

        print(
            f"{threshold:<12.2f}"
            f"{average_emotions:<22.3f}"
            f"{texts_with_prediction}/"
            f"{total_texts:<25}"
        )


def show_sample_predictions(
    probabilities,
    dataframe,
    threshold=DEFAULT_THRESHOLD,
    number_of_samples=10,
):
    """
    Display sample validation predictions,
    probabilities and threshold-based detections.
    """

    print("\n" + "=" * 70)
    print(
        "SAMPLE VALIDATION PREDICTIONS "
        f"(Threshold = {threshold:.2f})"
    )
    print("=" * 70)

    samples = min(
        number_of_samples,
        len(dataframe),
    )

    for index in range(samples):

        text = str(
            dataframe.iloc[index]["text"]
        )

        probability_row = probabilities[
            index
        ]

        primary_index = int(
            np.argmax(
                probability_row
            )
        )

        primary_emotion = EMOTION_LABELS[
            primary_index
        ]

        true_emotions = [
            emotion
            for emotion in EMOTION_LABELS
            if dataframe.iloc[index][emotion] == 1
        ]

        detected_emotions = [
            (
                emotion,
                probability,
            )
            for emotion, probability in zip(
                EMOTION_LABELS,
                probability_row,
            )
            if probability >= threshold
        ]

        print("\n" + "-" * 70)

        print(
            f"Text: {text}"
        )

        print(
            "True emotions: "
            + (
                ", ".join(true_emotions)
                if true_emotions
                else "None"
            )
        )

        print(
            f"Primary prediction: "
            f"{primary_emotion} "
            f"({probability_row[primary_index]:.4f})"
        )

        print(
            "All probabilities:"
        )

        for emotion, probability in zip(
            EMOTION_LABELS,
            probability_row,
        ):

            print(
                f"  {emotion:<10}: "
                f"{probability:.4f}"
            )

        print(
            "Detected at threshold:"
        )

        if detected_emotions:

            for emotion, probability in (
                detected_emotions
            ):

                print(
                    f"  {emotion:<10}: "
                    f"{probability:.4f}"
                )

        else:

            print("  None")


def main():
    """
    Run complete DistilBERT confidence validation.
    """

    print("=" * 70)
    print(
        "MOOD MENTOR - DISTILBERT "
        "CONFIDENCE VALIDATION"
    )
    print("=" * 70)

    print(
        "\nLoading validation dataset..."
    )

    dataframe = pd.read_csv(
        VALIDATION_DATA_PATH
    )

    print(
        f"Validation examples: "
        f"{len(dataframe)}"
    )

    print(
        "\nLoading trained DistilBERT model..."
    )

    tokenizer, model = load_trained_model()

    print(
        "✓ Trained DistilBERT model loaded"
    )

    print(
        "\nGenerating model probabilities..."
    )

    probabilities = generate_probabilities(
        dataframe,
        tokenizer,
        model,
    )

    print(
        "✓ Probabilities generated"
    )

    print(
        f"\nProbability matrix shape: "
        f"{probabilities.shape}"
    )

    primary_accuracy = (
        calculate_primary_accuracy(
            probabilities,
            dataframe,
        )
    )

    print("\n" + "=" * 70)
    print("PRIMARY EMOTION CHECK")
    print("=" * 70)

    print(
        f"\nPrimary emotion matches at least one "
        f"true emotion: "
        f"{primary_accuracy:.4f}"
    )

    analyze_emotion_confidence(
        probabilities,
        dataframe,
    )

    analyze_thresholds(
        probabilities,
    )

    show_sample_predictions(
        probabilities,
        dataframe,
        threshold=DEFAULT_THRESHOLD,
        number_of_samples=10,
    )

    print("\n" + "=" * 70)
    print(
        "DISTILBERT CONFIDENCE VALIDATION "
        "COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
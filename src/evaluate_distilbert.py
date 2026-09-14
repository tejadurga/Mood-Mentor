import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.emotion_config import EMOTION_LABELS


# --------------------------------------------------
# Evaluation Configuration
# --------------------------------------------------

MODEL_PATH = "models/distilbert_emotion"
TEST_DATA_PATH = "data/emotion_test.csv"

BATCH_SIZE = 8
MAX_LENGTH = 128
PREDICTION_THRESHOLD = 0.50


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


def generate_predictions(
    dataframe,
    tokenizer,
    model,
):
    """
    Generate six-emotion probability predictions
    for every text in the test dataset.
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


def build_true_labels(dataframe):
    """
    Extract the six ground-truth emotion labels
    from the test dataset.
    """

    return dataframe[
        EMOTION_LABELS
    ].values.astype(int)


def evaluate_model(
    true_labels,
    probabilities,
    threshold,
):
    """
    Convert probabilities into binary multi-label
    predictions and calculate evaluation metrics.
    """

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    # Exact-match accuracy:
    # all six emotion labels must match exactly.
    exact_match_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    return (
        predicted_labels,
        exact_match_accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    )


def print_per_emotion_metrics(
    true_labels,
    predicted_labels,
):
    """
    Display precision, recall and F1 for each
    of the six target emotions.
    """

    precision = precision_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("PER-EMOTION METRICS")
    print("=" * 70)

    print(
        f"\n{'Emotion':<12}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1':<15}"
    )

    print("-" * 70)

    for index, emotion in enumerate(
        EMOTION_LABELS
    ):

        print(
            f"{emotion:<12}"
            f"{precision[index]:<15.4f}"
            f"{recall[index]:<15.4f}"
            f"{f1[index]:<15.4f}"
        )


def show_sample_predictions(
    dataframe,
    probabilities,
    predicted_labels,
    number_of_samples=10,
):
    """
    Display a small sample of true and predicted
    emotion labels from the test dataset.
    """

    print("\n" + "=" * 70)
    print("SAMPLE TEST PREDICTIONS")
    print("=" * 70)

    samples = min(
        number_of_samples,
        len(dataframe),
    )

    for index in range(samples):

        text = str(
            dataframe.iloc[index]["text"]
        )

        true_emotions = [
            emotion
            for emotion in EMOTION_LABELS
            if dataframe.iloc[index][emotion] == 1
        ]

        predicted_emotions = [
            emotion
            for emotion, value in zip(
                EMOTION_LABELS,
                predicted_labels[index],
            )
            if value == 1
        ]

        primary_index = int(
            np.argmax(
                probabilities[index]
            )
        )

        primary_emotion = EMOTION_LABELS[
            primary_index
        ]

        primary_confidence = probabilities[
            index,
            primary_index,
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
            "Predicted emotions: "
            + (
                ", ".join(predicted_emotions)
                if predicted_emotions
                else "None"
            )
        )

        print(
            f"Primary emotion: "
            f"{primary_emotion} "
            f"({primary_confidence:.4f})"
        )


def main():
    """
    Run the complete DistilBERT test-set evaluation.
    """

    print("=" * 70)
    print("MOOD MENTOR - DISTILBERT MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading test dataset...")

    dataframe = pd.read_csv(
        TEST_DATA_PATH
    )

    print(
        f"Test examples: "
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
        "\nGenerating test predictions..."
    )

    probabilities = generate_predictions(
        dataframe,
        tokenizer,
        model,
    )

    print(
        "✓ Predictions generated"
    )

    print(
        f"Probability matrix shape: "
        f"{probabilities.shape}"
    )

    true_labels = build_true_labels(
        dataframe
    )

    (
        predicted_labels,
        accuracy,
        precision,
        recall,
        macro_f1,
    ) = evaluate_model(
        true_labels,
        probabilities,
        PREDICTION_THRESHOLD,
    )

    print("\n" + "=" * 70)
    print(
        "OVERALL DISTILBERT METRICS "
        f"(Threshold = {PREDICTION_THRESHOLD:.2f})"
    )
    print("=" * 70)

    print(
        f"\nExact-match Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision      : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall         : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1-Score       : "
        f"{macro_f1:.4f}"
    )

    print_per_emotion_metrics(
        true_labels,
        predicted_labels,
    )

    show_sample_predictions(
        dataframe,
        probabilities,
        predicted_labels,
    )

    print("\n" + "=" * 70)
    print("DISTILBERT EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.emotion_config import EMOTION_LABELS


MODEL_PATH = "models/bert_emotion"
PREDICTION_THRESHOLD = 0.50


def load_trained_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


def predict(text, tokenizer, model):

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    probabilities = torch.sigmoid(outputs.logits)[0]

    results = []

    for emotion, probability in zip(
        EMOTION_LABELS,
        probabilities
    ):

        confidence = probability.item()

        if confidence >= PREDICTION_THRESHOLD:

            results.append(
                (emotion, confidence)
            )

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return probabilities, results


def main():

    print("=" * 70)
    print("MOOD MENTOR - BERT MULTI-LABEL EMOTION TEST")
    print("=" * 70)

    tokenizer, model = load_trained_model()

    print("\n✓ Trained BERT model loaded")

    test_cases = [

        (
            "Single Emotion - Joy",
            "I am extremely happy and excited about my success!"
        ),

        (
            "Single Emotion - Sadness",
            "I feel deeply sad and heartbroken today."
        ),

        (
            "Single Emotion - Anger",
            "I am furious about what happened."
        ),

        (
            "Single Emotion - Fear",
            "I am terrified about what might happen."
        ),

        (
            "Single Emotion - Surprise",
            "Wow! I never expected this to happen!"
        ),

        (
            "Single Emotion - Disgust",
            "That behavior is absolutely disgusting."
        ),

        (
            "Mixed - Joy + Fear",
            "I am excited about the opportunity but nervous about the outcome."
        ),

        (
            "Mixed - Anger + Disgust",
            "This disgusting behavior makes me extremely angry."
        ),

        (
            "Mixed - Sadness + Anger",
            "I am heartbroken and angry about what happened."
        ),

        (
            "Strong Emotion",
            "I am absolutely furious and disgusted by this horrible behavior!"
        )
    ]

    for title, text in test_cases:

        probabilities, detected = predict(
            text,
            tokenizer,
            model
        )

        print("\n" + "-" * 70)
        print(title)
        print(f"Text: {text}")

        print("\nProbabilities:")

        for emotion, probability in zip(
            EMOTION_LABELS,
            probabilities
        ):

            print(
                f"  {emotion:<10}: "
                f"{probability.item():.4f}"
            )

        print("\nDetected emotions:")

        if detected:

            for emotion, confidence in detected:

                print(
                    f"  {emotion:<10} "
                    f"({confidence:.4f})"
                )

            print(
                f"\nPrimary emotion: {detected[0][0]}"
            )

        else:

            print("  No emotion crossed the threshold.")

    print("\n" + "=" * 70)
    print("MULTI-LABEL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
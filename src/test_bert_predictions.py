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


def predict_emotions(text, tokenizer, model):

    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    probabilities = torch.sigmoid(outputs.logits)[0]

    detected_emotions = []

    for emotion, probability in zip(
        EMOTION_LABELS,
        probabilities
    ):

        confidence = probability.item()

        if confidence >= PREDICTION_THRESHOLD:

            detected_emotions.append(
                {
                    "emotion": emotion,
                    "confidence": confidence
                }
            )

    detected_emotions.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    return detected_emotions, probabilities


def main():

    print("=" * 60)
    print("MOOD MENTOR - BERT EMOTION PREDICTION TEST")
    print("=" * 60)

    print("\nLoading trained BERT model...")

    tokenizer, model = load_trained_model()

    print("✓ Trained BERT model loaded successfully")

    test_texts = [
        "I am extremely happy about this wonderful news!",
        "I feel very sad and disappointed today.",
        "I am really angry about what happened.",
        "I am scared and nervous about the situation.",
        "Wow! I cannot believe this happened!",
        "This disgusting behavior makes me angry.",
        "I am excited about the opportunity but nervous about the outcome."
    ]

    for text in test_texts:

        detected_emotions, probabilities = predict_emotions(
            text,
            tokenizer,
            model
        )

        print("\n" + "-" * 60)
        print(f"Text: {text}")

        print("\nEmotion probabilities:")

        for emotion, probability in zip(
            EMOTION_LABELS,
            probabilities
        ):

            print(
                f"  {emotion:<10}: "
                f"{probability.item():.4f}"
            )

        print("\nDetected emotions:")

        if detected_emotions:

            for result in detected_emotions:

                print(
                    f"  {result['emotion']:<10} "
                    f"confidence: "
                    f"{result['confidence']:.4f}"
                )

            print(
                f"\nPrimary emotion: "
                f"{detected_emotions[0]['emotion']}"
            )

        else:

            print(
                "  No emotion crossed "
                f"the threshold ({PREDICTION_THRESHOLD})."
            )

    print("\n" + "=" * 60)
    print("BERT PREDICTION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
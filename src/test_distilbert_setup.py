import torch

from src.distilbert_model import (
    load_distilbert_tokenizer,
    load_distilbert_model
)

from src.emotion_config import (
    EMOTION_LABELS,
    NUM_LABELS
)


def main():

    print("=" * 60)
    print("MOOD MENTOR - DISTILBERT SETUP TEST")
    print("=" * 60)

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = load_distilbert_tokenizer()

    print("✓ DistilBERT tokenizer loaded")

    sample_text = (
        "I am excited about the opportunity "
        "but nervous about the outcome."
    )

    print("\nTesting tokenizer...")

    encoding = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    print(
        f"input_ids shape      : "
        f"{encoding['input_ids'].shape}"
    )

    print(
        f"attention_mask shape : "
        f"{encoding['attention_mask'].shape}"
    )

    print("\nLoading DistilBERT model...")

    model = load_distilbert_model()

    print("✓ DistilBERT model loaded")

    print("\nModel configuration:")

    print(f"Number of labels : {model.config.num_labels}")
    print(
        f"Problem type     : "
        f"{model.config.problem_type}"
    )

    print("\nEmotion labels:")

    for index, emotion in enumerate(EMOTION_LABELS):

        print(
            f"  {index} → {emotion}"
        )

    print("\nChecking configuration...")

    assert NUM_LABELS == 6
    assert model.config.num_labels == 6
    assert model.config.problem_type == "multi_label_classification"

    print("\n✓ Six-emotion configuration verified")
    print("✓ Multi-label configuration verified")

    print("\nRunning a forward pass...")

    model.eval()

    with torch.no_grad():

        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    print(
        f"Logits shape: "
        f"{outputs.logits.shape}"
    )

    assert outputs.logits.shape == (1, 6)

    print("✓ Forward pass successful")

    print("\n" + "=" * 60)
    print("DISTILBERT SETUP TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
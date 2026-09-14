from src.bert_model import (
    load_bert_tokenizer,
    load_bert_model
)


def main():

    print("=" * 60)
    print("MOOD MENTOR - BERT SETUP TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print("\nLoading BERT tokenizer...")

    tokenizer = load_bert_tokenizer()

    print("✓ BERT tokenizer loaded successfully.")

    # --------------------------------------------------
    # Test tokenization
    # --------------------------------------------------

    text = (
        "I am excited about the new opportunity "
        "but nervous about the outcome."
    )

    print("\nOriginal text:")
    print(text)

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    print("\nTokenizer output:")

    print(
        "Input IDs shape:",
        encoded["input_ids"].shape
    )

    print(
        "Attention mask shape:",
        encoded["attention_mask"].shape
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading pre-trained BERT model...")

    model = load_bert_model()

    print("✓ BERT model loaded successfully.")

    # --------------------------------------------------
    # Model information
    # --------------------------------------------------

    print("\nModel configuration:")

    print(
        "Model:",
        "bert-base-uncased"
    )

    print(
        "Number of labels:",
        model.config.num_labels
    )

    print(
        "Problem type:",
        model.config.problem_type
    )

    print(
        "Labels:",
        model.config.id2label
    )

    print("\n" + "=" * 60)
    print("BERT SETUP TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
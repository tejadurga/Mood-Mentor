from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.emotion_config import (
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)


# --------------------------------------------------
# DistilBERT Configuration
# --------------------------------------------------

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"


def load_distilbert_tokenizer():
    """
    Load the pre-trained DistilBERT tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_MODEL_NAME
    )

    return tokenizer


def load_distilbert_model():
    """
    Load pre-trained DistilBERT configured for
    six-emotion multi-label classification.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification"
    )

    return model
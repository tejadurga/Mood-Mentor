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
# BERT Configuration
# --------------------------------------------------

BERT_MODEL_NAME = "bert-base-uncased"


def load_bert_tokenizer():
    """
    Load the pre-trained BERT tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    return tokenizer


def load_bert_model():
    """
    Load pre-trained BERT configured for
    six-emotion multi-label classification.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification"
    )

    return model
from functools import lru_cache

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 128


@lru_cache(maxsize=1)
def load_semantic_model():
    """
    Load the pretrained sentence embedding model.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        SEMANTIC_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        SEMANTIC_MODEL_NAME
    )

    model.eval()

    return tokenizer, model


def mean_pooling(
    model_output,
    attention_mask,
):
    """
    Create one sentence embedding by averaging token
    embeddings while ignoring padding tokens.
    """

    token_embeddings = model_output.last_hidden_state

    expanded_attention_mask = attention_mask.unsqueeze(-1)

    expanded_attention_mask = expanded_attention_mask.expand(
        token_embeddings.size()
    ).float()

    masked_embeddings = (
        token_embeddings * expanded_attention_mask
    )

    summed_embeddings = masked_embeddings.sum(
        dim=1
    )

    token_counts = expanded_attention_mask.sum(
        dim=1
    ).clamp(min=1e-9)

    return summed_embeddings / token_counts


def encode_texts(texts):
    """
    Convert a list of texts into normalized semantic embeddings.
    """

    if not texts:
        return torch.empty(
            (0, 384),
            dtype=torch.float32
        )

    tokenizer, model = load_semantic_model()

    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():

        model_output = model(
            **encoded_inputs
        )

    embeddings = mean_pooling(
        model_output=model_output,
        attention_mask=encoded_inputs["attention_mask"],
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return embeddings


def build_user_query_text(
    user_text,
    detected_emotions,
    emotion_intensity,
    emotional_state,
):
    """
    Build a semantic query using the original user text
    plus the current emotional context.
    """

    emotions = ", ".join(
        detected_emotions
    ) if detected_emotions else "none"

    return (
        f"User statement: {user_text}. "
        f"Detected emotions: {emotions}. "
        f"Emotion intensity: {emotion_intensity:.2f}. "
        f"Emotional state: {emotional_state}."
    )


def build_content_text(content):
    """
    Convert a wellness content row into a semantic document.
    """

    target_emotions = ", ".join(
        content["target_emotions"]
    )

    return (
        f"{content['title']}. "
        f"{content['description']} "
        f"Category: {content['category']}. "
        f"Target emotions: {target_emotions}. "
        f"Intensity level: {content['intensity_level']}. "
        f"Activity type: {content['activity_type']}. "
        f"Duration: {content['duration_minutes']} minutes."
    )


def calculate_semantic_similarity(
    query_embedding,
    content_embeddings,
):
    """
    Calculate cosine similarity between the user query
    embedding and wellness content embeddings.
    """

    if query_embedding.numel() == 0:
        return torch.empty(0)

    if content_embeddings.numel() == 0:
        return torch.empty(0)

    similarities = torch.mm(
        query_embedding,
        content_embeddings.T,
    )

    return similarities.squeeze(0)


def semantic_match_wellness_content(
    user_text,
    detected_emotions,
    emotion_intensity,
    emotional_state,
    wellness_content,
    top_k=5,
    minimum_similarity=0.30,
):
    """
    Match the user's current text/emotional context against
    wellness content using semantic embeddings.

    Returns the most semantically relevant content items.
    """

    if wellness_content.empty:
        return pd.DataFrame()

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if minimum_similarity < -1 or minimum_similarity > 1:
        raise ValueError(
            "minimum_similarity must be between -1 and 1."
        )

    query_text = build_user_query_text(
        user_text=user_text,
        detected_emotions=detected_emotions,
        emotion_intensity=emotion_intensity,
        emotional_state=emotional_state,
    )

    content_texts = [
        build_content_text(row)
        for _, row in wellness_content.iterrows()
    ]

    query_embedding = encode_texts(
        [query_text]
    )

    content_embeddings = encode_texts(
        content_texts
    )

    similarities = calculate_semantic_similarity(
        query_embedding=query_embedding,
        content_embeddings=content_embeddings,
    )

    result = wellness_content.copy()

    result["semantic_similarity"] = (
        similarities.cpu().numpy()
    )

    # Convert cosine similarity from [-1, 1] into
    # a convenient relevance scale of [0, 1].
    result["semantic_relevance"] = (
        (result["semantic_similarity"] + 1.0) / 2.0
    )

    result = result[
        result["semantic_similarity"]
        >= minimum_similarity
    ].copy()

    result = result.sort_values(
        by=[
            "semantic_similarity",
            "content_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    result["semantic_rank"] = (
        result.index + 1
    )

    return result.head(
        top_k
    )
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment
from src.emotion_state import analyze_emotional_state
from src.recommendation_engine import (
    build_hybrid_recommendations,
)
from src.recommendation_ranking import (
    rank_recommendations,
    get_top_recommendation,
)


MODEL_PATH = "models/bert_emotion"
USER_ID = "demo_user_001"
EMOTION_THRESHOLD = 0.50


def load_emotion_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


def predict_emotions(
    text,
    tokenizer,
    model,
    device,
):

    preprocessing_result = preprocess_text(
        text
    )

    if not preprocessing_result["success"]:
        raise ValueError(
            preprocessing_result["message"]
        )

    processed_text = preprocessing_result[
        "processed_text"
    ]

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.sigmoid(
        outputs.logits
    )[0].cpu().numpy()

    return probabilities
def run_analysis(text):

    # ------------------------------------------------
    # SENTIMENT
    # ------------------------------------------------

    sentiment_result = analyze_sentiment(
        text
    )

    if not sentiment_result["success"]:

        raise ValueError(
            sentiment_result["message"]
        )

    # ------------------------------------------------
    # BERT EMOTION ANALYSIS
    # ------------------------------------------------

    tokenizer, model, device = (
        load_emotion_model()
    )

    probabilities = predict_emotions(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    # ------------------------------------------------
    # EMOTIONAL STATE
    # ------------------------------------------------

    emotional_state = (
        analyze_emotional_state(
            probabilities=probabilities,
            sentiment_result=sentiment_result,
            threshold=EMOTION_THRESHOLD,
        )
    )

    detected_emotions = (
        emotional_state[
            "detected_emotions"
        ]
    )

    emotion_intensity = (
        emotional_state[
            "emotion_intensity"
        ]
    )

    emotional_state_text = (
        emotional_state[
            "emotional_state"
        ]
    )

    # ------------------------------------------------
    # HYBRID RECOMMENDATIONS
    # ------------------------------------------------

    hybrid_recommendations = (
        build_hybrid_recommendations(
            user_id=USER_ID,
            user_text=text,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            emotional_state=emotional_state_text,
        )
    )

    # ------------------------------------------------
    # RANKING
    # ------------------------------------------------

    ranking_result = rank_recommendations(
        recommendations=hybrid_recommendations,
        top_k=5,
        minimum_score=0.30,
    )

    top_recommendation = (
        get_top_recommendation(
            ranking_result
        )
    )

    return (
        sentiment_result,
        emotional_state,
        ranking_result,
        top_recommendation,
    )


def print_results(
    text,
    sentiment_result,
    emotional_state,
    ranking_result,
    top_recommendation,
):

    print("\n" + "=" * 60)
    print("MOOD MENTOR — CLI")
    print("=" * 60)

    print("\nINPUT")
    print("-" * 60)
    print(text)

    print("\nSENTIMENT")
    print("-" * 60)

    print(
        f"Sentiment : "
        f"{sentiment_result['sentiment']}"
    )

    print(
        f"Compound  : "
        f"{sentiment_result['compound']:.4f}"
    )

    print("\nEMOTIONAL STATE")
    print("-" * 60)

    print(
        f"Dominant Emotion : "
        f"{emotional_state['dominant_emotion']}"
    )

    print(
        f"Confidence       : "
        f"{emotional_state['dominant_confidence']:.4f}"
    )

    print(
        f"Detected         : "
        f"{', '.join(emotional_state['detected_emotions'])}"
    )

    print(
        f"Intensity        : "
        f"{emotional_state['emotion_intensity']:.4f}"
    )

    print(
        f"Intensity Level  : "
        f"{emotional_state['intensity_level']}"
    )

    print(
        f"Polarity         : "
        f"{emotional_state['polarity']}"
    )

    print(
        f"State            : "
        f"{emotional_state['emotional_state']}"
    )

    print("\nRECOMMENDATIONS")
    print("-" * 60)

    recommendations = (
        ranking_result[
            "recommendations"
        ]
    )

    if recommendations.empty:

        print(
            "No recommendations found."
        )

    else:

        for _, recommendation in (
            recommendations.iterrows()
        ):

            print(
                f"{int(recommendation['rank'])}. "
                f"{recommendation['title']}"
            )

            print(
                f"   Content ID : "
                f"{recommendation['content_id']}"
            )

            print(
                f"   Score      : "
                f"{recommendation['hybrid_score']:.4f}"
            )

            print()

    print("RECOMMENDATION VALIDATION")
    print("-" * 60)

    print(
        f"Candidates          : "
        f"{ranking_result['total_candidates']}"
    )

    print(
        f"Ranked              : "
        f"{ranking_result['total_ranked']}"
    )

    print(
        f"Duplicates detected : "
        f"{len(ranking_result['duplicates_detected'])}"
    )

    print(
        f"Low relevance removed: "
        f"{ranking_result['low_relevance_removed']}"
    )

    print("\nTOP RECOMMENDATION")
    print("-" * 60)

    if top_recommendation:

        print(
            f"Title : "
            f"{top_recommendation['title']}"
        )

        print(
            f"Score : "
            f"{top_recommendation['hybrid_score']:.4f}"
        )

    else:

        print(
            "No top recommendation available."
        )

    print("\n" + "=" * 60)
    print("CLI ANALYSIS COMPLETED")
    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mood Mentor emotion analysis "
            "and recommendation CLI."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        help=(
            "Text to analyze."
        ),
    )

    args = parser.parse_args()

    text = args.text

    # ------------------------------------------------
    # INTERACTIVE INPUT
    # ------------------------------------------------

    if not text:

        text = input(
            "\nEnter how you are feeling: "
        ).strip()

    if not text:

        print(
            "\nError: Please enter some text."
        )

        return

    try:

        (
            sentiment_result,
            emotional_state,
            ranking_result,
            top_recommendation,
        ) = run_analysis(text)

        print_results(
            text=text,
            sentiment_result=sentiment_result,
            emotional_state=emotional_state,
            ranking_result=ranking_result,
            top_recommendation=top_recommendation,
        )

    except Exception as error:

        print(
            "\nMood Mentor CLI error:"
        )

        print(error)


if __name__ == "__main__":
    main()
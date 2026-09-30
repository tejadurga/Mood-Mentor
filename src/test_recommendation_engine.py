from src.recommendation_engine import (
    build_hybrid_recommendations,
)


def main():

    user_id = "demo_user_001"

    user_text = (
        "I feel scared and overwhelmed "
        "about tomorrow."
    )

    detected_emotions = [
        "fear",
        "sadness",
    ]

    emotion_intensity = 0.82

    emotional_state = (
        "Very High-intensity mixed "
        "emotional state"
    )

    recommendations = (
        build_hybrid_recommendations(
            user_id=user_id,
            user_text=user_text,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            emotional_state=emotional_state,
        )
    )

    assert not recommendations.empty

    # ---------------------------------------------
    # Required columns
    # ---------------------------------------------

    assert (
        "historical_trend_signal"
        in recommendations.columns
    )

    assert (
        "feedback_signal"
        in recommendations.columns
    )

    assert (
        "semantic_relevance"
        in recommendations.columns
    )

    # ---------------------------------------------
    # Validate signal ranges
    # ---------------------------------------------

    assert (
        recommendations[
            "historical_trend_signal"
        ]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        recommendations[
            "feedback_signal"
        ]
        .between(0.0, 1.0)
        .all()
    )

    assert (
        recommendations[
            "hybrid_score"
        ]
        .between(0.0, 1.0)
        .all()
    )

    print(
        "Hybrid Recommendation Engine "
        "with Trend + Feedback Test"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"User: {user_id}"
    )

    print(
        f"Text: {user_text}"
    )

    print(
        "Current emotions:",
        ", ".join(
            detected_emotions
        )
    )

    print(
        f"Current intensity: "
        f"{emotion_intensity:.2f}"
    )

    print(
        f"Emotional state: "
        f"{emotional_state}"
    )

    print(
        "\nHybrid recommendation results:"
    )

    for _, recommendation in (
        recommendations.iterrows()
    ):

        print(
            f"\nRank "
            f"{int(recommendation['hybrid_rank'])}: "
            f"{recommendation['content_id']} - "
            f"{recommendation['title']}"
        )

        print(
            f"  Personalized score: "
            f"{recommendation['recommendation_score']:.4f}"
        )

        print(
            f"  Content similarity: "
            f"{recommendation['content_similarity']:.4f}"
        )

        print(
            f"  Semantic relevance: "
            f"{recommendation['semantic_relevance']:.4f}"
        )

        print(
            f"  Collaborative signal: "
            f"{recommendation['collaborative_signal']:.4f}"
        )

        print(
            f"  Emotion match: "
            f"{recommendation['emotion_match']:.4f}"
        )

        print(
            f"  Historical trend signal: "
            f"{recommendation['historical_trend_signal']:.4f}"
        )

        print(
            f"  Feedback signal: "
            f"{recommendation['feedback_signal']:.4f}"
        )

        print(
            f"  Final hybrid score: "
            f"{recommendation['hybrid_score']:.4f}"
        )

    # ---------------------------------------------
    # Verify historical signal
    # ---------------------------------------------

    trend_values = (
        recommendations[
            "historical_trend_signal"
        ]
        .unique()
        .tolist()
    )

    assert 0.9000 in trend_values

    print(
        "\nHistorical emotional trend "
        "influence: PASSED"
    )

    # ---------------------------------------------
    # Verify feedback influence
    # ---------------------------------------------

    c003 = recommendations[
        recommendations["content_id"]
        == "C003"
    ]

    c014 = recommendations[
        recommendations["content_id"]
        == "C014"
    ]

    if not c003.empty and not c014.empty:

        c003_feedback = float(
            c003.iloc[0][
                "feedback_signal"
            ]
        )

        c014_feedback = float(
            c014.iloc[0][
                "feedback_signal"
            ]
        )

        print(
            f"C003 feedback signal: "
            f"{c003_feedback:.4f}"
        )

        print(
            f"C014 feedback signal: "
            f"{c014_feedback:.4f}"
        )

        assert (
            c003_feedback
            > c014_feedback
        )

        print(
            "Positive vs negative feedback "
            "influence: PASSED"
        )

    print(
        "\nHybrid recommendation engine "
        "with trend + feedback test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
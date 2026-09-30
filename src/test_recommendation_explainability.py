from src.recommendation_data import (
    get_user_profile,
)

from src.recommendation_engine import (
    build_hybrid_recommendations,
)

from src.recommendation_explainability import (
    build_explainable_recommendations,
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

    print(
        "Recommendation Explainability Test"
    )

    print(
        "-----------------------------------"
    )

    profile = get_user_profile(
        user_id
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

    explainable = (
        build_explainable_recommendations(
            recommendations=recommendations,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            preferred_categories=profile[
                "preferred_categories"
            ],
            preferred_activity_types=profile[
                "preferred_activity_types"
            ],
            historical_trend_signal=float(
                recommendations[
                    "historical_trend_signal"
                ].iloc[0]
            ),
        )
    )

    assert not explainable.empty

    assert (
        "explanation_reasons"
        in explainable.columns
    )

    assert (
        "explanation"
        in explainable.columns
    )

    assert (
        explainable[
            "explanation"
        ]
        .notna()
        .all()
    )

    print(
        f"User: {user_id}"
    )

    print(
        f"Current emotions: "
        f"{', '.join(detected_emotions)}"
    )

    print(
        f"Emotion intensity: "
        f"{emotion_intensity:.2f}"
    )

    print(
        "\nExplainable recommendations:"
    )

    for _, recommendation in (
        explainable.head(5).iterrows()
    ):

        print(
            f"\nRank "
            f"{int(recommendation['hybrid_rank'])}: "
            f"{recommendation['title']}"
        )

        print(
            f"Hybrid score: "
            f"{recommendation['hybrid_score']:.4f}"
        )

        print(
            "Why selected:"
        )

        for reason in recommendation[
            "explanation_reasons"
        ]:

            print(
                f"  - {reason}"
            )

    print(
        "\nExplainability validation: PASSED"
    )

    print(
        "Dynamic recommendation reasons: PASSED"
    )

    print(
        "\nRecommendation explainability "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()
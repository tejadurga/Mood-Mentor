from src.recommendation_rules import (
    generate_personalized_scores,
)


def main():

    user_id = "demo_user_001"

    detected_emotions = [
        "fear",
        "sadness",
    ]

    emotion_intensity = 0.82

    recommendations = generate_personalized_scores(
        user_id=user_id,
        detected_emotions=detected_emotions,
        emotion_intensity=emotion_intensity,
    )

    print("Personalized Recommendation Test")
    print("--------------------------------")

    print(f"User: {user_id}")
    print(
        "Detected emotions:",
        ", ".join(detected_emotions)
    )
    print(
        f"Emotion intensity: {emotion_intensity:.2f}"
    )

    print("\nGenerated recommendation scores:")

    for recommendation in recommendations:

        print(
            f"{recommendation['content_id']} - "
            f"{recommendation['title']}"
        )

        print(
            f"  Emotion match: "
            f"{recommendation['emotion_match']:.2f}"
        )

        print(
            f"  Intensity match: "
            f"{recommendation['intensity_match']:.2f}"
        )

        print(
            f"  Preference match: "
            f"{recommendation['preference_match']:.2f}"
        )

        print(
            f"  Duration match: "
            f"{recommendation['duration_match']:.2f}"
        )

        print(
            f"  History preference: "
            f"{recommendation['history_preference']:.2f}"
        )

        print(
            f"  Final recommendation score: "
            f"{recommendation['recommendation_score']:.4f}"
        )

        print()

    print(
        "Personalized recommendation scoring "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()
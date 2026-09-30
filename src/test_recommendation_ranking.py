from src.recommendation_engine import (
    build_hybrid_recommendations,
)

from src.recommendation_ranking import (
    rank_recommendations,
    get_top_recommendation,
)


def main():

    user_id = "demo_user_001"

    detected_emotions = [
        "fear",
        "sadness",
    ]

    emotion_intensity = 0.82

    print("Recommendation Ranking Model Test")
    print("----------------------------------")

    print(f"User: {user_id}")

    print(
        "Detected emotions:",
        ", ".join(detected_emotions),
    )

    print(
        f"Emotion intensity: "
        f"{emotion_intensity:.2f}"
    )

    hybrid_recommendations = (
        build_hybrid_recommendations(
            user_id=user_id,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
        )
    )

    ranking_result = rank_recommendations(
        recommendations=hybrid_recommendations,
        top_k=5,
        minimum_score=0.30,
    )

    print("\nRanking validation:")
    print(
        f"Total candidates: "
        f"{ranking_result['total_candidates']}"
    )

    print(
        f"Total ranked: "
        f"{ranking_result['total_ranked']}"
    )

    print(
        f"Duplicates detected: "
        f"{ranking_result['duplicates_detected']}"
    )

    print(
        f"Duplicates removed: "
        f"{ranking_result['duplicates_removed']}"
    )

    print(
        f"Low-relevance recommendations removed: "
        f"{ranking_result['low_relevance_removed']}"
    )

    print("\nTop recommendations:")

    recommendations = ranking_result[
        "recommendations"
    ]

    for _, recommendation in recommendations.iterrows():

        print(
            f"Rank {int(recommendation['rank'])}: "
            f"{recommendation['content_id']} - "
            f"{recommendation['title']}"
        )

        print(
            f"  Hybrid score: "
            f"{recommendation['hybrid_score']:.4f}"
        )

        print(
            f"  Emotion match: "
            f"{recommendation['emotion_match']:.4f}"
        )

        print(
            f"  Content similarity: "
            f"{recommendation['content_similarity']:.4f}"
        )

        print()

    top_recommendation = get_top_recommendation(
        ranking_result
    )

    print("Top recommendation:")

    if top_recommendation is None:
        print("No recommendation available.")
    else:
        print(
            f"  {top_recommendation['content_id']} - "
            f"{top_recommendation['title']}"
        )

        print(
            f"  Score: "
            f"{top_recommendation['hybrid_score']:.4f}"
        )

        print(
            f"  Rank: "
            f"{int(top_recommendation['rank'])}"
        )

    print(
        "\nRecommendation ranking model "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()
from src.recommendation_data import (
    load_wellness_content,
)

from src.semantic_matching import (
    build_user_query_text,
    semantic_match_wellness_content,
)


def test_semantic_query_builder():

    query = build_user_query_text(
        user_text=(
            "I feel scared and overwhelmed "
            "about tomorrow."
        ),
        detected_emotions=[
            "fear",
            "sadness",
        ],
        emotion_intensity=0.82,
        emotional_state=(
            "Very High-intensity mixed "
            "emotional state"
        ),
    )

    assert "scared" in query
    assert "fear" in query
    assert "sadness" in query
    assert "0.82" in query

    print("Query builder test: PASSED")


def test_semantic_matching():

    wellness_content = load_wellness_content()

    results = semantic_match_wellness_content(
        user_text=(
            "I feel scared and overwhelmed "
            "about tomorrow."
        ),
        detected_emotions=[
            "fear",
            "sadness",
        ],
        emotion_intensity=0.82,
        emotional_state=(
            "Very High-intensity mixed "
            "emotional state"
        ),
        wellness_content=wellness_content,
        top_k=5,
        minimum_similarity=0.30,
    )

    assert not results.empty

    assert len(results) <= 5

    assert results["semantic_similarity"].is_monotonic_decreasing

    assert results["semantic_relevance"].between(
        0.0,
        1.0,
    ).all()

    assert results["content_id"].is_unique

    print("Semantic matching test: PASSED")

    print("\nTop semantic matches:")
    print()

    for _, row in results.iterrows():

        print(
            f"Rank {int(row['semantic_rank'])}: "
            f"{row['content_id']} - "
            f"{row['title']}"
        )

        print(
            f"  Cosine similarity: "
            f"{row['semantic_similarity']:.4f}"
        )

        print(
            f"  Semantic relevance: "
            f"{row['semantic_relevance']:.4f}"
        )

        print(
            f"  Category: "
            f"{row['category']}"
        )

        print(
            f"  Activity: "
            f"{row['activity_type']}"
        )

        print()


def main():

    print("Semantic Wellness Content Matching Test")
    print("----------------------------------------")
    print()

    test_semantic_query_builder()

    print()

    test_semantic_matching()

    print(
        "Semantic wellness content matching "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()
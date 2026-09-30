import pandas as pd

from src.recommendation_data import (
    get_user_profile,
    get_user_history,
    load_user_history,
    load_wellness_content,
)

from src.recommendation_rules import (
    generate_personalized_scores,
)

from src.recommendation_content import (
    calculate_content_similarity_for_all,
)

from src.semantic_matching import (
    semantic_match_wellness_content,
)

from src.emotion_trends import (
    analyze_emotion_trends,
)

from src.recommendation_feedback import (
    calculate_feedback_signal,
)


def calculate_collaborative_signal(
    content_id,
    user_id,
    user_history,
):
    """
    Calculate a lightweight collaborative signal.

    The score uses interactions from other demo users
    for the same content item.
    """

    if user_history.empty:
        return 0.5

    other_user_history = user_history[
        user_history["user_id"] != user_id
    ]

    if other_user_history.empty:
        return 0.5

    matching_records = other_user_history[
        other_user_history["content_id"] == content_id
    ]

    if matching_records.empty:
        return 0.5

    scores = []

    for _, record in matching_records.iterrows():

        helpfulness = float(
            record["helpfulness_rating"]
        )

        helpfulness_score = (
            helpfulness / 5.0
        )

        if bool(record["was_completed"]):

            interaction_score = (
                0.6
                + (helpfulness_score * 0.4)
            )

        else:

            interaction_score = (
                helpfulness_score * 0.4
            )

        scores.append(
            interaction_score
        )

    if not scores:
        return 0.5

    return round(
        float(
            sum(scores)
            / len(scores)
        ),
        4,
    )


def build_hybrid_recommendations(
    user_id,
    user_text,
    detected_emotions,
    emotion_intensity,
    emotional_state,
):
    """
    Build hybrid recommendation candidates.

    Signals included:

    - personalized recommendation score
    - metadata-based content similarity
    - semantic text similarity
    - collaborative interaction signal
    - current emotion relevance
    - historical emotional trend
    - user recommendation feedback
    """

    # ------------------------------------------------
    # LOAD USER / CONTENT DATA
    # ------------------------------------------------

    profile = get_user_profile(
        user_id
    )

    user_history = get_user_history(
        user_id
    )

    all_history = load_user_history()

    wellness_content = load_wellness_content()

    # ------------------------------------------------
    # PERSONALIZED RECOMMENDATION SCORE
    # ------------------------------------------------

    personalized_scores = (
        generate_personalized_scores(
            user_id=user_id,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
        )
    )

    # ------------------------------------------------
    # CONTENT-BASED SIMILARITY
    # ------------------------------------------------

    content_scores = (
        calculate_content_similarity_for_all(
            detected_emotions=detected_emotions,
            preferred_categories=profile[
                "preferred_categories"
            ],
            preferred_activity_types=profile[
                "preferred_activity_types"
            ],
        )
    )

    content_score_map = {
        item["content_id"]:
            item["content_similarity"]
        for item in content_scores
    }

    # ------------------------------------------------
    # SEMANTIC MATCHING
    # ------------------------------------------------

    semantic_results = (
        semantic_match_wellness_content(
            user_text=user_text,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            emotional_state=emotional_state,
            wellness_content=wellness_content,
            top_k=len(wellness_content),
            minimum_similarity=0.0,
        )
    )

    semantic_score_map = {
        row["content_id"]:
            float(
                row["semantic_relevance"]
            )
        for _, row in semantic_results.iterrows()
    }

    # ------------------------------------------------
    # EMOTIONAL TREND ANALYSIS
    # ------------------------------------------------

    trend_analysis = analyze_emotion_trends(
        user_id=user_id,
        current_emotions=detected_emotions,
    )

    historical_trend_signal = (
        trend_analysis["trend_signal"]
    )

    # ------------------------------------------------
    # BUILD HYBRID SCORES
    # ------------------------------------------------

    hybrid_recommendations = []

    for recommendation in personalized_scores:

        content_id = (
            recommendation["content_id"]
        )

        # --------------------------------------------
        # Existing recommendation signals
        # --------------------------------------------

        content_similarity = (
            content_score_map.get(
                content_id,
                0.0,
            )
        )

        semantic_relevance = (
            semantic_score_map.get(
                content_id,
                0.0,
            )
        )

        collaborative_signal = (
            calculate_collaborative_signal(
                content_id=content_id,
                user_id=user_id,
                user_history=all_history,
            )
        )

        emotion_match = (
            recommendation[
                "emotion_match"
            ]
        )

        personalized_score = (
            recommendation[
                "recommendation_score"
            ]
        )

        # --------------------------------------------
        # NEW FEEDBACK SIGNAL
        # --------------------------------------------

        feedback_signal = (
            calculate_feedback_signal(
                user_id=user_id,
                content_id=content_id,
            )
        )

        # --------------------------------------------
        # FINAL HYBRID SCORE
        #
        # Personalized score:      30%
        # Content similarity:      15%
        # Semantic relevance:      20%
        # Collaborative signal:    10%
        # Current emotion match:   10%
        # Historical trend:         5%
        # User feedback:           10%
        #
        # Total:                  100%
        # --------------------------------------------

        hybrid_score = (
            (personalized_score * 0.30)
            + (content_similarity * 0.15)
            + (semantic_relevance * 0.20)
            + (collaborative_signal * 0.10)
            + (emotion_match * 0.10)
            + (historical_trend_signal * 0.05)
            + (feedback_signal * 0.10)
        )

        hybrid_recommendations.append(
            {
                **recommendation,

                "content_similarity":
                    round(
                        content_similarity,
                        4,
                    ),

                "semantic_relevance":
                    round(
                        semantic_relevance,
                        4,
                    ),

                "collaborative_signal":
                    round(
                        collaborative_signal,
                        4,
                    ),

                "historical_trend_signal":
                    round(
                        historical_trend_signal,
                        4,
                    ),

                "feedback_signal":
                    round(
                        feedback_signal,
                        4,
                    ),

                "hybrid_score":
                    round(
                        hybrid_score,
                        4,
                    ),
            }
        )

    # ------------------------------------------------
    # CREATE RESULT DATAFRAME
    # ------------------------------------------------

    result_dataframe = pd.DataFrame(
        hybrid_recommendations
    )

    if result_dataframe.empty:
        return result_dataframe

    # ------------------------------------------------
    # DYNAMIC SORTING
    # ------------------------------------------------

    result_dataframe = (
        result_dataframe.sort_values(
            by=[
                "hybrid_score",
                "feedback_signal",
                "historical_trend_signal",
                "semantic_relevance",
                "content_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------
    # ASSIGN HYBRID RANK
    # ------------------------------------------------

    result_dataframe["hybrid_rank"] = (
        result_dataframe.index + 1
    )

    return result_dataframe
import pandas as pd


DEFAULT_MIN_RELEVANCE = 0.30
DEFAULT_TOP_K = 5


def validate_recommendation_scores(recommendations):
    """
    Validate that recommendation scores are numeric
    and fall within the expected [0, 1] range.
    """

    if recommendations.empty:
        return True

    required_columns = {
        "content_id",
        "title",
        "hybrid_score",
    }

    missing_columns = required_columns - set(
        recommendations.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing ranking columns: {sorted(missing_columns)}"
        )

    scores = pd.to_numeric(
        recommendations["hybrid_score"],
        errors="coerce",
    )

    if scores.isna().any():
        raise ValueError(
            "Recommendation scores contain non-numeric values."
        )

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(
            "Recommendation scores must be between 0 and 1."
        )

    return True


def detect_duplicate_recommendations(recommendations):
    """
    Detect duplicate content IDs in the recommendation set.
    """

    if recommendations.empty:
        return []

    duplicate_mask = recommendations["content_id"].duplicated(
        keep=False
    )

    duplicates = recommendations.loc[
        duplicate_mask,
        "content_id",
    ].tolist()

    return sorted(set(duplicates))


def remove_duplicate_recommendations(recommendations):
    """
    Remove duplicate content items while keeping the
    highest-scoring occurrence.
    """

    if recommendations.empty:
        return recommendations.copy()

    sorted_recommendations = recommendations.sort_values(
        by=["hybrid_score", "content_id"],
        ascending=[False, True],
    )

    deduplicated = sorted_recommendations.drop_duplicates(
        subset=["content_id"],
        keep="first",
    )

    return deduplicated.reset_index(drop=True)


def filter_low_relevance(
    recommendations,
    minimum_score=DEFAULT_MIN_RELEVANCE,
):
    """
    Remove recommendations below the minimum relevance score.
    """

    if recommendations.empty:
        return recommendations.copy()

    filtered = recommendations[
        recommendations["hybrid_score"] >= minimum_score
    ].copy()

    return filtered.reset_index(drop=True)


def rank_recommendations(
    recommendations,
    top_k=DEFAULT_TOP_K,
    minimum_score=DEFAULT_MIN_RELEVANCE,
):
    """
    Perform the complete recommendation ranking process.

    Steps:
    1. Validate scores
    2. Detect duplicates
    3. Remove duplicates
    4. Remove low-relevance items
    5. Sort dynamically by score
    6. Assign final ranking positions
    7. Return the top K results
    """

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError(
            "Recommendations must be a pandas DataFrame."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if minimum_score < 0 or minimum_score > 1:
        raise ValueError(
            "minimum_score must be between 0 and 1."
        )

    validate_recommendation_scores(
        recommendations
    )

    duplicates_before_removal = (
        detect_duplicate_recommendations(
            recommendations
        )
    )

    deduplicated = remove_duplicate_recommendations(
        recommendations
    )

    filtered = filter_low_relevance(
        recommendations=deduplicated,
        minimum_score=minimum_score,
    )

    # Sort by score first.
    # content_id provides deterministic ordering when
    # two recommendations have exactly the same score.
    ranked = filtered.sort_values(
        by=["hybrid_score", "content_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranked["rank"] = ranked.index + 1

    final_recommendations = ranked.head(
        top_k
    ).copy()

    return {
        "recommendations": final_recommendations,
        "duplicates_detected": duplicates_before_removal,
        "duplicates_removed": len(
            duplicates_before_removal
        ),
        "low_relevance_removed": len(
            deduplicated
        ) - len(filtered),
        "total_candidates": len(
            recommendations
        ),
        "total_ranked": len(
            ranked
        ),
    }


def get_top_recommendation(
    ranking_result,
):
    """
    Return the highest-ranked recommendation.
    """

    recommendations = ranking_result[
        "recommendations"
    ]

    if recommendations.empty:
        return None

    return recommendations.iloc[0].to_dict()
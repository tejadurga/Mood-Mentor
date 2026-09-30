from pathlib import Path
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "recommendation_eval.csv"
)

TOP_K = 5


def parse_ids(value):
    """
    Convert semicolon-separated content IDs into a set.
    """

    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def load_evaluation_dataset():
    """
    Load the controlled recommendation evaluation dataset.
    """

    if not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{EVALUATION_DATA_PATH}"
        )

    dataframe = pd.read_csv(
        EVALUATION_DATA_PATH
    )

    required_columns = {
        "scenario_id",
        "user_id",
        "user_text",
        "detected_emotions",
        "emotion_intensity",
        "expected_relevant_content_ids",
        "expected_accepted_content_ids",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["detected_emotions"] = (
        dataframe["detected_emotions"]
        .apply(parse_ids)
    )

    dataframe["expected_relevant_content_ids"] = (
        dataframe[
            "expected_relevant_content_ids"
        ]
        .apply(parse_ids)
    )

    dataframe["expected_accepted_content_ids"] = (
        dataframe[
            "expected_accepted_content_ids"
        ]
        .apply(parse_ids)
    )

    dataframe["emotion_intensity"] = pd.to_numeric(
        dataframe["emotion_intensity"],
        errors="coerce"
    )

    return dataframe


def precision_at_k(
    recommended_ids,
    relevant_ids,
    k=TOP_K,
):
    """
    Precision@K = relevant recommendations / K.
    """

    top_k = recommended_ids[:k]

    if not top_k:
        return 0.0

    hits = sum(
        item in relevant_ids
        for item in top_k
    )

    return hits / len(top_k)


def recall_at_k(
    recommended_ids,
    relevant_ids,
    k=TOP_K,
):
    """
    Recall@K = relevant recommendations retrieved /
    total relevant recommendations.
    """

    if not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]

    hits = sum(
        item in relevant_ids
        for item in top_k
    )

    return hits / len(relevant_ids)


def f1_at_k(
    recommended_ids,
    relevant_ids,
    k=TOP_K,
):
    """
    F1 score using Precision@K and Recall@K.
    """

    precision = precision_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    recall = recall_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def ndcg_at_k(
    recommended_ids,
    relevant_ids,
    k=TOP_K,
):
    """
    Normalized Discounted Cumulative Gain@K.

    Relevant items receive relevance 1.
    """

    top_k = recommended_ids[:k]

    dcg = 0.0

    for index, content_id in enumerate(
        top_k,
        start=1
    ):

        if content_id in relevant_ids:

            dcg += (
                1.0
                / __import__("math").log2(
                    index + 1
                )
            )

    ideal_hits = min(
        len(relevant_ids),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0
        / __import__("math").log2(
            index + 1
        )
        for index in range(
            1,
            ideal_hits + 1
        )
    )

    return dcg / idcg


def reciprocal_rank(
    recommended_ids,
    relevant_ids,
):
    """
    Reciprocal rank of the first relevant recommendation.
    """

    for index, content_id in enumerate(
        recommended_ids,
        start=1
    ):

        if content_id in relevant_ids:
            return 1.0 / index

    return 0.0


def acceptance_proxy_at_k(
    recommended_ids,
    accepted_ids,
    k=TOP_K,
):
    """
    Offline acceptance proxy using controlled
    expected acceptance labels.

    This is NOT observed real-user acceptance.
    """

    top_k = recommended_ids[:k]

    if not top_k:
        return 0.0

    accepted = sum(
        item in accepted_ids
        for item in top_k
    )

    return accepted / len(top_k)


def recommendation_diversity(
    recommendations,
    k=TOP_K,
):
    """
    Calculate category-based recommendation diversity.

    Diversity = unique categories / number of
    displayed recommendations.
    """

    top_k = recommendations[:k]

    if top_k.empty:
        return 0.0

    unique_categories = (
        top_k["category"]
        .nunique()
    )

    return (
        unique_categories
        / len(top_k)
    )


def calculate_metrics(
    recommended_dataframe,
    relevant_ids,
    accepted_ids,
    k=TOP_K,
):
    """
    Calculate all offline recommendation metrics.
    """

    recommended_ids = (
        recommended_dataframe[
            "content_id"
        ]
        .tolist()
    )

    precision = precision_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    recall = recall_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    f1 = f1_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    ndcg = ndcg_at_k(
        recommended_ids,
        relevant_ids,
        k
    )

    rr = reciprocal_rank(
        recommended_ids,
        relevant_ids
    )

    acceptance = acceptance_proxy_at_k(
        recommended_ids,
        accepted_ids,
        k
    )

    diversity = recommendation_diversity(
        recommended_dataframe,
        k
    )

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "ndcg_at_k": ndcg,
        "reciprocal_rank": rr,
        "acceptance_proxy_at_k": acceptance,
        "diversity": diversity,
    }


def aggregate_metrics(
    scenario_results
):
    """
    Calculate average metrics across scenarios.
    """

    if not scenario_results:
        return {}

    metrics = [
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "ndcg_at_k",
        "reciprocal_rank",
        "acceptance_proxy_at_k",
        "diversity",
        "response_time_seconds",
    ]

    return {
        metric: sum(
            result[metric]
            for result in scenario_results
        ) / len(scenario_results)
        for metric in metrics
    }


def calculate_observed_feedback_acceptance():
    """
    Calculate observed acceptance rate from stored feedback.

    This uses actual stored interaction records in the
    current project. It is separate from the offline
    acceptance proxy used for controlled evaluation.
    """

    feedback_path = (
        PROJECT_ROOT
        / "data"
        / "recommendation_feedback.csv"
    )

    if not feedback_path.exists():
        return None

    feedback = pd.read_csv(
        feedback_path
    )

    if feedback.empty:
        return None

    valid_interactions = feedback[
        feedback["interaction_type"].isin(
            [
                "accepted",
                "rejected",
            ]
        )
    ]

    if valid_interactions.empty:
        return None

    accepted_count = (
        valid_interactions[
            "interaction_type"
        ]
        == "accepted"
    ).sum()

    total_decisions = len(
        valid_interactions
    )

    return (
        accepted_count
        / total_decisions
    )


def compare_models(
    baseline_results,
    advanced_results,
):
    """
    Compare average baseline and advanced metrics.
    """

    baseline_average = aggregate_metrics(
        baseline_results
    )

    advanced_average = aggregate_metrics(
        advanced_results
    )

    comparison_rows = []

    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "ndcg_at_k",
        "reciprocal_rank",
        "acceptance_proxy_at_k",
        "diversity",
        "response_time_seconds",
    ]

    for metric in metric_names:

        baseline_value = (
            baseline_average[metric]
        )

        advanced_value = (
            advanced_average[metric]
        )

        difference = (
            advanced_value
            - baseline_value
        )

        comparison_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "advanced": advanced_value,
                "difference": difference,
            }
        )

    return pd.DataFrame(
        comparison_rows
    )
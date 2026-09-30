import time

import pandas as pd

from src.recommendation_data import (
    load_wellness_content,
)

from src.recommendation_engine import (
    build_hybrid_recommendations,
)

from src.recommendation_ranking import (
    rank_recommendations,
)

from src.recommendation_evaluation import (
    load_evaluation_dataset,
    calculate_metrics,
    aggregate_metrics,
    compare_models,
    calculate_observed_feedback_acceptance,
    TOP_K,
)


def build_baseline_recommendations(
    detected_emotions,
    wellness_content,
):
    """
    Simple baseline recommender.

    It uses only current emotion overlap and does not
    use preferences, history, semantic similarity,
    trend, feedback, or hybrid ranking.
    """

    rows = []

    detected_set = {
        emotion.lower()
        for emotion in detected_emotions
    }

    for _, content in wellness_content.iterrows():

        target_set = {
            str(emotion).lower()
            for emotion in content[
                "target_emotions"
            ]
        }

        overlap = (
            detected_set.intersection(
                target_set
            )
        )

        if detected_set:

            emotion_score = (
                len(overlap)
                / len(detected_set)
            )

        else:

            emotion_score = 0.0

        rows.append(
            {
                "content_id":
                    content["content_id"],

                "title":
                    content["title"],

                "category":
                    content["category"],

                "activity_type":
                    content["activity_type"],

                "duration_minutes":
                    content["duration_minutes"],

                "emotion_score":
                    emotion_score,
            }
        )

    baseline = pd.DataFrame(
        rows
    )

    baseline = baseline.sort_values(
        by=[
            "emotion_score",
            "content_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    baseline["baseline_rank"] = (
        baseline.index + 1
    )

    return baseline


def evaluate_baseline(
    scenario,
    wellness_content,
):
    """
    Evaluate the baseline recommender for one scenario.
    """

    start_time = time.perf_counter()

    baseline = build_baseline_recommendations(
        detected_emotions=(
            scenario[
                "detected_emotions"
            ]
        ),
        wellness_content=wellness_content,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    top_k = baseline.head(
        TOP_K
    ).copy()

    metrics = calculate_metrics(
        recommended_dataframe=top_k.rename(
            columns={
                "emotion_score":
                    "hybrid_score"
            }
        ),
        relevant_ids=scenario[
            "expected_relevant_content_ids"
        ],
        accepted_ids=scenario[
            "expected_accepted_content_ids"
        ],
        k=TOP_K,
    )

    metrics["response_time_seconds"] = (
        elapsed
    )

    metrics["scenario_id"] = (
        scenario["scenario_id"]
    )

    metrics["top_recommendations"] = (
        top_k["content_id"].tolist()
    )

    return metrics


def evaluate_advanced(
    scenario,
):
    """
    Evaluate the complete advanced recommendation
    pipeline for one scenario.
    """

    start_time = time.perf_counter()

    hybrid = (
        build_hybrid_recommendations(
            user_id=scenario["user_id"],
            user_text=scenario["user_text"],
            detected_emotions=(
                list(
                    scenario[
                        "detected_emotions"
                    ]
                )
            ),
            emotion_intensity=float(
                scenario[
                    "emotion_intensity"
                ]
            ),
            emotional_state=(
                "Evaluation emotional state"
            ),
        )
    )

    ranking = rank_recommendations(
        recommendations=hybrid,
        top_k=TOP_K,
        minimum_score=0.30,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    top_k = ranking[
        "recommendations"
    ].copy()

    metrics = calculate_metrics(
        recommended_dataframe=top_k,
        relevant_ids=scenario[
            "expected_relevant_content_ids"
        ],
        accepted_ids=scenario[
            "expected_accepted_content_ids"
        ],
        k=TOP_K,
    )

    metrics["response_time_seconds"] = (
        elapsed
    )

    metrics["scenario_id"] = (
        scenario["scenario_id"]
    )

    metrics["top_recommendations"] = (
        top_k["content_id"].tolist()
    )

    return metrics


def format_percentage(value):
    return f"{value * 100:.2f}%"


def main():

    print("=" * 80)
    print(
        "MOOD MENTOR - RECOMMENDATION "
        "VALIDATION & PERFORMANCE TEST"
    )
    print("=" * 80)

    dataset = load_evaluation_dataset()

    wellness_content = load_wellness_content()

    print(
        f"\nEvaluation scenarios: "
        f"{len(dataset)}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    baseline_results = []

    advanced_results = []

    # ------------------------------------------------
    # Run scenario evaluations
    # ------------------------------------------------

    for _, scenario in dataset.iterrows():

        print("\n" + "-" * 80)

        print(
            f"Scenario: "
            f"{scenario['scenario_id']}"
        )

        print(
            f"User: "
            f"{scenario['user_id']}"
        )

        print(
            f"Text: "
            f"{scenario['user_text']}"
        )

        print(
            "Emotions: "
            + ", ".join(
                scenario[
                    "detected_emotions"
                ]
            )
        )

        baseline = evaluate_baseline(
            scenario=scenario,
            wellness_content=(
                wellness_content
            ),
        )

        advanced = evaluate_advanced(
            scenario=scenario
        )

        baseline_results.append(
            baseline
        )

        advanced_results.append(
            advanced
        )

        print(
            "\nBaseline Top-5:"
        )

        print(
            "  "
            + " > ".join(
                baseline[
                    "top_recommendations"
                ]
            )
        )

        print(
            "Advanced Top-5:"
        )

        print(
            "  "
            + " > ".join(
                advanced[
                    "top_recommendations"
                ]
            )
        )

        print(
            "\nBaseline metrics:"
        )

        print(
            f"  Precision@5: "
            f"{format_percentage(baseline['precision_at_k'])}"
        )

        print(
            f"  Recall@5: "
            f"{format_percentage(baseline['recall_at_k'])}"
        )

        print(
            f"  F1@5: "
            f"{format_percentage(baseline['f1_at_k'])}"
        )

        print(
            f"  NDCG@5: "
            f"{baseline['ndcg_at_k']:.4f}"
        )

        print(
            f"  Diversity: "
            f"{format_percentage(baseline['diversity'])}"
        )

        print(
            f"  Response time: "
            f"{baseline['response_time_seconds']:.4f}s"
        )

        print(
            "\nAdvanced metrics:"
        )

        print(
            f"  Precision@5: "
            f"{format_percentage(advanced['precision_at_k'])}"
        )

        print(
            f"  Recall@5: "
            f"{format_percentage(advanced['recall_at_k'])}"
        )

        print(
            f"  F1@5: "
            f"{format_percentage(advanced['f1_at_k'])}"
        )

        print(
            f"  NDCG@5: "
            f"{advanced['ndcg_at_k']:.4f}"
        )

        print(
            f"  Diversity: "
            f"{format_percentage(advanced['diversity'])}"
        )

        print(
            f"  Response time: "
            f"{advanced['response_time_seconds']:.4f}s"
        )

    # ------------------------------------------------
    # Aggregate comparison
    # ------------------------------------------------

    baseline_average = aggregate_metrics(
        baseline_results
    )

    advanced_average = aggregate_metrics(
        advanced_results
    )

    comparison = compare_models(
        baseline_results=baseline_results,
        advanced_results=advanced_results,
    )

    print("\n" + "=" * 80)
    print(
        "BASELINE VS ADVANCED COMPARISON"
    )
    print("=" * 80)

    print(
        f"\n{'Metric':<28}"
        f"{'Baseline':<18}"
        f"{'Advanced':<18}"
        f"{'Difference':<18}"
    )

    print("-" * 80)

    for _, row in comparison.iterrows():

        metric = row["metric"]

        baseline_value = row[
            "baseline"
        ]

        advanced_value = row[
            "advanced"
        ]

        difference = row[
            "difference"
        ]

        if metric == "response_time_seconds":

            print(
                f"{metric:<28}"
                f"{baseline_value:<18.4f}"
                f"{advanced_value:<18.4f}"
                f"{difference:<18.4f}"
            )

        elif metric in {
            "ndcg_at_k",
            "reciprocal_rank",
        }:

            print(
                f"{metric:<28}"
                f"{baseline_value:<18.4f}"
                f"{advanced_value:<18.4f}"
                f"{difference:<18.4f}"
            )

        else:

            print(
                f"{metric:<28}"
                f"{format_percentage(baseline_value):<18}"
                f"{format_percentage(advanced_value):<18}"
                f"{format_percentage(difference):<18}"
            )

    # ------------------------------------------------
    # Observed feedback
    # ------------------------------------------------

    observed_acceptance = (
        calculate_observed_feedback_acceptance()
    )

    print("\n" + "=" * 80)
    print(
        "OBSERVED FEEDBACK ACCEPTANCE"
    )
    print("=" * 80)

    if observed_acceptance is None:

        print(
            "\nNo observed accepted/rejected "
            "feedback records are available."
        )

    else:

        print(
            "\nObserved acceptance rate from "
            "stored user feedback: "
            f"{observed_acceptance * 100:.2f}%"
        )

    # ------------------------------------------------
    # Final notes
    # ------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "VALIDATION NOTES"
    )
    print("=" * 80)

    print(
        "\n1. Precision@5, Recall@5, F1@5 and "
        "NDCG@5 are offline metrics."
    )

    print(
        "2. The evaluation ground truth is a "
        "controlled synthetic test dataset."
    )

    print(
        "3. Acceptance@5 is an offline proxy "
        "based on expected accepted items."
    )

    print(
        "4. Observed acceptance rate uses the "
        "actual stored feedback CSV."
    )

    print(
        "5. Advanced response time includes the "
        "semantic/recommendation pipeline."
    )

    print(
        "\nRecommendation validation completed."
    )


if __name__ == "__main__":
    main()
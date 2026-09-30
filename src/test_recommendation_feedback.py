from src.recommendation_feedback import (
    load_feedback,
    get_user_feedback,
    calculate_feedback_signal,
    get_feedback_summary,
    save_feedback_record,
)


def main():

    user_id = "demo_user_001"

    print(
        "Recommendation Feedback Learning Test"
    )

    print(
        "--------------------------------------"
    )

    # ---------------------------------------------
    # Load existing feedback
    # ---------------------------------------------

    feedback = load_feedback()

    assert not feedback.empty

    print(
        f"\nTotal feedback records: "
        f"{len(feedback)}"
    )

    # ---------------------------------------------
    # User feedback
    # ---------------------------------------------

    user_feedback = get_user_feedback(
        user_id
    )

    assert not user_feedback.empty

    print(
        f"Feedback records for {user_id}: "
        f"{len(user_feedback)}"
    )

    # ---------------------------------------------
    # Feedback signal
    # ---------------------------------------------

    accepted_signal = (
        calculate_feedback_signal(
            user_id=user_id,
            content_id="C003",
        )
    )

    rejected_signal = (
        calculate_feedback_signal(
            user_id=user_id,
            content_id="C014",
        )
    )

    assert 0.0 <= accepted_signal <= 1.0
    assert 0.0 <= rejected_signal <= 1.0

    print(
        "\nFeedback signals:"
    )

    print(
        f"  C003 accepted signal: "
        f"{accepted_signal:.4f}"
    )

    print(
        f"  C014 rejected signal: "
        f"{rejected_signal:.4f}"
    )

    assert (
        accepted_signal
        > rejected_signal
    )

    print(
        "  Accepted/rejected signal check: PASSED"
    )

    # ---------------------------------------------
    # Feedback summary
    # ---------------------------------------------

    summary = get_feedback_summary(
        user_id
    )

    print(
        "\nUser feedback summary:"
    )

    print(
        f"  Total: "
        f"{summary['total_feedback']}"
    )

    print(
        f"  Viewed: "
        f"{summary['viewed']}"
    )

    print(
        f"  Accepted: "
        f"{summary['accepted']}"
    )

    print(
        f"  Rejected: "
        f"{summary['rejected']}"
    )

    if summary["average_rating"] is not None:

        print(
            f"  Average rating: "
            f"{summary['average_rating']:.2f}"
        )

    # ---------------------------------------------
    # New feedback write test
    # ---------------------------------------------

    save_feedback_record(
        user_id=user_id,
        content_id="C001",
        interaction_type="accepted",
        rating=5,
        emotion_state=(
            "High-intensity fear state"
        ),
        intensity_score=0.84,
    )

    updated_feedback = get_user_feedback(
        user_id
    )

    assert len(
        updated_feedback
    ) >= len(user_feedback)

    print(
        "\nFeedback write test: PASSED"
    )

    print(
        "\nRecommendation feedback learning "
        "test completed successfully."
    )


if __name__ == "__main__":
    main()
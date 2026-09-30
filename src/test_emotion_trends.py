from src.emotion_trends import (
    get_user_emotion_history,
    calculate_emotion_frequency,
    calculate_emotion_intensity_trend,
    calculate_polarity_trend,
    detect_repeated_emotional_patterns,
    get_recent_emotional_state,
    calculate_emotional_trend_signal,
    analyze_emotion_trends,
)


def main():

    user_id = "demo_user_001"

    history = get_user_emotion_history(
        user_id
    )

    assert not history.empty

    print(
        "Emotional Trend & User State Tracking Test"
    )

    print(
        "-------------------------------------------"
    )

    print(
        f"User: {user_id}"
    )

    print(
        f"Historical records: {len(history)}"
    )

    # ---------------------------------------------
    # Emotion frequency
    # ---------------------------------------------

    frequency = calculate_emotion_frequency(
        history
    )

    assert frequency

    print(
        "\nEmotion frequency:"
    )

    for emotion, count in frequency.items():

        print(
            f"  {emotion}: {count}"
        )

    # ---------------------------------------------
    # Intensity trend
    # ---------------------------------------------

    intensity_trend = (
        calculate_emotion_intensity_trend(
            history
        )
    )

    assert "direction" in intensity_trend

    print(
        "\nEmotion intensity trend:"
    )

    print(
        f"  Earlier average: "
        f"{intensity_trend['earlier_average']:.4f}"
    )

    print(
        f"  Recent average: "
        f"{intensity_trend['recent_average']:.4f}"
    )

    print(
        f"  Change: "
        f"{intensity_trend['change']:.4f}"
    )

    print(
        f"  Direction: "
        f"{intensity_trend['direction']}"
    )

    # ---------------------------------------------
    # Polarity trend
    # ---------------------------------------------

    polarity_trend = (
        calculate_polarity_trend(
            history
        )
    )

    print(
        "\nPolarity trend:"
    )

    print(
        f"  Earlier average: "
        f"{polarity_trend['earlier_average']:.4f}"
    )

    print(
        f"  Recent average: "
        f"{polarity_trend['recent_average']:.4f}"
    )

    print(
        f"  Change: "
        f"{polarity_trend['change']:.4f}"
    )

    print(
        f"  Direction: "
        f"{polarity_trend['direction']}"
    )

    # ---------------------------------------------
    # Repeated patterns
    # ---------------------------------------------

    repeated_patterns = (
        detect_repeated_emotional_patterns(
            history
        )
    )

    assert repeated_patterns

    print(
        "\nRepeated emotional patterns:"
    )

    for emotion, count in (
        repeated_patterns.items()
    ):

        print(
            f"  {emotion}: "
            f"{count} occurrences"
        )

    # ---------------------------------------------
    # Recent state
    # ---------------------------------------------

    recent_state = (
        get_recent_emotional_state(
            history
        )
    )

    assert recent_state is not None

    print(
        "\nRecent emotional state:"
    )

    print(
        f"  Dominant emotion: "
        f"{recent_state['dominant_emotion']}"
    )

    print(
        f"  Intensity: "
        f"{recent_state['intensity_score']:.4f}"
    )

    print(
        f"  Intensity level: "
        f"{recent_state['intensity_level']}"
    )

    print(
        f"  Polarity: "
        f"{recent_state['polarity']}"
    )

    print(
        f"  Emotional state: "
        f"{recent_state['emotional_state']}"
    )

    # ---------------------------------------------
    # Trend signal
    # ---------------------------------------------

    current_emotions = [
        "fear",
        "sadness",
    ]

    trend_signal = (
        calculate_emotional_trend_signal(
            detected_emotions=current_emotions,
            user_history=history
        )
    )

    assert 0.0 <= trend_signal <= 1.0

    print(
        "\nCurrent emotion trend signal:"
    )

    print(
        f"  Current emotions: "
        f"{', '.join(current_emotions)}"
    )

    print(
        f"  Historical pattern signal: "
        f"{trend_signal:.4f}"
    )

    # ---------------------------------------------
    # Complete analysis
    # ---------------------------------------------

    analysis = analyze_emotion_trends(
        user_id=user_id,
        current_emotions=current_emotions
    )

    assert analysis["history_records"] > 0
    assert analysis["trend_signal"] >= 0.0

    print(
        "\nComplete historical analysis:"
    )

    print(
        f"  Records: "
        f"{analysis['history_records']}"
    )

    print(
        f"  Trend signal: "
        f"{analysis['trend_signal']:.4f}"
    )

    print(
        "\nEmotional trend tracking test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
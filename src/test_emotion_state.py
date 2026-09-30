from src.emotion_state import analyze_emotional_state


def main():

    print("=" * 70)
    print("MOOD MENTOR - EMOTIONAL STATE TEST")
    print("=" * 70)

    # Example probability output from BERT.
    # These values are test inputs for the state-analysis
    # logic and are not model predictions.
    probabilities = [
        0.82,  # joy
        0.04,  # sadness
        0.02,  # anger
        0.71,  # fear
        0.08,  # surprise
        0.01,  # disgust
    ]

    sentiment_result = {
        "success": True,
        "sentiment": "Positive",
        "compound": 0.55,
    }

    result = analyze_emotional_state(
        probabilities,
        sentiment_result,
        threshold=0.50
    )

    print("\nDominant emotion:")
    print(
        result["dominant_emotion"]
    )

    print("\nDominant confidence:")
    print(
        f"{result['dominant_confidence']:.4f}"
    )

    print("\nDetected emotions:")
    print(
        ", ".join(
            result["detected_emotions"]
        )
    )

    print("\nEmotion intensity:")
    print(
        f"{result['emotion_intensity']:.4f}"
    )

    print("\nIntensity level:")
    print(
        result["intensity_level"]
    )

    print("\nPolarity:")
    print(
        result["polarity"]
    )

    print("\nPolarity score:")
    print(
        f"{result['polarity_score']:.4f}"
    )

    print("\nMixed emotional state:")
    print(
        result["is_mixed"]
    )

    print("\nEmotion severity:")
    print(
        result["emotion_severity"]
    )

    print("\nFinal emotional state:")
    print(
        result["emotional_state"]
    )

    print("\n" + "=" * 70)
    print("EMOTIONAL STATE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
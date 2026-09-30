import numpy as np

from src.emotion_config import EMOTION_LABELS


# --------------------------------------------------
# Emotion State Configuration
# --------------------------------------------------

DEFAULT_THRESHOLD = 0.50


def calculate_emotion_intensity(
    probabilities
):
    """
    Calculate emotional intensity dynamically from
    the strongest model-generated emotion probability.

    The resulting value is normalized between 0 and 1.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float
    )

    if probability_array.size == 0:
        return 0.0

    intensity = float(
        np.max(probability_array)
    )

    return float(
        np.clip(
            intensity,
            0.0,
            1.0
        )
    )


def classify_intensity(
    intensity
):
    """
    Convert the numerical intensity value into
    a descriptive intensity level.
    """

    if intensity < 0.30:
        return "Low"

    if intensity < 0.60:
        return "Moderate"

    if intensity < 0.80:
        return "High"

    return "Very High"


def determine_severity(
    intensity
):
    """
    Convert emotional intensity into a descriptive
    emotional severity level.

    This is a heuristic classification for the project
    and is not a clinical severity assessment.
    """

    if intensity < 0.30:
        return "Low"

    if intensity < 0.60:
        return "Moderate"

    if intensity < 0.80:
        return "High"

    return "Very High"


def build_emotion_results(
    probabilities,
    threshold=DEFAULT_THRESHOLD
):
    """
    Convert BERT probabilities into structured
    emotion results.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float
    )

    if len(probability_array) != len(
        EMOTION_LABELS
    ):

        raise ValueError(
            "Number of probabilities must match "
            "the number of configured emotions."
        )

    emotion_results = []

    for emotion, probability in zip(
        EMOTION_LABELS,
        probability_array
    ):

        confidence = float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )

        emotion_results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "detected":
                    confidence >= threshold
            }
        )

    return emotion_results


def analyze_emotional_state(
    probabilities,
    sentiment_result,
    threshold=DEFAULT_THRESHOLD
):
    """
    Build the complete emotional state from BERT
    emotion probabilities and VADER sentiment output.
    """

    probability_array = np.asarray(
        probabilities,
        dtype=float
    )

    if probability_array.size == 0:

        raise ValueError(
            "Emotion probabilities cannot be empty."
        )

    if len(probability_array) != len(
        EMOTION_LABELS
    ):

        raise ValueError(
            "Emotion probability count does not "
            "match the configured emotion labels."
        )

    # --------------------------------------------------
    # Emotion Results
    # --------------------------------------------------

    emotion_results = build_emotion_results(
        probability_array,
        threshold
    )

    # --------------------------------------------------
    # Dominant Emotion
    # --------------------------------------------------

    dominant_index = int(
        np.argmax(
            probability_array
        )
    )

    dominant_emotion = EMOTION_LABELS[
        dominant_index
    ]

    dominant_confidence = float(
        probability_array[
            dominant_index
        ]
    )

    # --------------------------------------------------
    # Multiple Emotions
    # --------------------------------------------------

    detected_emotions = [
        result["emotion"]
        for result in emotion_results
        if result["detected"]
    ]

    # Always keep the dominant emotion available
    # even if no probability reaches the threshold.
    if not detected_emotions:

        detected_emotions = [
            dominant_emotion
        ]

    # --------------------------------------------------
    # Emotional Intensity
    # --------------------------------------------------

    intensity = calculate_emotion_intensity(
        probability_array
    )

    intensity_level = classify_intensity(
        intensity
    )

    # --------------------------------------------------
    # Emotional Severity
    # --------------------------------------------------

    severity = determine_severity(
        intensity
    )

    # --------------------------------------------------
    # Mixed Emotional State
    # --------------------------------------------------

    is_mixed = (
        len(detected_emotions) > 1
    )

    if is_mixed:

        emotional_state = (
            f"{intensity_level}-intensity "
            "mixed emotional state"
        )

    else:

        emotional_state = (
            f"{intensity_level}-intensity "
            f"{dominant_emotion} state"
        )

    # --------------------------------------------------
    # VADER Polarity
    # --------------------------------------------------

    if not sentiment_result.get(
        "success",
        False
    ):

        polarity = "Unknown"
        polarity_score = None

    else:

        polarity = sentiment_result[
            "sentiment"
        ]

        polarity_score = float(
            sentiment_result[
                "compound"
            ]
        )

    # --------------------------------------------------
    # Final Emotional State
    # --------------------------------------------------

    return {
        "dominant_emotion":
            dominant_emotion,

        "dominant_confidence":
            dominant_confidence,

        "detected_emotions":
            detected_emotions,

        "emotion_results":
            emotion_results,

        "emotion_intensity":
            intensity,

        "intensity_level":
            intensity_level,

        "polarity":
            polarity,

        "polarity_score":
            polarity_score,

        "is_mixed":
            is_mixed,

        "emotion_severity":
            severity,

        "emotional_state":
            emotional_state
    }
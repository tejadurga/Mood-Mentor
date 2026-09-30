def _format_emotion_list(emotions):
    """
    Convert a list of emotions into readable text.
    """

    if not emotions:
        return "the detected emotional state"

    formatted = [
        str(emotion).title()
        for emotion in emotions
    ]

    if len(formatted) == 1:
        return formatted[0]

    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"

    return (
        ", ".join(formatted[:-1])
        + f", and {formatted[-1]}"
    )


def _intensity_reason(emotion_intensity):
    """
    Generate a reason based on current emotion intensity.
    """

    intensity = float(emotion_intensity)

    if intensity >= 0.80:
        return (
            "very high emotional intensity was detected"
        )

    if intensity >= 0.60:
        return (
            "high emotional intensity was detected"
        )

    if intensity >= 0.30:
        return (
            "moderate emotional intensity was detected"
        )

    return (
        "low emotional intensity was detected"
    )


def _preference_reason(preference_match):
    """
    Generate a user-preference explanation.
    """

    score = float(preference_match)

    if score >= 0.80:
        return (
            "it strongly matches the user's preferred "
            "wellness categories or activities"
        )

    if score >= 0.40:
        return (
            "it partially matches the user's "
            "wellness preferences"
        )

    return (
        "it has limited direct preference matching"
    )


def _history_reason(
    history_preference,
    feedback_signal,
):
    """
    Generate an explanation from previous behavior
    and recommendation feedback.
    """

    history_score = float(
        history_preference
    )

    feedback_score = float(
        feedback_signal
    )

    if (
        history_score >= 0.80
        and feedback_score >= 0.80
    ):
        return (
            "similar content was previously completed "
            "or rated positively"
        )

    if history_score >= 0.80:
        return (
            "similar content was previously "
            "completed successfully"
        )

    if feedback_score >= 0.80:
        return (
            "previous feedback for this content "
            "was positive"
        )

    if feedback_score <= 0.30:
        return (
            "previous feedback for this content "
            "was limited or negative"
        )

    return (
        "there is limited previous interaction "
        "history for this item"
    )


def _semantic_reason(
    semantic_relevance,
):
    """
    Generate a semantic relevance explanation.
    """

    score = float(
        semantic_relevance
    )

    if score >= 0.70:
        return (
            "its content is strongly related "
            "to the meaning of the user's message"
        )

    if score >= 0.60:
        return (
            "its content is semantically related "
            "to the user's message"
        )

    return (
        "its semantic relevance is moderate"
    )


def _emotion_reason(
    detected_emotions,
    target_emotions,
):
    """
    Explain the overlap between current emotions
    and the content's target emotions.
    """

    detected_set = {
        str(emotion).lower()
        for emotion in detected_emotions
    }

    target_set = {
        str(emotion).lower()
        for emotion in target_emotions
    }

    matching = sorted(
        detected_set.intersection(
            target_set
        )
    )

    if not matching:
        return None

    readable = _format_emotion_list(
        matching
    )

    return (
        f"it is relevant to the detected "
        f"{readable} emotion"
    )


def generate_recommendation_explanation(
    recommendation,
    detected_emotions,
    emotion_intensity,
    preferred_categories,
    preferred_activity_types,
    historical_trend_signal=0.0,
):
    """
    Generate a dynamic explanation for one recommendation.

    The explanation uses:
    - detected emotion
    - emotion intensity
    - user preferences
    - previous history
    - feedback
    - historical emotional trend
    - semantic relevance
    """

    reasons = []

    # ------------------------------------------------
    # Emotion relevance
    # ------------------------------------------------

    emotion_reason = _emotion_reason(
        detected_emotions=detected_emotions,
        target_emotions=recommendation[
            "target_emotions"
        ],
    )

    if emotion_reason:
        reasons.append(
            emotion_reason
        )

    # ------------------------------------------------
    # Emotion intensity
    # ------------------------------------------------

    reasons.append(
        _intensity_reason(
            emotion_intensity
        )
    )

    # ------------------------------------------------
    # User preference
    # ------------------------------------------------

    preference_match = recommendation[
        "preference_match"
    ]

    reasons.append(
        _preference_reason(
            preference_match
        )
    )

    # ------------------------------------------------
    # Historical behavior + feedback
    # ------------------------------------------------

    history_preference = recommendation[
        "history_preference"
    ]

    feedback_signal = recommendation[
        "feedback_signal"
    ]

    reasons.append(
        _history_reason(
            history_preference=history_preference,
            feedback_signal=feedback_signal,
        )
    )

    # ------------------------------------------------
    # Historical emotional trend
    # ------------------------------------------------

    if historical_trend_signal >= 0.80:

        reasons.append(
            "the current emotions strongly match "
            "a repeated historical emotional pattern"
        )

    elif historical_trend_signal >= 0.50:

        reasons.append(
            "the current emotions have some "
            "similarity to previous emotional patterns"
        )

    # ------------------------------------------------
    # Semantic relevance
    # ------------------------------------------------

    semantic_relevance = recommendation[
        "semantic_relevance"
    ]

    reasons.append(
        _semantic_reason(
            semantic_relevance
        )
    )

    # ------------------------------------------------
    # Preference details
    # ------------------------------------------------

    category = str(
        recommendation["category"]
    )

    activity = str(
        recommendation["activity_type"]
    )

    preferred_category_set = {
        str(item).lower()
        for item in preferred_categories
    }

    preferred_activity_set = {
        str(item).lower()
        for item in preferred_activity_types
    }

    if category.lower() in preferred_category_set:

        reasons.append(
            f"the {category} category matches "
            "a saved user preference"
        )

    elif activity.lower() in preferred_activity_set:

        reasons.append(
            f"the {activity} activity matches "
            "a saved user preference"
        )

    return reasons


def build_explainable_recommendations(
    recommendations,
    detected_emotions,
    emotion_intensity,
    preferred_categories,
    preferred_activity_types,
    historical_trend_signal=0.0,
):
    """
    Add dynamic explanation reasons to every recommendation.
    """

    if recommendations.empty:
        return recommendations.copy()

    result = recommendations.copy()

    explanations = []

    for _, recommendation in result.iterrows():

        reasons = generate_recommendation_explanation(
            recommendation=recommendation,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            preferred_categories=preferred_categories,
            preferred_activity_types=preferred_activity_types,
            historical_trend_signal=(
                historical_trend_signal
            ),
        )

        explanations.append(
            reasons
        )

    result["explanation_reasons"] = explanations

    result["explanation"] = [
        " ".join(
            f"• {reason}."
            for reason in reasons
        )
        for reasons in explanations
    ]

    return result
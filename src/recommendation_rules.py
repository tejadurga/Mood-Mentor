from src.recommendation_data import (
    get_user_profile,
    get_user_history,
    load_wellness_content,
)


def calculate_emotion_match(
    detected_emotions,
    target_emotions,
):
    """
    Calculate how well a recommendation matches
    the user's currently detected emotions.
    """

    if not detected_emotions or not target_emotions:
        return 0.0

    detected_set = {
        emotion.lower()
        for emotion in detected_emotions
    }

    target_set = {
        emotion.lower()
        for emotion in target_emotions
    }

    matching_emotions = detected_set.intersection(target_set)

    if not matching_emotions:
        return 0.0

    return len(matching_emotions) / len(detected_set)


def calculate_intensity_match(
    emotion_intensity,
    content_intensity_level,
):
    """
    Compare the current emotional intensity with the
    recommended content intensity level.
    """

    intensity_level = content_intensity_level.lower()

    if emotion_intensity < 0.30:
        current_level = "low"
    elif emotion_intensity < 0.60:
        current_level = "moderate"
    elif emotion_intensity < 0.80:
        current_level = "high"
    else:
        current_level = "very high"

    compatible_levels = {
        "low": {"low", "moderate"},
        "moderate": {"low", "moderate", "high"},
        "high": {"moderate", "high", "very high"},
        "very high": {"high", "very high"},
    }

    if intensity_level in compatible_levels[current_level]:
        return 1.0

    return 0.0


def calculate_preference_match(
    content,
    user_profile,
):
    """
    Calculate how well a recommendation matches
    the user's stored preferences.
    """

    score = 0.0

    preferred_categories = {
        category.lower()
        for category in user_profile["preferred_categories"]
    }

    preferred_activities = {
        activity.lower()
        for activity in user_profile["preferred_activity_types"]
    }

    preferred_intensities = {
        intensity.lower()
        for intensity in user_profile["preferred_intensity_levels"]
    }

    if str(content["category"]).lower() in preferred_categories:
        score += 0.4

    if str(content["activity_type"]).lower() in preferred_activities:
        score += 0.4

    if str(content["intensity_level"]).lower() in preferred_intensities:
        score += 0.2

    return min(score, 1.0)


def calculate_duration_match(
    duration_minutes,
    max_duration_minutes,
):
    """
    Check whether the content duration fits
    within the user's preferred maximum duration.
    """

    try:
        duration = float(duration_minutes)
        maximum = float(max_duration_minutes)
    except (TypeError, ValueError):
        return 0.0

    if duration <= maximum:
        return 1.0

    if duration <= maximum + 5:
        return 0.5

    return 0.0


def calculate_history_preference(
    content_id,
    user_history,
):
    """
    Learn from previous interactions.

    Completed and helpful content receives a positive score.
    Skipped content receives a lower score.
    """

    if user_history.empty:
        return 0.5

    matching_history = user_history[
        user_history["content_id"] == content_id
    ]

    if matching_history.empty:
        return 0.5

    total_score = 0.0
    total_weight = 0.0

    for _, record in matching_history.iterrows():
        helpfulness = float(record["helpfulness_rating"])

        helpfulness_score = helpfulness / 5.0

        if bool(record["was_completed"]):
            interaction_score = (
                0.6 + (helpfulness_score * 0.4)
            )
        else:
            interaction_score = (
                helpfulness_score * 0.4
            )

        total_score += interaction_score
        total_weight += 1.0

    if total_weight == 0:
        return 0.5

    return total_score / total_weight


def calculate_recommendation_score(
    content,
    detected_emotions,
    emotion_intensity,
    user_profile,
    user_history,
):
    """
    Calculate a personalized recommendation score.

    The score combines:
    - emotion relevance
    - intensity compatibility
    - user preferences
    - duration compatibility
    - historical interaction preference
    """

    emotion_match = calculate_emotion_match(
        detected_emotions=detected_emotions,
        target_emotions=content["target_emotions"],
    )

    intensity_match = calculate_intensity_match(
        emotion_intensity=emotion_intensity,
        content_intensity_level=content["intensity_level"],
    )

    preference_match = calculate_preference_match(
        content=content,
        user_profile=user_profile,
    )

    duration_match = calculate_duration_match(
        duration_minutes=content["duration_minutes"],
        max_duration_minutes=user_profile[
            "max_duration_minutes"
        ],
    )

    history_preference = calculate_history_preference(
        content_id=content["content_id"],
        user_history=user_history,
    )

    final_score = (
        (emotion_match * 0.35)
        + (intensity_match * 0.20)
        + (preference_match * 0.20)
        + (duration_match * 0.10)
        + (history_preference * 0.15)
    )

    return {
        "emotion_match": round(emotion_match, 4),
        "intensity_match": round(intensity_match, 4),
        "preference_match": round(preference_match, 4),
        "duration_match": round(duration_match, 4),
        "history_preference": round(history_preference, 4),
        "recommendation_score": round(final_score, 4),
    }


def generate_personalized_scores(
    user_id,
    detected_emotions,
    emotion_intensity,
):
    """
    Generate personalized scores for all wellness content.
    """

    user_profile = get_user_profile(user_id)
    user_history = get_user_history(user_id)
    wellness_content = load_wellness_content()

    scored_recommendations = []

    for _, content in wellness_content.iterrows():

        score_details = calculate_recommendation_score(
            content=content,
            detected_emotions=detected_emotions,
            emotion_intensity=emotion_intensity,
            user_profile=user_profile,
            user_history=user_history,
        )

        recommendation = {
            "content_id": content["content_id"],
            "title": content["title"],
            "description": content["description"],
            "category": content["category"],
            "target_emotions": content["target_emotions"],
            "intensity_level": content["intensity_level"],
            "activity_type": content["activity_type"],
            "duration_minutes": content["duration_minutes"],
            **score_details,
        }

        scored_recommendations.append(
            recommendation
        )

    return scored_recommendations
from src.recommendation_data import load_wellness_content


def calculate_content_similarity(
    detected_emotions,
    preferred_categories,
    preferred_activity_types,
    content,
):
    """
    Calculate a content-based similarity score using
    emotion, category, and activity metadata.
    """

    detected_emotion_set = {
        emotion.lower()
        for emotion in detected_emotions
    }

    preferred_category_set = {
        category.lower()
        for category in preferred_categories
    }

    preferred_activity_set = {
        activity.lower()
        for activity in preferred_activity_types
    }

    target_emotion_set = {
        emotion.lower()
        for emotion in content["target_emotions"]
    }

    emotion_score = 0.0
    category_score = 0.0
    activity_score = 0.0

    if detected_emotion_set and target_emotion_set:
        emotion_overlap = detected_emotion_set.intersection(
            target_emotion_set
        )

        emotion_score = len(emotion_overlap) / len(
            detected_emotion_set
        )

    if str(content["category"]).lower() in preferred_category_set:
        category_score = 1.0

    if str(content["activity_type"]).lower() in preferred_activity_set:
        activity_score = 1.0

    content_similarity = (
        (emotion_score * 0.50)
        + (category_score * 0.30)
        + (activity_score * 0.20)
    )

    return round(content_similarity, 4)


def calculate_content_similarity_for_all(
    detected_emotions,
    preferred_categories,
    preferred_activity_types,
):
    """
    Calculate content similarity for all wellness items.
    """

    wellness_content = load_wellness_content()

    results = []

    for _, content in wellness_content.iterrows():

        similarity = calculate_content_similarity(
            detected_emotions=detected_emotions,
            preferred_categories=preferred_categories,
            preferred_activity_types=preferred_activity_types,
            content=content,
        )

        results.append(
            {
                "content_id": content["content_id"],
                "content_similarity": similarity,
            }
        )

    return results
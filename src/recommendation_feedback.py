from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEEDBACK_PATH = (
    PROJECT_ROOT
    / "data"
    / "recommendation_feedback.csv"
)


REQUIRED_COLUMNS = {
    "user_id",
    "timestamp",
    "content_id",
    "interaction_type",
    "rating",
    "emotion_state",
    "intensity_score",
    "preference_key",
    "preference_value",
}


VALID_INTERACTIONS = {
    "viewed",
    "accepted",
    "rejected",
}


def load_feedback():
    """
    Load recommendation feedback history.
    """

    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(
            columns=sorted(REQUIRED_COLUMNS)
        )

    dataframe = pd.read_csv(
        FEEDBACK_PATH
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing feedback columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce"
    )

    dataframe["rating"] = pd.to_numeric(
        dataframe["rating"],
        errors="coerce"
    )

    dataframe["intensity_score"] = pd.to_numeric(
        dataframe["intensity_score"],
        errors="coerce"
    )

    return dataframe


def save_feedback_record(
    user_id,
    content_id,
    interaction_type,
    rating,
    emotion_state,
    intensity_score,
    preference_key="",
    preference_value="",
):
    """
    Save one user recommendation interaction.
    """

    interaction_type = (
        str(interaction_type)
        .strip()
        .lower()
    )

    if interaction_type not in VALID_INTERACTIONS:
        raise ValueError(
            "interaction_type must be one of: "
            "viewed, accepted, rejected"
        )

    if rating is None:
        rating_value = None
    else:
        rating_value = float(rating)

        if rating_value < 1 or rating_value > 5:
            raise ValueError(
                "rating must be between 1 and 5."
            )

    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "content_id": content_id,
        "interaction_type": interaction_type,
        "rating": rating_value,
        "emotion_state": emotion_state,
        "intensity_score": float(
            intensity_score
        ),
        "preference_key": preference_key,
        "preference_value": preference_value,
    }

    new_record = pd.DataFrame(
        [record]
    )

    if FEEDBACK_PATH.exists():

        existing = pd.read_csv(
            FEEDBACK_PATH
        )

        updated = pd.concat(
            [
                existing,
                new_record
            ],
            ignore_index=True
        )

    else:

        updated = new_record

    updated.to_csv(
        FEEDBACK_PATH,
        index=False
    )

    return record


def save_preference_change(
    user_id,
    preference_key,
    preference_value,
    emotion_state="",
    intensity_score=0.0,
):
    """
    Store a user preference change as feedback history.
    """

    return save_feedback_record(
        user_id=user_id,
        content_id="",
        interaction_type="viewed",
        rating=None,
        emotion_state=emotion_state,
        intensity_score=intensity_score,
        preference_key=preference_key,
        preference_value=preference_value,
    )


def get_user_feedback(user_id):
    """
    Return all feedback for one user.
    """

    feedback = load_feedback()

    if feedback.empty:
        return feedback

    return feedback[
        feedback["user_id"] == user_id
    ].copy().reset_index(drop=True)


def calculate_feedback_signal(
    user_id,
    content_id,
):
    """
    Calculate a feedback signal for one content item.

    Accepted interactions and high ratings increase
    the signal. Rejected interactions reduce it.

    Returns a value between 0 and 1.
    """

    feedback = get_user_feedback(
        user_id
    )

    if feedback.empty:
        return 0.5

    matching = feedback[
        feedback["content_id"] == content_id
    ].copy()

    if matching.empty:
        return 0.5

    interaction_scores = []

    for _, record in matching.iterrows():

        interaction = str(
            record["interaction_type"]
        ).lower()

        rating = record["rating"]

        if pd.isna(rating):
            rating_score = 0.5
        else:
            rating_score = float(rating) / 5.0

        if interaction == "accepted":

            score = (
                0.70
                + (rating_score * 0.30)
            )

        elif interaction == "viewed":

            score = (
                0.35
                + (rating_score * 0.30)
            )

        elif interaction == "rejected":

            score = (
                rating_score * 0.20
            )

        else:

            score = 0.5

        interaction_scores.append(
            score
        )

    if not interaction_scores:
        return 0.5

    return round(
        float(
            sum(interaction_scores)
            / len(interaction_scores)
        ),
        4,
    )


def get_feedback_summary(user_id):
    """
    Create a summary of the user's feedback behavior.
    """

    feedback = get_user_feedback(
        user_id
    )

    if feedback.empty:
        return {
            "total_feedback": 0,
            "viewed": 0,
            "accepted": 0,
            "rejected": 0,
            "average_rating": None,
        }

    viewed = int(
        (
            feedback["interaction_type"]
            == "viewed"
        ).sum()
    )

    accepted = int(
        (
            feedback["interaction_type"]
            == "accepted"
        ).sum()
    )

    rejected = int(
        (
            feedback["interaction_type"]
            == "rejected"
        ).sum()
    )

    ratings = feedback[
        feedback["rating"].notna()
    ]["rating"]

    average_rating = (
        float(ratings.mean())
        if not ratings.empty
        else None
    )

    return {
        "total_feedback": len(
            feedback
        ),
        "viewed": viewed,
        "accepted": accepted,
        "rejected": rejected,
        "average_rating": average_rating,
    }
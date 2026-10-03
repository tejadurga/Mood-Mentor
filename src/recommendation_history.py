import pandas as pd

from datetime import datetime
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

HISTORY_PATH = Path(
    "data/recommendation_history.csv"
)


# ==================================================
# LOAD HISTORY
# ==================================================

def load_recommendation_history():
    """
    Load stored recommendation history.

    Returns:
        pandas.DataFrame
    """

    if not HISTORY_PATH.exists():

        return pd.DataFrame(
            columns=[
                "timestamp",
                "user_id",
                "content_id",
                "title",
                "rank",
                "hybrid_score",
                "emotion_state",
                "intensity_score",
                "detected_emotions",
            ]
        )

    try:

        dataframe = pd.read_csv(
            HISTORY_PATH
        )

        return dataframe

    except Exception:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "user_id",
                "content_id",
                "title",
                "rank",
                "hybrid_score",
                "emotion_state",
                "intensity_score",
                "detected_emotions",
            ]
        )


# ==================================================
# SAVE RECOMMENDATION HISTORY
# ==================================================

def save_recommendation_history(
    user_id,
    recommendations,
    emotional_state,
    detected_emotions,
):
    """
    Save generated recommendations to history.

    Only valid recommendation rows are stored.
    """

    if recommendations is None:
        return False

    if not isinstance(
        recommendations,
        pd.DataFrame
    ):

        recommendations = pd.DataFrame(
            recommendations
        )

    if recommendations.empty:
        return False

    required_columns = [
        "content_id",
        "title",
        "rank",
        "hybrid_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in recommendations.columns
    ]

    if missing_columns:

        raise ValueError(
            "Recommendation history is missing "
            f"required columns: {missing_columns}"
        )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    detected_emotions_text = ", ".join(
        str(emotion)
        for emotion in detected_emotions
    )

    records = []

    for _, recommendation in (
        recommendations.iterrows()
    ):

        records.append(
            {
                "timestamp":
                    timestamp,

                "user_id":
                    user_id,

                "content_id":
                    recommendation[
                        "content_id"
                    ],

                "title":
                    recommendation[
                        "title"
                    ],

                "rank":
                    int(
                        recommendation[
                            "rank"
                        ]
                    ),

                "hybrid_score":
                    round(
                        float(
                            recommendation[
                                "hybrid_score"
                            ]
                        ),
                        4
                    ),

                "emotion_state":
                    emotional_state.get(
                        "emotional_state",
                        ""
                    ),

                "intensity_score":
                    round(
                        float(
                            emotional_state.get(
                                "emotion_intensity",
                                0.0
                            )
                        ),
                        4
                    ),

                "detected_emotions":
                    detected_emotions_text,
            }
        )

    history_dataframe = pd.DataFrame(
        records
    )

    existing_history = (
        load_recommendation_history()
    )

    combined_history = pd.concat(
        [
            existing_history,
            history_dataframe
        ],
        ignore_index=True
    )

    combined_history.to_csv(
        HISTORY_PATH,
        index=False
    )

    return True


# ==================================================
# USER HISTORY
# ==================================================

def get_user_recommendation_history(
    user_id
):
    """
    Return recommendation history
    for a specific user.
    """

    history = (
        load_recommendation_history()
    )

    if history.empty:
        return history

    if "user_id" not in history.columns:
        return pd.DataFrame()

    user_history = history[
        history["user_id"].astype(str)
        == str(user_id)
    ].copy()

    return user_history.sort_values(
        by="timestamp",
        ascending=False
    ).reset_index(
        drop=True
    )
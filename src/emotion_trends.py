from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMOTION_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "emotion_history.csv"
)


def load_emotion_history():
    """
    Load historical emotional-state records.
    """

    if not EMOTION_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Emotion history file not found: "
            f"{EMOTION_HISTORY_PATH}"
        )

    dataframe = pd.read_csv(
        EMOTION_HISTORY_PATH
    )

    required_columns = {
        "user_id",
        "timestamp",
        "dominant_emotion",
        "detected_emotions",
        "intensity_score",
        "intensity_level",
        "polarity",
        "polarity_score",
        "emotional_state",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing emotion history columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp",
            "user_id",
            "dominant_emotion",
            "intensity_score",
            "polarity_score",
        ]
    )

    dataframe["intensity_score"] = pd.to_numeric(
        dataframe["intensity_score"],
        errors="coerce"
    )

    dataframe["polarity_score"] = pd.to_numeric(
        dataframe["polarity_score"],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=[
            "intensity_score",
            "polarity_score",
        ]
    )

    return dataframe.sort_values(
        by="timestamp"
    ).reset_index(drop=True)


def get_user_emotion_history(user_id):
    """
    Return chronological emotion history for one user.
    """

    history = load_emotion_history()

    user_history = history[
        history["user_id"] == user_id
    ].copy()

    return user_history.sort_values(
        by="timestamp"
    ).reset_index(drop=True)


def parse_detected_emotions(value):
    """
    Convert semicolon-separated emotions into a list.
    """

    if pd.isna(value):
        return []

    return [
        emotion.strip().lower()
        for emotion in str(value).split(";")
        if emotion.strip()
    ]


def calculate_emotion_frequency(user_history):
    """
    Count how frequently each dominant emotion occurs.
    """

    if user_history.empty:
        return {}

    frequency = (
        user_history[
            "dominant_emotion"
        ]
        .str.lower()
        .value_counts()
        .to_dict()
    )

    return {
        emotion: int(count)
        for emotion, count in frequency.items()
    }


def calculate_emotion_intensity_trend(
    user_history
):
    """
    Compare earlier and recent average intensity.
    """

    if user_history.empty:
        return {
            "earlier_average": 0.0,
            "recent_average": 0.0,
            "change": 0.0,
            "direction": "No data",
        }

    intensities = (
        user_history[
            "intensity_score"
        ]
        .astype(float)
        .tolist()
    )

    if len(intensities) == 1:
        return {
            "earlier_average": intensities[0],
            "recent_average": intensities[0],
            "change": 0.0,
            "direction": "Stable",
        }

    split_index = max(
        1,
        len(intensities) // 2
    )

    earlier = intensities[
        :split_index
    ]

    recent = intensities[
        split_index:
    ]

    earlier_average = sum(
        earlier
    ) / len(earlier)

    recent_average = sum(
        recent
    ) / len(recent)

    change = (
        recent_average
        - earlier_average
    )

    if change > 0.05:
        direction = "Increasing"

    elif change < -0.05:
        direction = "Decreasing"

    else:
        direction = "Stable"

    return {
        "earlier_average": round(
            earlier_average,
            4
        ),
        "recent_average": round(
            recent_average,
            4
        ),
        "change": round(
            change,
            4
        ),
        "direction": direction,
    }


def calculate_polarity_trend(
    user_history
):
    """
    Compare earlier and recent sentiment polarity.
    """

    if user_history.empty:
        return {
            "earlier_average": 0.0,
            "recent_average": 0.0,
            "change": 0.0,
            "direction": "No data",
        }

    polarity_values = (
        user_history[
            "polarity_score"
        ]
        .astype(float)
        .tolist()
    )

    if len(polarity_values) == 1:
        return {
            "earlier_average": polarity_values[0],
            "recent_average": polarity_values[0],
            "change": 0.0,
            "direction": "Stable",
        }

    split_index = max(
        1,
        len(polarity_values) // 2
    )

    earlier = polarity_values[
        :split_index
    ]

    recent = polarity_values[
        split_index:
    ]

    earlier_average = sum(
        earlier
    ) / len(earlier)

    recent_average = sum(
        recent
    ) / len(recent)

    change = (
        recent_average
        - earlier_average
    )

    if change > 0.10:
        direction = "More Positive"

    elif change < -0.10:
        direction = "More Negative"

    else:
        direction = "Stable"

    return {
        "earlier_average": round(
            earlier_average,
            4
        ),
        "recent_average": round(
            recent_average,
            4
        ),
        "change": round(
            change,
            4
        ),
        "direction": direction,
    }


def detect_repeated_emotional_patterns(
    user_history,
    minimum_occurrences=2
):
    """
    Identify emotions that repeatedly appear
    in the user's history.
    """

    frequency = calculate_emotion_frequency(
        user_history
    )

    repeated = {
        emotion: count
        for emotion, count in frequency.items()
        if count >= minimum_occurrences
    }

    return dict(
        sorted(
            repeated.items(),
            key=lambda item: (
                -item[1],
                item[0]
            )
        )
    )


def get_recent_emotional_state(
    user_history,
    recent_count=3
):
    """
    Return the most recent emotional state and
    recent state history.
    """

    if user_history.empty:
        return None

    recent = user_history.tail(
        recent_count
    )

    latest = recent.iloc[-1]

    recent_emotions = (
        recent["dominant_emotion"]
        .str.title()
        .tolist()
    )

    return {
        "dominant_emotion": str(
            latest["dominant_emotion"]
        ),
        "intensity_score": float(
            latest["intensity_score"]
        ),
        "intensity_level": str(
            latest["intensity_level"]
        ),
        "polarity": str(
            latest["polarity"]
        ),
        "polarity_score": float(
            latest["polarity_score"]
        ),
        "emotional_state": str(
            latest["emotional_state"]
        ),
        "recent_dominant_emotions":
            recent_emotions,
        "timestamp": latest[
            "timestamp"
        ],
    }


def calculate_emotional_trend_signal(
    detected_emotions,
    user_history
):
    """
    Calculate how strongly the current emotions
    match the user's historical emotional patterns.
    """

    if not detected_emotions:
        return 0.0

    if user_history.empty:
        return 0.0

    detected_set = {
        emotion.lower()
        for emotion in detected_emotions
    }

    frequency = calculate_emotion_frequency(
        user_history
    )

    if not frequency:
        return 0.0

    max_frequency = max(
        frequency.values()
    )

    frequency_scores = []

    for emotion in detected_set:

        count = frequency.get(
            emotion,
            0
        )

        frequency_scores.append(
            count / max_frequency
        )

    frequency_signal = (
        sum(frequency_scores)
        / len(frequency_scores)
    )

    recent_history = user_history.tail(
        3
    )

    recent_emotions = set()

    for value in recent_history[
        "detected_emotions"
    ]:

        recent_emotions.update(
            parse_detected_emotions(value)
        )

    recent_overlap = (
        detected_set.intersection(
            recent_emotions
        )
    )

    recent_signal = (
        1.0
        if recent_overlap
        else 0.0
    )

    trend_signal = (
        (frequency_signal * 0.60)
        + (recent_signal * 0.40)
    )

    return round(
        float(trend_signal),
        4
    )


def analyze_emotion_trends(
    user_id,
    current_emotions=None
):
    """
    Generate a complete historical emotional profile.
    """

    user_history = get_user_emotion_history(
        user_id
    )

    if user_history.empty:

        return {
            "user_id": user_id,
            "history_records": 0,
            "emotion_frequency": {},
            "intensity_trend": None,
            "polarity_trend": None,
            "repeated_patterns": {},
            "recent_state": None,
            "trend_signal": 0.0,
        }

    trend_signal = 0.0

    if current_emotions:
        trend_signal = (
            calculate_emotional_trend_signal(
                detected_emotions=current_emotions,
                user_history=user_history
            )
        )

    return {
        "user_id": user_id,

        "history_records": len(
            user_history
        ),

        "emotion_frequency":
            calculate_emotion_frequency(
                user_history
            ),

        "intensity_trend":
            calculate_emotion_intensity_trend(
                user_history
            ),

        "polarity_trend":
            calculate_polarity_trend(
                user_history
            ),

        "repeated_patterns":
            detect_repeated_emotional_patterns(
                user_history
            ),

        "recent_state":
            get_recent_emotional_state(
                user_history
            ),

        "trend_signal":
            trend_signal,
    }
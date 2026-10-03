import pandas as pd


# ==================================================
# GENERIC HELPERS
# ==================================================

def _normalize_text(value):
    """
    Normalize text for safe filtering.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _parse_datetime_column(
    dataframe,
    column_name
):
    """
    Convert a column to datetime safely.
    """
    dataframe = dataframe.copy()

    if column_name in dataframe.columns:

        dataframe[column_name] = pd.to_datetime(
            dataframe[column_name],
            errors="coerce"
        )

    return dataframe


# ==================================================
# SEARCH
# ==================================================

def search_dataframe(
    dataframe,
    search_text="",
    search_columns=None
):
    """
    Search records across selected columns.

    Returns:
        filtered DataFrame
    """

    if dataframe is None:
        return pd.DataFrame()

    if not isinstance(
        dataframe,
        pd.DataFrame
    ):
        dataframe = pd.DataFrame(
            dataframe
        )

    if dataframe.empty:
        return dataframe.copy()

    search_text = _normalize_text(
        search_text
    )

    if not search_text:
        return dataframe.copy()

    if search_columns is None:

        search_columns = [
            column
            for column in dataframe.columns
            if dataframe[column].dtype == "object"
        ]

    available_columns = [
        column
        for column in search_columns
        if column in dataframe.columns
    ]

    if not available_columns:
        return dataframe.copy()

    mask = pd.Series(
        False,
        index=dataframe.index
    )

    for column in available_columns:

        mask = mask | (
            dataframe[column]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        )

    return dataframe[
        mask
    ].copy()


# ==================================================
# DATE FILTER
# ==================================================

def filter_by_date_range(
    dataframe,
    date_column,
    start_date=None,
    end_date=None
):
    """
    Filter records between start and end dates.
    """

    if dataframe is None:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    if date_column not in dataframe.columns:
        return dataframe

    dataframe = _parse_datetime_column(
        dataframe,
        date_column
    )

    if start_date is not None:

        start_timestamp = pd.Timestamp(
            start_date
        )

        dataframe = dataframe[
            dataframe[date_column]
            >= start_timestamp
        ]

    if end_date is not None:

        end_timestamp = (
            pd.Timestamp(end_date)
            + pd.Timedelta(days=1)
        )

        dataframe = dataframe[
            dataframe[date_column]
            < end_timestamp
        ]

    return dataframe.copy()


# ==================================================
# EMOTION FILTER
# ==================================================

def filter_by_emotion(
    dataframe,
    emotion,
    columns=None
):
    """
    Filter records containing the selected emotion.
    """

    if dataframe is None:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    emotion = _normalize_text(
        emotion
    )

    if not emotion or emotion == "all":
        return dataframe

    if columns is None:

        columns = [
            "emotion",
            "dominant_emotion",
            "primary_emotion",
            "detected_emotions",
            "emotion_state"
        ]

    available_columns = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    if not available_columns:
        return dataframe

    mask = pd.Series(
        False,
        index=dataframe.index
    )

    for column in available_columns:

        mask = mask | (
            dataframe[column]
            .astype(str)
            .str.lower()
            .str.contains(
                emotion,
                case=False,
                na=False
            )
        )

    return dataframe[
        mask
    ].copy()


# ==================================================
# INTENSITY FILTER
# ==================================================

def filter_by_intensity(
    dataframe,
    intensity_column,
    minimum_intensity=0.0,
    maximum_intensity=1.0
):
    """
    Filter records by normalized intensity score.
    """

    if dataframe is None:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    if intensity_column not in dataframe.columns:
        return dataframe

    dataframe[
        intensity_column
    ] = pd.to_numeric(
        dataframe[
            intensity_column
        ],
        errors="coerce"
    )

    return dataframe[
        dataframe[intensity_column]
        .fillna(0.0)
        .between(
            float(minimum_intensity),
            float(maximum_intensity)
        )
    ].copy()


# ==================================================
# CATEGORY / RECOMMENDATION TYPE FILTER
# ==================================================

def filter_by_recommendation_type(
    dataframe,
    selected_type,
    column_name="category"
):
    """
    Filter recommendation records by
    recommendation category/type.
    """

    if dataframe is None:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    selected_type = _normalize_text(
        selected_type
    )

    if (
        not selected_type
        or selected_type == "all"
    ):
        return dataframe

    if column_name not in dataframe.columns:
        return dataframe

    return dataframe[
        dataframe[column_name]
        .astype(str)
        .str.lower()
        == selected_type
    ].copy()


# ==================================================
# FEEDBACK STATUS FILTER
# ==================================================

def filter_by_feedback_status(
    dataframe,
    feedback_status,
    column_name="interaction_type"
):
    """
    Filter feedback/recommendation records by
    feedback interaction status.
    """

    if dataframe is None:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    feedback_status = _normalize_text(
        feedback_status
    )

    if (
        not feedback_status
        or feedback_status == "all"
    ):
        return dataframe

    if column_name not in dataframe.columns:
        return dataframe

    return dataframe[
        dataframe[column_name]
        .astype(str)
        .str.lower()
        == feedback_status
    ].copy()


# ==================================================
# WELLNESS CONTENT FILTER
# ==================================================

def filter_wellness_content(
    dataframe,
    search_text="",
    emotion="All",
    category="All",
    activity_type="All",
    minimum_duration=0,
    maximum_duration=9999
):
    """
    Search and filter wellness content.
    """

    filtered = search_dataframe(
        dataframe=dataframe,
        search_text=search_text,
        search_columns=[
            "content_id",
            "title",
            "description",
            "category",
            "activity_type",
            "target_emotions"
        ]
    )

    filtered = filter_by_emotion(
        filtered,
        emotion,
        columns=[
            "target_emotions",
            "emotion",
            "description",
            "title"
        ]
    )

    filtered = filter_by_recommendation_type(
        filtered,
        category,
        column_name="category"
    )

    if (
        activity_type
        and str(activity_type).lower()
        != "all"
        and "activity_type" in filtered.columns
    ):

        filtered = filtered[
            filtered["activity_type"]
            .astype(str)
            .str.lower()
            == str(activity_type).lower()
        ].copy()

    if "duration_minutes" in filtered.columns:

        filtered[
            "duration_minutes"
        ] = pd.to_numeric(
            filtered[
                "duration_minutes"
            ],
            errors="coerce"
        )

        filtered = filtered[
            filtered["duration_minutes"]
            .fillna(0)
            .between(
                float(minimum_duration),
                float(maximum_duration)
            )
        ].copy()

    return filtered.reset_index(
        drop=True
    )


# ==================================================
# RECOMMENDATION HISTORY FILTER
# ==================================================

def filter_recommendation_history(
    dataframe,
    search_text="",
    start_date=None,
    end_date=None,
    emotion="All",
    minimum_intensity=0.0,
    maximum_intensity=1.0,
    recommendation_type="All"
):
    """
    Search and filter recommendation history.
    """

    filtered = search_dataframe(
        dataframe=dataframe,
        search_text=search_text,
        search_columns=[
            "content_id",
            "title",
            "emotion_state",
            "detected_emotions"
        ]
    )

    if "timestamp" in filtered.columns:

        filtered = filter_by_date_range(
            filtered,
            date_column="timestamp",
            start_date=start_date,
            end_date=end_date
        )

    filtered = filter_by_emotion(
        filtered,
        emotion
    )

    filtered = filter_by_intensity(
        filtered,
        intensity_column="intensity_score",
        minimum_intensity=minimum_intensity,
        maximum_intensity=maximum_intensity
    )

    filtered = filter_by_recommendation_type(
        filtered,
        recommendation_type,
        column_name="category"
    )

    return filtered.reset_index(
        drop=True
    )


# ==================================================
# FEEDBACK FILTER
# ==================================================

def filter_feedback(
    dataframe,
    search_text="",
    start_date=None,
    end_date=None,
    emotion="All",
    feedback_status="All"
):
    """
    Search and filter recommendation feedback.
    """

    filtered = search_dataframe(
        dataframe=dataframe,
        search_text=search_text,
        search_columns=[
            "user_id",
            "content_id",
            "interaction_type",
            "emotion_state",
            "preference_key",
            "preference_value"
        ]
    )

    if "timestamp" in filtered.columns:

        filtered = filter_by_date_range(
            filtered,
            date_column="timestamp",
            start_date=start_date,
            end_date=end_date
        )

    filtered = filter_by_emotion(
        filtered,
        emotion
    )

    filtered = filter_by_feedback_status(
        filtered,
        feedback_status
    )

    return filtered.reset_index(
        drop=True
    )
import pandas as pd
import streamlit as st

from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

EMOTION_HISTORY_PATH = Path(
    "data/emotion_history.csv"
)


# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def load_emotion_history():
    """
    Load stored emotional history safely.
    """

    if not EMOTION_HISTORY_PATH.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        EMOTION_HISTORY_PATH
    )

    if dataframe.empty:
        return dataframe

    return dataframe


# ==================================================
# COLUMN NORMALIZATION
# ==================================================

def normalize_history_columns(
    dataframe
):
    """
    Normalize common history column names.
    """

    dataframe = dataframe.copy()

    rename_map = {}

    for column in dataframe.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        rename_map[column] = normalized

    dataframe = dataframe.rename(
        columns=rename_map
    )

    return dataframe


# ==================================================
# DATE DETECTION
# ==================================================

def detect_date_column(
    dataframe
):
    """
    Find the timestamp/date column.
    """

    candidates = [
        "timestamp",
        "date",
        "datetime",
        "created_at",
        "recorded_at"
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


# ==================================================
# EMOTION DETECTION
# ==================================================

def detect_emotion_column(
    dataframe
):
    """
    Find the emotion column.
    """

    candidates = [
        "emotion",
        "dominant_emotion",
        "primary_emotion"
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


# ==================================================
# INTENSITY DETECTION
# ==================================================

def detect_intensity_column(
    dataframe
):
    """
    Find the intensity column.
    """

    candidates = [
        "intensity_score",
        "intensity",
        "emotion_intensity"
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


# ==================================================
# PREPARE HISTORY
# ==================================================

def prepare_emotion_history(
    dataframe,
    user_id=None
):
    """
    Prepare emotion history for trend analysis.
    """

    dataframe = normalize_history_columns(
        dataframe
    )

    if dataframe.empty:
        return dataframe

    if (
        user_id is not None
        and "user_id" in dataframe.columns
    ):

        dataframe = dataframe[
            dataframe["user_id"].astype(str)
            == str(user_id)
        ].copy()

    date_column = detect_date_column(
        dataframe
    )

    emotion_column = detect_emotion_column(
        dataframe
    )

    intensity_column = detect_intensity_column(
        dataframe
    )

    if date_column is None:
        return pd.DataFrame()

    if emotion_column is None:
        return pd.DataFrame()

    dataframe["date"] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=["date"]
    )

    dataframe["emotion"] = (
        dataframe[emotion_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if intensity_column is not None:

        dataframe["intensity"] = pd.to_numeric(
            dataframe[intensity_column],
            errors="coerce"
        )

    else:

        dataframe["intensity"] = 0.0

    dataframe = dataframe.sort_values(
        "date"
    )

    return dataframe[
        [
            "date",
            "emotion",
            "intensity"
        ]
    ].reset_index(
        drop=True
    )


# ==================================================
# DAILY TRENDS
# ==================================================

def calculate_daily_trends(
    dataframe
):
    """
    Calculate daily emotion frequency
    and average intensity.
    """

    if dataframe.empty:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    dataframe["period"] = (
        dataframe["date"]
        .dt.date
    )

    daily = (
        dataframe
        .groupby(
            ["period", "emotion"]
        )
        .agg(
            emotion_count=(
                "emotion",
                "count"
            ),
            average_intensity=(
                "intensity",
                "mean"
            )
        )
        .reset_index()
    )

    return daily


# ==================================================
# WEEKLY TRENDS
# ==================================================

def calculate_weekly_trends(
    dataframe
):
    """
    Calculate weekly emotion frequency
    and average intensity.
    """

    if dataframe.empty:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    dataframe["period"] = (
        dataframe["date"]
        .dt.to_period("W")
        .astype(str)
    )

    weekly = (
        dataframe
        .groupby(
            ["period", "emotion"]
        )
        .agg(
            emotion_count=(
                "emotion",
                "count"
            ),
            average_intensity=(
                "intensity",
                "mean"
            )
        )
        .reset_index()
    )

    return weekly


# ==================================================
# MONTHLY TRENDS
# ==================================================

def calculate_monthly_trends(
    dataframe
):
    """
    Calculate monthly emotion frequency
    and average intensity.
    """

    if dataframe.empty:
        return pd.DataFrame()

    dataframe = dataframe.copy()

    dataframe["period"] = (
        dataframe["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        dataframe
        .groupby(
            ["period", "emotion"]
        )
        .agg(
            emotion_count=(
                "emotion",
                "count"
            ),
            average_intensity=(
                "intensity",
                "mean"
            )
        )
        .reset_index()
    )

    return monthly


# ==================================================
# CHART DATA
# ==================================================

def build_intensity_chart_data(
    dataframe,
    period_column="period"
):
    """
    Create average intensity chart data.
    """

    if dataframe.empty:
        return pd.DataFrame()

    chart_data = (
        dataframe
        .groupby(period_column)[
            "average_intensity"
        ]
        .mean()
        .reset_index()
    )

    chart_data = chart_data.set_index(
        period_column
    )

    return chart_data


# ==================================================
# TREND DASHBOARD
# ==================================================

def display_emotional_trend_dashboard(
    user_id
):
    """
    Display interactive daily, weekly,
    and monthly emotional trend visualizations.
    """

    st.divider()

    st.header(
        "📈 Emotional Trend Visualization"
    )

    st.caption(
        f"Dynamic emotional trends for {user_id}"
    )

    raw_history = load_emotion_history()

    history = prepare_emotion_history(
        raw_history,
        user_id=user_id
    )

    if history.empty:

        st.info(
            "No emotional history is available "
            "for this user."
        )

        return

    # ==================================================
    # OVERVIEW
    # ==================================================

    st.subheader(
        "📊 Trend Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Records",
            len(history)
        )

    with col2:

        st.metric(
            "First Record",
            history["date"]
            .min()
            .strftime("%Y-%m-%d")
        )

    with col3:

        st.metric(
            "Latest Record",
            history["date"]
            .max()
            .strftime("%Y-%m-%d")
        )

    # ==================================================
    # TABS
    # ==================================================

    daily_tab, weekly_tab, monthly_tab = (
        st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly"
            ]
        )
    )

    # ==================================================
    # DAILY
    # ==================================================

    with daily_tab:

        daily = calculate_daily_trends(
            history
        )

        if daily.empty:

            st.info(
                "No daily trend data available."
            )

        else:

            st.subheader(
                "Daily Emotional Trend"
            )

            daily_chart = (
                build_intensity_chart_data(
                    daily
                )
            )

            st.line_chart(
                daily_chart
            )

            st.dataframe(
                daily,
                use_container_width=True,
                hide_index=True
            )

    # ==================================================
    # WEEKLY
    # ==================================================

    with weekly_tab:

        weekly = calculate_weekly_trends(
            history
        )

        if weekly.empty:

            st.info(
                "No weekly trend data available."
            )

        else:

            st.subheader(
                "Weekly Emotional Trend"
            )

            weekly_chart = (
                build_intensity_chart_data(
                    weekly
                )
            )

            st.line_chart(
                weekly_chart
            )

            st.dataframe(
                weekly,
                use_container_width=True,
                hide_index=True
            )

    # ==================================================
    # MONTHLY
    # ==================================================

    with monthly_tab:

        monthly = calculate_monthly_trends(
            history
        )

        if monthly.empty:

            st.info(
                "No monthly trend data available."
            )

        else:

            st.subheader(
                "Monthly Emotional Trend"
            )

            monthly_chart = (
                build_intensity_chart_data(
                    monthly
                )
            )

            st.line_chart(
                monthly_chart
            )

            st.dataframe(
                monthly,
                use_container_width=True,
                hide_index=True
            )

    # ==================================================
    # USER INSIGHTS
    # ==================================================

    st.subheader(
        "💡 Emotional Insights"
    )

    most_common_emotion = (
        history["emotion"]
        .value_counts()
        .idxmax()
    )

    average_intensity = (
        history["intensity"]
        .mean()
    )

    st.write(
        f"**Most frequent emotion:** "
        f"{most_common_emotion.title()}"
    )

    st.write(
        f"**Average recorded intensity:** "
        f"{average_intensity:.2%}"
    )

    st.write(
        f"**Total emotional records:** "
        f"{len(history)}"
    )
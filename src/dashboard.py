import pandas as pd
import streamlit as st


# ==================================================
# DASHBOARD HELPERS
# ==================================================

def _format_emotion_name(value):
    """Return a clean display name for an emotion."""
    return str(value).strip().title()


def _history_to_dataframe(history):
    """
    Convert user history into a DataFrame safely.

    Supports DataFrame, list, tuple, and dictionary inputs.
    """

    if history is None:
        return pd.DataFrame()

    if isinstance(history, pd.DataFrame):
        return history.copy()

    if isinstance(history, (list, tuple)):
        return pd.DataFrame(history)

    if isinstance(history, dict):
        return pd.DataFrame(history)

    return pd.DataFrame()


# ==================================================
# MAIN DASHBOARD
# ==================================================

def display_mood_dashboard(
    sentiment_result,
    emotion_result,
    emotional_state,
    ranking_result,
    user_history,
    user_id,
):
    """
    Display the Milestone 4 advanced Mood Mentor dashboard.

    The dashboard shows:
        - sentiment
        - dominant emotion
        - emotion intensity
        - emotion confidence scores
        - detected emotions
        - top recommendations
        - recommendation scores
        - user history
    """

    st.divider()

    st.header("📊 Mood Mentor Dashboard")

    st.caption(
        f"Personalized dashboard for {user_id}"
    )

    # ==================================================
    # SUMMARY METRICS
    # ==================================================

    sentiment = sentiment_result.get(
        "sentiment",
        "Unknown"
    )

    compound = sentiment_result.get(
        "compound",
        0.0
    )

    dominant_emotion = emotional_state.get(
        "dominant_emotion",
        "Unknown"
    )

    intensity = float(
        emotional_state.get(
            "emotion_intensity",
            0.0
        )
    )

    intensity_level = emotional_state.get(
        "intensity_level",
        "Unknown"
    )

    primary_confidence = float(
        emotion_result.get(
            "primary_confidence",
            0.0
        )
    )

    detected_emotions = emotional_state.get(
        "detected_emotions",
        []
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sentiment",
            sentiment
        )

    with col2:
        st.metric(
            "Dominant Emotion",
            _format_emotion_name(
                dominant_emotion
            )
        )

    with col3:
        st.metric(
            "Emotion Intensity",
            f"{intensity:.2%}"
        )

    with col4:
        st.metric(
            "Primary Confidence",
            f"{primary_confidence:.2%}"
        )

    # ==================================================
    # EMOTION INSIGHTS
    # ==================================================

    st.subheader("🧠 Emotion Insights")

    emotion_rows = []

    for emotion in emotion_result.get(
        "emotions",
        []
    ):
        emotion_rows.append(
            {
                "Emotion": _format_emotion_name(
                    emotion["emotion"]
                ),
                "Confidence": float(
                    emotion["confidence"]
                ),
                "Detected": (
                    "Yes"
                    if emotion["detected"]
                    else "No"
                ),
            }
        )

    emotion_df = pd.DataFrame(
        emotion_rows
    )

    if not emotion_df.empty:

        chart_df = (
            emotion_df[
                [
                    "Emotion",
                    "Confidence"
                ]
            ]
            .set_index("Emotion")
        )

        st.bar_chart(
            chart_df,
            y="Confidence"
        )

        display_df = emotion_df.copy()

        display_df["Confidence"] = (
            display_df["Confidence"]
            .map(
                lambda value:
                    f"{value:.2%}"
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    detected_text = ", ".join(
        _format_emotion_name(
            emotion
        )
        for emotion in detected_emotions
    )

    if detected_text:
        st.success(
            f"Detected emotions: {detected_text}"
        )
    else:
        st.info(
            "No emotions crossed the detection threshold."
        )

    st.write(
        f"**Intensity Level:** "
        f"{intensity_level}"
    )

    st.write(
        f"**VADER Compound Score:** "
        f"{compound:.4f}"
    )

    # ==================================================
    # RECOMMENDATION SNAPSHOT
    # ==================================================

    st.subheader(
        "🌿 Recommendation Snapshot"
    )

    recommendations = (
        ranking_result.get(
            "recommendations",
            pd.DataFrame()
        )
    )

    if recommendations is None:
        recommendations = pd.DataFrame()

    if not isinstance(
        recommendations,
        pd.DataFrame
    ):
        recommendations = pd.DataFrame(
            recommendations
        )

    if recommendations.empty:

        st.info(
            "No recommendations are currently available."
        )

    else:

        top_count = min(
            5,
            len(recommendations)
        )

        recommendation_rows = []

        for _, row in (
            recommendations
            .head(top_count)
            .iterrows()
        ):

            recommendation_rows.append(
                {
                    "Rank": int(
                        row["rank"]
                    ),
                    "Recommendation":
                        row["title"],
                    "Category":
                        row["category"],
                    "Activity":
                        row["activity_type"],
                    "Duration":
                        f"{int(row['duration_minutes'])} min",
                    "Score":
                        f"{float(row['hybrid_score']):.2%}",
                }
            )

        recommendation_df = pd.DataFrame(
            recommendation_rows
        )

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

        top_recommendation = (
            recommendations.iloc[0]
        )

        st.success(
            "Top Recommendation: "
            f"{top_recommendation['title']} "
            f"({float(top_recommendation['hybrid_score']):.2%})"
        )

    # ==================================================
    # USER HISTORY
    # ==================================================

    st.subheader(
        "🕘 User Emotional History"
    )

    history_df = _history_to_dataframe(
        user_history
    )

    if history_df.empty:

        st.info(
            "No user history records are available."
        )

    else:

        st.write(
            f"Historical records: "
            f"**{len(history_df)}**"
        )

        # Show the complete available history
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
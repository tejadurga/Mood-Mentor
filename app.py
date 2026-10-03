import streamlit as st
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.emotion_config import EMOTION_LABELS
from src.emotion_state import analyze_emotional_state

from src.ingestion import ingest_direct_text
from src.preprocessing import preprocess_texts
from src.sentiment import analyze_sentiment

from src.report import (
    generate_sentiment_report,
    generate_summary,
    compare_expected_results
)

from src.recommendation_data import (
    get_user_profile,
    get_user_history
)

from src.recommendation_engine import (
    build_hybrid_recommendations
)

from src.recommendation_ranking import (
    rank_recommendations,
    get_top_recommendation
)

from src.recommendation_explainability import (
    build_explainable_recommendations
)

from src.emotion_trends import (
    analyze_emotion_trends
)

from src.recommendation_feedback import (
    save_feedback_record,
    get_feedback_summary,
    get_user_feedback
)
from src.dashboard import (
    display_mood_dashboard
)
from src.trend_dashboard import (
    display_emotional_trend_dashboard
)
from src.recommendation_history import (
    save_recommendation_history,
    get_user_recommendation_history
)
from src.search_filters import (
    search_dataframe,
    filter_by_date_range,
    filter_by_emotion,
    filter_by_intensity,
    filter_by_recommendation_type,
    filter_by_feedback_status,
    filter_wellness_content,
    filter_recommendation_history,
    filter_feedback
)
from src.wellness_report import (
    build_report_tables,
    build_csv_report,
    build_pdf_report
)
# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Mood Mentor",
    page_icon="🧠",
    layout="wide"
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

BERT_MODEL_PATH = "models/bert_emotion"

BERT_THRESHOLD = 0.50

DEFAULT_USER_ID = "demo_user_001"


@st.cache_resource
def load_bert_model():

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# ==================================================
# HEADER
# ==================================================

st.title("🧠 Mood Mentor")

st.subheader(
    "AI-Based Emotion Analysis and Personalized Wellness Recommendation"
)

st.write(
    "Mood Mentor combines VADER sentiment analysis, "
    "BERT-based multi-label emotion classification, "
    "emotional-state analysis, historical emotion trends, "
    "semantic matching, hybrid recommendation, "
    "ranking, explainability, and feedback learning."
)


# ==================================================
# SENTIMENT DISPLAY
# ==================================================

def display_sentiment(result):

    if not result["success"]:

        st.error(
            result["message"]
        )

        return

    sentiment = result["sentiment"]

    if sentiment == "Positive":

        st.success(
            f"😊 {sentiment}"
        )

    elif sentiment == "Negative":

        st.error(
            f"😞 {sentiment}"
        )

    else:

        st.info(
            f"😐 {sentiment}"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Positive",
            f"{result['positive']:.4f}"
        )

    with col2:

        st.metric(
            "Negative",
            f"{result['negative']:.4f}"
        )

    with col3:

        st.metric(
            "Neutral",
            f"{result['neutral']:.4f}"
        )

    with col4:

        st.metric(
            "Compound",
            f"{result['compound']:.4f}"
        )


# ==================================================
# BERT EMOTION ANALYSIS
# ==================================================

def analyze_emotions(text):

    tokenizer, model = load_bert_model()

    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    probabilities = torch.sigmoid(
        outputs.logits
    )[0]

    emotion_results = []

    for emotion, probability in zip(
        EMOTION_LABELS,
        probabilities
    ):

        confidence = probability.item()

        emotion_results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "detected":
                    confidence >= BERT_THRESHOLD
            }
        )

    detected_emotions = [
        result
        for result in emotion_results
        if result["detected"]
    ]

    primary_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    primary_emotion = EMOTION_LABELS[
        primary_index
    ]

    primary_confidence = probabilities[
        primary_index
    ].item()

    return {
        "success": True,
        "emotions": emotion_results,
        "detected_emotions": detected_emotions,
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_confidence,
        "probabilities": probabilities.cpu().numpy()
    }


# ==================================================
# BERT EMOTION DISPLAY
# ==================================================

def display_emotions(result):

    if not result["success"]:

        st.error(
            "Emotion analysis failed."
        )

        return

    st.subheader(
        "🧠 BERT Emotion Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Primary Emotion",
            result["primary_emotion"].title()
        )

    with col2:

        st.metric(
            "Primary Confidence",
            f"{result['primary_confidence']:.2%}"
        )

    st.write(
        f"Prediction threshold: "
        f"**{BERT_THRESHOLD:.2f}**"
    )

    st.write(
        "**Detected Emotions:**"
    )

    if result["detected_emotions"]:

        detected_text = ", ".join(
            emotion["emotion"].title()
            for emotion in result["detected_emotions"]
        )

        st.success(
            detected_text
        )

    else:

        st.info(
            "No emotion crossed the prediction threshold."
        )

    st.write(
        "**Emotion Confidence Scores:**"
    )

    emotion_data = []

    for emotion_result in result["emotions"]:

        emotion_data.append(
            {
                "Emotion":
                    emotion_result["emotion"].title(),

                "Confidence":
                    emotion_result["confidence"],

                "Detected":
                    "Yes"
                    if emotion_result["detected"]
                    else "No"
            }
        )

    emotion_df = pd.DataFrame(
        emotion_data
    )

    emotion_df["Confidence"] = (
        emotion_df["Confidence"]
        .map(
            lambda value: f"{value:.2%}"
        )
    )

    st.dataframe(
        emotion_df,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# EMOTIONAL STATE DISPLAY
# ==================================================

def display_emotional_state(state):

    st.subheader(
        "💭 Emotional State Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dominant Emotion",
            state["dominant_emotion"].title()
        )

    with col2:

        st.metric(
            "Emotion Intensity",
            f"{state['emotion_intensity']:.2%}"
        )

    with col3:

        st.metric(
            "Severity",
            state["emotion_severity"]
        )

    st.write(
        "**Detected Emotions:**"
    )

    if state["detected_emotions"]:

        detected_text = ", ".join(
            emotion.title()
            for emotion in state[
                "detected_emotions"
            ]
        )

        st.success(
            detected_text
        )

    st.write(
        f"**Intensity Level:** "
        f"{state['intensity_level']}"
    )

    st.write(
        f"**Sentiment Polarity:** "
        f"{state['polarity']}"
    )

    if state["polarity_score"] is not None:

        st.write(
            f"**Polarity Score:** "
            f"{state['polarity_score']:.4f}"
        )

    st.write(
        f"**Mixed Emotional State:** "
        f"{'Yes' if state['is_mixed'] else 'No'}"
    )

    st.info(
        f"**Final Emotional State:** "
        f"{state['emotional_state']}"
    )


# ==================================================
# EMOTIONAL TREND DISPLAY
# ==================================================

def display_emotional_trends(
    trend_analysis
):

    st.subheader(
        "📈 Emotional Trend & User State"
    )

    if trend_analysis[
        "history_records"
    ] == 0:

        st.info(
            "No historical emotion records "
            "are available for this user."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Historical Records",
            trend_analysis[
                "history_records"
            ]
        )

    with col2:

        trend = trend_analysis[
            "intensity_trend"
        ]

        st.metric(
            "Intensity Trend",
            trend["direction"]
        )

    with col3:

        polarity = trend_analysis[
            "polarity_trend"
        ]

        st.metric(
            "Polarity Trend",
            polarity["direction"]
        )

    st.write(
        "**Emotion Frequency:**"
    )

    frequency = trend_analysis[
        "emotion_frequency"
    ]

    if frequency:

        frequency_data = []

        for emotion, count in frequency.items():

            frequency_data.append(
                {
                    "Emotion":
                        emotion.title(),

                    "Occurrences":
                        count
                }
            )

        frequency_df = pd.DataFrame(
            frequency_data
        )

        st.dataframe(
            frequency_df,
            use_container_width=True,
            hide_index=True
        )

    st.write(
        "**Repeated Emotional Patterns:**"
    )

    repeated_patterns = (
        trend_analysis[
            "repeated_patterns"
        ]
    )

    if repeated_patterns:

        pattern_text = ", ".join(
            f"{emotion.title()} "
            f"({count} times)"
            for emotion, count
            in repeated_patterns.items()
        )

        st.info(
            pattern_text
        )

    else:

        st.info(
            "No repeated emotional pattern detected."
        )

    recent_state = trend_analysis[
        "recent_state"
    ]

    if recent_state:

        st.write(
            "**Recent Emotional State:**"
        )

        recent_col1, recent_col2, recent_col3 = (
            st.columns(3)
        )

        with recent_col1:

            st.write(
                f"**Emotion:** "
                f"{recent_state['dominant_emotion'].title()}"
            )

        with recent_col2:

            st.write(
                f"**Intensity:** "
                f"{recent_state['intensity_score']:.2%}"
            )

        with recent_col3:

            st.write(
                f"**Polarity:** "
                f"{recent_state['polarity']}"
            )

    st.write(
        f"**Historical Pattern Signal:** "
        f"{trend_analysis['trend_signal']:.4f}"
    )


# ==================================================
# RECOMMENDATION DISPLAY
# ==================================================

def display_recommendations(
    ranking_result,
    user_id,
    user_text,
    emotional_state
):

    st.subheader(
        "🌿 Personalized Wellness Recommendations"
    )

    recommendations = ranking_result[
        "recommendations"
    ]

    if recommendations.empty:

        st.warning(
            "No recommendation matched the current criteria."
        )

        return

    # ------------------------------------------------
    # TOP RECOMMENDATION
    # ------------------------------------------------

    top_recommendation = (
        get_top_recommendation(
            ranking_result
        )
    )

    st.success(
        "Top recommendation identified dynamically."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Top Recommendation",
            top_recommendation["title"]
        )

    with col2:

        st.metric(
            "Recommendation Score",
            f"{top_recommendation['hybrid_score']:.2%}"
        )

    with col3:

        st.metric(
            "Rank",
            int(
                top_recommendation["rank"]
            )
        )

    st.write(
        top_recommendation["description"]
    )

    # ------------------------------------------------
    # WHY THIS WAS RECOMMENDED
    # ------------------------------------------------

    st.subheader(
        "💡 Why was this recommended?"
    )

    for reason in top_recommendation[
        "explanation_reasons"
    ]:

        st.write(
            f"• {reason}."
        )

    st.divider()

    # ------------------------------------------------
    # RANKED TABLE
    # ------------------------------------------------

    st.subheader(
        "🏆 Ranked Recommendations"
    )

    recommendation_rows = []

    for _, recommendation in (
        recommendations.iterrows()
    ):

        recommendation_rows.append(
            {
                "Rank":
                    int(
                        recommendation["rank"]
                    ),

                "Recommendation":
                    recommendation["title"],

                "Category":
                    recommendation["category"],

                "Activity":
                    recommendation["activity_type"],

                "Duration":
                    f"{int(recommendation['duration_minutes'])} min",

                "Score":
                    f"{recommendation['hybrid_score']:.2%}",

                "Semantic":
                    f"{recommendation['semantic_relevance']:.2%}",

                "Emotion Match":
                    f"{recommendation['emotion_match']:.2%}",

                "Feedback":
                    f"{recommendation['feedback_signal']:.2%}"
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

    # ------------------------------------------------
    # INDIVIDUAL DETAILS + FEEDBACK
    # ------------------------------------------------

    st.subheader(
        "🔎 Recommendation Details & Feedback"
    )

    for _, recommendation in (
        recommendations.iterrows()
    ):

        content_id = recommendation[
            "content_id"
        ]

        title = recommendation[
            "title"
        ]

        with st.expander(
            f"Rank {int(recommendation['rank'])} — {title}"
        ):

            st.write(
                recommendation[
                    "description"
                ]
            )

            st.write(
                "**Why selected:**"
            )

            for reason in recommendation[
                "explanation_reasons"
            ]:

                st.write(
                    f"• {reason}."
                )

            st.divider()

            detail_col1, detail_col2 = (
                st.columns(2)
            )

            with detail_col1:

                st.write(
                    f"**Category:** "
                    f"{recommendation['category']}"
                )

                st.write(
                    f"**Activity:** "
                    f"{recommendation['activity_type']}"
                )

                st.write(
                    f"**Duration:** "
                    f"{int(recommendation['duration_minutes'])} minutes"
                )

                st.write(
                    f"**Target Emotions:** "
                    f"{', '.join(
                        emotion.title()
                        for emotion in recommendation[
                            'target_emotions'
                        ]
                    )}"
                )

            with detail_col2:

                st.write(
                    f"**Hybrid Score:** "
                    f"{recommendation['hybrid_score']:.4f}"
                )

                st.write(
                    f"**Emotion Match:** "
                    f"{recommendation['emotion_match']:.4f}"
                )

                st.write(
                    f"**Semantic Relevance:** "
                    f"{recommendation['semantic_relevance']:.4f}"
                )

                st.write(
                    f"**Historical Trend:** "
                    f"{recommendation['historical_trend_signal']:.4f}"
                )

                st.write(
                    f"**Feedback Signal:** "
                    f"{recommendation['feedback_signal']:.4f}"
                )

            st.divider()

            # ----------------------------------------
            # FEEDBACK
            # ----------------------------------------

            st.write(
                "**Recommendation Feedback**"
            )

            form_key = (
                f"feedback_{user_id}_{content_id}"
            )

            with st.form(
                key=form_key
            ):

                interaction_type = st.radio(
                    "How did you interact with this recommendation?",
                    options=[
                        "viewed",
                        "accepted",
                        "rejected"
                    ],
                    horizontal=True
                )

                rating = st.slider(
                    "Rate this recommendation",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1
                )

                submit_feedback = st.form_submit_button(
                    "Save Feedback"
                )

                if submit_feedback:

                    save_feedback_record(
                        user_id=user_id,
                        content_id=content_id,
                        interaction_type=interaction_type,
                        rating=rating,
                        emotion_state=(
                            emotional_state[
                                "emotional_state"
                            ]
                        ),
                        intensity_score=float(
                            emotional_state[
                                "emotion_intensity"
                            ]
                        )
                    )

                    st.success(
                        "Feedback saved successfully. "
                        "Future recommendation scoring can "
                        "use this interaction."
                    )

    st.caption(
        f"Recommendation profile: {user_id}"
    )


# ==================================================
# FEEDBACK SUMMARY
# ==================================================

def display_feedback_summary(
    user_id
):

    st.subheader(
        "📝 Recommendation Feedback History"
    )

    summary = get_feedback_summary(
        user_id
    )

    if summary[
        "total_feedback"
    ] == 0:

        st.info(
            "No feedback has been recorded yet."
        )

        return

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Total",
            summary["total_feedback"]
        )

    with col2:

        st.metric(
            "Viewed",
            summary["viewed"]
        )

    with col3:

        st.metric(
            "Accepted",
            summary["accepted"]
        )

    with col4:

        st.metric(
            "Rejected",
            summary["rejected"]
        )

    with col5:

        average_rating = (
            summary["average_rating"]
        )

        if average_rating is None:

            display_rating = "N/A"

        else:

            display_rating = (
                f"{average_rating:.2f}"
            )

        st.metric(
            "Avg Rating",
            display_rating
        )

    with st.expander(
        "View Stored Feedback"
    ):

        feedback = get_user_feedback(
            user_id
        )

        if not feedback.empty:

            display_columns = [
                "timestamp",
                "content_id",
                "interaction_type",
                "rating",
                "emotion_state",
                "intensity_score"
            ]

            available_columns = [
                column
                for column in display_columns
                if column in feedback.columns
            ]

            st.dataframe(
                feedback[
                    available_columns
                ],
                use_container_width=True,
                hide_index=True
            )


# ==================================================
# MILESTONE 1 — INITIAL SENTIMENT REPORT
# ==================================================

st.header(
    "📊 Initial Sentiment Report"
)

st.write(
    "Run the sample corpus through ingestion, "
    "preprocessing and VADER sentiment analysis."
)

if st.button(
    "Generate Initial Sentiment Report",
    type="secondary"
):

    try:

        corpus = pd.read_csv(
            "data/sample_corpus.csv"
        )

    except FileNotFoundError:

        st.error(
            "Sample corpus not found. "
            "Please create data/sample_corpus.csv."
        )

        st.stop()

    if "text" not in corpus.columns:

        st.error(
            "Sample corpus must contain a 'text' column."
        )

        st.stop()

    if "expected_sentiment" not in corpus.columns:

        st.error(
            "Sample corpus must contain an "
            "'expected_sentiment' column."
        )

        st.stop()

    input_texts = (
        corpus["text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    expected_labels = (
        corpus["expected_sentiment"]
        .dropna()
        .astype(str)
        .tolist()
    )

    preprocessing_results = preprocess_texts(
        input_texts
    )

    if not preprocessing_results:

        st.error(
            "No valid text was available "
            "after preprocessing."
        )

        st.stop()

    sentiment_results = []

    for data in preprocessing_results:

        sentiment = analyze_sentiment(
            data["processed_text"]
        )

        if sentiment["success"]:

            sentiment_results.append(
                {
                    "original_text":
                        data["original_text"],

                    "processed_text":
                        data["processed_text"],

                    "sentiment":
                        sentiment["sentiment"],

                    "positive":
                        sentiment["positive"],

                    "negative":
                        sentiment["negative"],

                    "neutral":
                        sentiment["neutral"],

                    "compound":
                        sentiment["compound"]
                }
            )

    report_df = generate_sentiment_report(
        sentiment_results
    )

    summary = generate_summary(
        report_df
    )

    comparison_df, comparison = (
        compare_expected_results(
            report_df,
            expected_labels
        )
    )

    st.success(
        "✓ Initial sentiment report generated successfully."
    )

    st.subheader(
        "📈 Analysis Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Analyzed Samples",
            summary["total_samples"]
        )

    with col2:

        st.metric(
            "Positive",
            summary["positive"]
        )

    with col3:

        st.metric(
            "Negative",
            summary["negative"]
        )

    with col4:

        st.metric(
            "Neutral",
            summary["neutral"]
        )

    st.subheader(
        "📋 Detailed Sentiment Report"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader(
        "✅ Expected vs Generated Results"
    )

    st.write(
        f"Correct predictions: "
        f"**{comparison['correct']} / "
        f"{comparison['total']}**"
    )

    st.write(
        f"Validation accuracy: "
        f"**{comparison['accuracy'] * 100:.2f}%**"
    )

    csv_data = comparison_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Sentiment Report",
        data=csv_data,
        file_name="sentiment_report.csv",
        mime="text/csv"
    )


# ==================================================
# MILESTONE 3 — PERSONALIZED WELLNESS ANALYSIS
# ==================================================

st.divider()

st.header(
    "🌿 Personalized Wellness Analysis"
)

st.write(
    "Enter text to analyze sentiment, detect emotions, "
    "track historical emotional patterns, and generate "
    "personalized wellness recommendations."
)


# ==================================================
# USER PROFILE
# ==================================================

st.subheader(
    "👤 User Profile"
)

user_id = st.selectbox(
    "Select a demo user profile",
    options=[
        "demo_user_001",
        "demo_user_002"
    ],
    index=0
)

try:

    selected_profile = get_user_profile(
        user_id
    )

except Exception as error:

    st.error(
        f"Unable to load user profile: {error}"
    )

    st.stop()


with st.expander(
    "View User Preferences"
):

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Preferred Categories:**"
        )

        st.write(
            ", ".join(
                selected_profile[
                    "preferred_categories"
                ]
            )
        )

        st.write(
            "**Preferred Activities:**"
        )

        st.write(
            ", ".join(
                selected_profile[
                    "preferred_activity_types"
                ]
            )
        )

    with col2:

        st.write(
            "**Maximum Duration:** "
            f"{selected_profile['max_duration_minutes']} minutes"
        )

        st.write(
            "**Preferred Intensity Levels:**"
        )

        st.write(
            ", ".join(
                selected_profile[
                    "preferred_intensity_levels"
                ]
            )
        )


# ==================================================
# USER TEXT
# ==================================================

user_text = st.text_area(
    "Enter how you are feeling:",
    placeholder=(
        "Example: I feel scared and overwhelmed "
        "about tomorrow."
    ),
    height=140
)


# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button(
    "Analyze Mood & Get Recommendations",
    type="primary"
):

    if not user_text.strip():

        st.warning(
            "Please enter some text before analysis."
        )

        st.stop()

    # ------------------------------------------------
    # INGESTION
    # ------------------------------------------------

    ingestion_result = ingest_direct_text(
        user_text
    )

    if not ingestion_result["success"]:

        st.error(
            ingestion_result["message"]
        )

        st.stop()

    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    preprocessing_results = preprocess_texts(
        ingestion_result["texts"]
    )

    if not preprocessing_results:

        st.error(
            "No valid text was available "
            "after preprocessing."
        )

        st.stop()

    data = preprocessing_results[0]

    original_text = data[
        "original_text"
    ]

    processed_text = data[
        "processed_text"
    ]

    # ------------------------------------------------
    # TEXT PROCESSING
    # ------------------------------------------------

    with st.expander(
        "View Text Processing"
    ):

        st.write(
            "**Original Text:**"
        )

        st.write(
            original_text
        )

        st.write(
            "**Processed Text for VADER:**"
        )

        st.write(
            processed_text
        )

    # ------------------------------------------------
    # VADER
    # ------------------------------------------------

    st.divider()

    st.subheader(
        "📊 VADER Sentiment Analysis"
    )

    sentiment_result = analyze_sentiment(
        processed_text
    )

    display_sentiment(
        sentiment_result
    )

    # ------------------------------------------------
    # BERT
    # ------------------------------------------------

    st.divider()

    try:

        emotion_result = analyze_emotions(
            original_text
        )

        display_emotions(
            emotion_result
        )

    except Exception as error:

        st.error(
            f"BERT emotion analysis failed: {error}"
        )

        st.stop()

    # ------------------------------------------------
    # EMOTIONAL STATE
    # ------------------------------------------------

    try:

        emotional_state = (
            analyze_emotional_state(
                probabilities=emotion_result[
                    "probabilities"
                ],
                sentiment_result=sentiment_result,
                threshold=BERT_THRESHOLD
            )
        )

        st.divider()

        display_emotional_state(
            emotional_state
        )

    except Exception as error:

        st.error(
            f"Emotional state analysis failed: {error}"
        )

        st.stop()

    # ------------------------------------------------
    # EMOTIONAL TREND
    # ------------------------------------------------

    try:

        trend_analysis = (
            analyze_emotion_trends(
                user_id=user_id,
                current_emotions=(
                    emotional_state[
                        "detected_emotions"
                    ]
                )
            )
        )

        st.divider()

        display_emotional_trends(
            trend_analysis
        )
        # ------------------------------------------------
        # MILESTONE 4 — EMOTIONAL TREND VISUALIZATION
        # ------------------------------------------------

        try:

            display_emotional_trend_dashboard(
                user_id=user_id
            )
        

        except Exception as error:

            st.error(
                f"Advanced emotional trend visualization "
                f"failed: {error}"
            )

    except Exception as error:

        st.error(
            f"Emotional trend analysis failed: {error}"
        )

        st.stop()

    # ------------------------------------------------
    # RECOMMENDATION ENGINE
    # ------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 Recommendation Engine"
    )

    try:

        detected_emotion_names = (
            emotional_state[
                "detected_emotions"
            ]
        )

        hybrid_recommendations = (
            build_hybrid_recommendations(
                user_id=user_id,
                user_text=original_text,
                detected_emotions=(
                    detected_emotion_names
                ),
                emotion_intensity=(
                    emotional_state[
                        "emotion_intensity"
                    ]
                ),
                emotional_state=(
                    emotional_state[
                        "emotional_state"
                    ]
                )
            )
        )

        if hybrid_recommendations.empty:

            st.warning(
                "No recommendation candidates were generated."
            )

            st.stop()

        # ------------------------------------------------
        # FINAL RANKING
        # ------------------------------------------------

        ranking_result = rank_recommendations(
            recommendations=(
                hybrid_recommendations
            ),
            top_k=5,
            minimum_score=0.30
        )

        # ------------------------------------------------
        # EXPLAINABILITY
        # ------------------------------------------------

        explainable_recommendations = (
            build_explainable_recommendations(
                recommendations=ranking_result[
                    "recommendations"
                ],
                detected_emotions=(
                    emotional_state[
                        "detected_emotions"
                    ]
                ),
                emotion_intensity=(
                    emotional_state[
                        "emotion_intensity"
                    ]
                ),
                preferred_categories=(
                    selected_profile[
                        "preferred_categories"
                    ]
                ),
                preferred_activity_types=(
                    selected_profile[
                        "preferred_activity_types"
                    ]
                ),
                historical_trend_signal=float(
                    hybrid_recommendations[
                        "historical_trend_signal"
                    ].iloc[0]
                )
            )
        )

        ranking_result[
            "recommendations"
        ] = explainable_recommendations

        # ------------------------------------------------
        # MILESTONE 4 — SAVE RECOMMENDATION HISTORY
        # ------------------------------------------------

        try:

            save_recommendation_history(
                user_id=user_id,
                recommendations=ranking_result[
                    "recommendations"
                ],
                emotional_state=emotional_state,
                detected_emotions=(
                    emotional_state[
                        "detected_emotions"
                    ]
                )
            )

        except Exception as error:

            st.warning(
                f"Recommendation history could not "
                f"be saved: {error}"
            )

        # ------------------------------------------------
        # DISPLAY RECOMMENDATIONS
        # ------------------------------------------------

        display_recommendations(
            ranking_result=ranking_result,
            user_id=user_id,
            user_text=original_text,
            emotional_state=emotional_state
        )
            # ------------------------------------------------
        # MILESTONE 4 — TASK 5
        # REPORT GENERATION & EXPORT
        # ------------------------------------------------

        st.divider()

        st.header(
            "📄 Report Generation & Export"
        )

        st.write(
            "Generate a complete report containing "
            "the current emotional analysis, trends, "
            "and personalized recommendations."
        )

        try:

            report_sections = build_report_tables(
                user_id=user_id,
                user_text=original_text,
                sentiment_result=sentiment_result,
                emotion_result=emotion_result,
                emotional_state=emotional_state,
                trend_analysis=trend_analysis,
                ranking_result=ranking_result
            )

            csv_report = build_csv_report(
                report_sections
            )

            pdf_report = build_pdf_report(
                report_sections
            )

            report_col1, report_col2 = (
                st.columns(2)
            )

            with report_col1:

                st.download_button(
                    label="⬇️ Download Complete CSV Report",
                    data=csv_report,
                    file_name=(
                        f"mood_mentor_report_"
                        f"{user_id}.csv"
                    ),
                    mime="text/csv",
                    key="complete_csv_report"
                )

            with report_col2:

                st.download_button(
                    label="⬇️ Download Complete PDF Report",
                    data=pdf_report,
                    file_name=(
                        f"mood_mentor_report_"
                        f"{user_id}.pdf"
                    ),
                    mime="application/pdf",
                    key="complete_pdf_report"
                )

            st.success(
                "Complete report generated successfully."
            )

        except Exception as error:

            st.error(
                f"Report generation failed: {error}"
            )
        # ------------------------------------------------
        # MILESTONE 4 — ADVANCED MOOD DASHBOARD
        # ------------------------------------------------

        try:

            dashboard_history = get_user_history(
                user_id
            )

            display_mood_dashboard(
                sentiment_result=sentiment_result,
                emotion_result=emotion_result,
                emotional_state=emotional_state,
                ranking_result=ranking_result,
                user_history=dashboard_history,
                user_id=user_id
            )

        except Exception as error:

            st.error(
                f"Dashboard rendering failed: {error}"
            )

        # ------------------------------------------------
        # VALIDATION SUMMARY
        # ------------------------------------------------

        with st.expander(
            "🔎 Recommendation System Validation"
        ):

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Candidates",
                    ranking_result[
                        "total_candidates"
                    ]
                )

            with col2:

                st.metric(
                    "Ranked",
                    ranking_result[
                        "total_ranked"
                    ]
                )

            with col3:

                st.metric(
                    "Duplicates Removed",
                    ranking_result[
                        "duplicates_removed"
                    ]
                )

            with col4:

                st.metric(
                    "Low-Relevance Removed",
                    ranking_result[
                        "low_relevance_removed"
                    ]
                )

            st.write(
                "The recommendation order is calculated "
                "dynamically from current emotion, "
                "emotion intensity, preferences, "
                "historical patterns, previous interactions, "
                "semantic relevance, feedback, and "
                "content similarity."
            )

    except Exception as error:

        st.error(
            f"Recommendation engine failed: {error}"
        )

           
    

        # ------------------------------------------------
        # VALIDATION SUMMARY
        # ------------------------------------------------

        with st.expander(
            "🔎 Recommendation System Validation"
        ):

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Candidates",
                    ranking_result[
                        "total_candidates"
                    ]
                )

            with col2:

                st.metric(
                    "Ranked",
                    ranking_result[
                        "total_ranked"
                    ]
                )

            with col3:

                st.metric(
                    "Duplicates Removed",
                    ranking_result[
                        "duplicates_removed"
                    ]
                )

            with col4:

                st.metric(
                    "Low-Relevance Removed",
                    ranking_result[
                        "low_relevance_removed"
                    ]
                )

            st.write(
                "The recommendation order is calculated "
                "dynamically from current emotion, "
                "emotion intensity, preferences, "
                "historical patterns, previous interactions, "
                "semantic relevance, feedback, and "
                "content similarity."
            )

    except Exception as error:

        st.error(
            f"Recommendation engine failed: {error}"
        )

# ==================================================
# STORED FEEDBACK
# ==================================================

st.divider()

display_feedback_summary(
    user_id
)

# ==================================================
# MILESTONE 4 — RECOMMENDATION HISTORY
# ==================================================

st.divider()

st.subheader(
    "🕘 Recommendation History"
)

try:

    recommendation_history = (
        get_user_recommendation_history(
            user_id
        )
    )

    if recommendation_history.empty:

        st.info(
            "No previous recommendations "
            "have been recorded yet."
        )

    else:

        st.write(
            f"Previous recommendation records: "
            f"**{len(recommendation_history)}**"
        )

        history_columns = [
            "timestamp",
            "content_id",
            "title",
            "rank",
            "hybrid_score",
            "emotion_state",
            "intensity_score",
            "detected_emotions",
        ]

        available_columns = [
            column
            for column in history_columns
            if column in recommendation_history.columns
        ]

        history_display = (
            recommendation_history[
                available_columns
            ].copy()
        )

        if "hybrid_score" in history_display.columns:

            history_display[
                "hybrid_score"
            ] = history_display[
                "hybrid_score"
            ].map(
                lambda value:
                    f"{float(value):.2%}"
            )

        if "intensity_score" in history_display.columns:

            history_display[
                "intensity_score"
            ] = history_display[
                "intensity_score"
            ].map(
                lambda value:
                    f"{float(value):.2%}"
            )

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )

except Exception as error:

    st.error(
        f"Recommendation history could not "
        f"be displayed: {error}"
    )
# ==================================================
# MILESTONE 4 — TASK 4
# ADVANCED SEARCH & FILTERING
# ==================================================

st.divider()

st.header(
    "🔎 Advanced Search & Filtering"
)

st.write(
    "Search and filter emotional records, "
    "recommendation history, wellness content, "
    "and feedback records."
)


# ==================================================
# SEARCH TABS
# ==================================================

emotion_tab, recommendation_tab, wellness_tab, feedback_tab = (
    st.tabs(
        [
            "🧠 Emotional Records",
            "🌿 Recommendations",
            "🧘 Wellness Content",
            "📝 Feedback"
        ]
    )
)


# ==================================================
# EMOTIONAL RECORDS
# ==================================================

with emotion_tab:

    st.subheader(
        "🧠 Search Emotional Records"
    )

    try:

        emotion_history = pd.read_csv(
            "data/emotion_history.csv"
        )

        if "user_id" in emotion_history.columns:

            emotion_history = emotion_history[
                emotion_history["user_id"].astype(str)
                == str(user_id)
            ].copy()

        if emotion_history.empty:

            st.info(
                "No emotional records are available "
                "for this user."
            )

        else:

            search_text = st.text_input(
                "Search emotional records",
                placeholder=(
                    "Search emotion, state, sentiment..."
                ),
                key="emotion_search"
            )

            filtered_emotions = search_dataframe(
                emotion_history,
                search_text=search_text,
                search_columns=[
                    "emotion",
                    "dominant_emotion",
                    "primary_emotion",
                    "detected_emotions",
                    "polarity",
                    "emotion_state"
                ]
            )

            # ------------------------------------------
            # DATE FILTER
            # ------------------------------------------

            if "timestamp" in filtered_emotions.columns:

                filtered_emotions[
                    "timestamp"
                ] = pd.to_datetime(
                    filtered_emotions[
                        "timestamp"
                    ],
                    errors="coerce"
                )

                valid_dates = (
                    filtered_emotions[
                        "timestamp"
                    ]
                    .dropna()
                )

                if not valid_dates.empty:

                    min_date = (
                        valid_dates
                        .min()
                        .date()
                    )

                    max_date = (
                        valid_dates
                        .max()
                        .date()
                    )

                    date_col1, date_col2 = (
                        st.columns(2)
                    )

                    with date_col1:

                        start_date = st.date_input(
                            "From",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="emotion_start_date"
                        )

                    with date_col2:

                        end_date = st.date_input(
                            "To",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="emotion_end_date"
                        )

                    filtered_emotions = (
                        filter_by_date_range(
                            filtered_emotions,
                            date_column="timestamp",
                            start_date=start_date,
                            end_date=end_date
                        )
                    )

            # ------------------------------------------
            # EMOTION FILTER
            # ------------------------------------------

            emotion_options = [
                "All"
            ]

            possible_emotion_columns = [
                "emotion",
                "dominant_emotion",
                "primary_emotion",
                "detected_emotions"
            ]

            emotion_values = set()

            for column in possible_emotion_columns:

                if column in filtered_emotions.columns:

                    values = (
                        filtered_emotions[
                            column
                        ]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

                    for value in values:

                        for emotion in (
                            value
                            .replace("|", ",")
                            .replace(";", ",")
                            .split(",")
                        ):

                            emotion = (
                                emotion.strip()
                            )

                            if emotion:
                                emotion_values.add(
                                    emotion.title()
                                )

            emotion_options.extend(
                sorted(emotion_values)
            )

            selected_emotion = st.selectbox(
                "Emotion",
                emotion_options,
                key="emotion_filter"
            )

            filtered_emotions = filter_by_emotion(
                filtered_emotions,
                selected_emotion
            )

            # ------------------------------------------
            # INTENSITY FILTER
            # ------------------------------------------

            intensity_column = next(
                (
                    column
                    for column in [
                        "intensity_score",
                        "intensity",
                        "emotion_intensity"
                    ]
                    if column in filtered_emotions.columns
                ),
                None
            )

            if intensity_column:

                filtered_emotions[
                    intensity_column
                ] = pd.to_numeric(
                    filtered_emotions[
                        intensity_column
                    ],
                    errors="coerce"
                )

                intensity_range = (
                    st.slider(
                        "Intensity Range",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.0, 1.0),
                        step=0.05,
                        key="emotion_intensity_filter"
                    )
                )

                filtered_emotions = (
                    filter_by_intensity(
                        filtered_emotions,
                        intensity_column=intensity_column,
                        minimum_intensity=intensity_range[0],
                        maximum_intensity=intensity_range[1]
                    )
                )

            # ------------------------------------------
            # RESULT
            # ------------------------------------------

            st.metric(
                "Matching Records",
                len(filtered_emotions)
            )

            st.dataframe(
                filtered_emotions,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            f"Emotional record filtering failed: {error}"
        )


# ==================================================
# RECOMMENDATION HISTORY
# ==================================================

with recommendation_tab:

    st.subheader(
        "🌿 Search Recommendation History"
    )

    try:

        recommendation_history = (
            get_user_recommendation_history(
                user_id
            )
        )

        if recommendation_history.empty:

            st.info(
                "No recommendation history is available "
                "for this user."
            )

        else:

            # ------------------------------------------
            # ENRICH HISTORY WITH CONTENT METADATA
            # ------------------------------------------

            try:

                wellness_content = pd.read_csv(
                    "data/wellness_content.csv"
                )

                metadata_columns = [
                    column
                    for column in [
                        "content_id",
                        "category",
                        "activity_type",
                        "duration_minutes"
                    ]
                    if column in wellness_content.columns
                ]

                if (
                    "content_id"
                    in metadata_columns
                ):

                    metadata = wellness_content[
                        metadata_columns
                    ].drop_duplicates(
                        subset=["content_id"]
                    )

                    recommendation_history = (
                        recommendation_history
                        .merge(
                            metadata,
                            on="content_id",
                            how="left",
                            suffixes=(
                                "",
                                "_content"
                            )
                        )
                    )

            except Exception:

                pass

            # ------------------------------------------
            # SEARCH
            # ------------------------------------------

            search_text = st.text_input(
                "Search recommendations",
                placeholder=(
                    "Search title, content ID, "
                    "emotion state..."
                ),
                key="recommendation_search"
            )

            filtered_recommendations = (
                search_dataframe(
                    recommendation_history,
                    search_text=search_text,
                    search_columns=[
                        "content_id",
                        "title",
                        "emotion_state",
                        "detected_emotions",
                        "category",
                        "activity_type"
                    ]
                )
            )

            # ------------------------------------------
            # DATE FILTER
            # ------------------------------------------

            if "timestamp" in filtered_recommendations.columns:

                filtered_recommendations[
                    "timestamp"
                ] = pd.to_datetime(
                    filtered_recommendations[
                        "timestamp"
                    ],
                    errors="coerce"
                )

                valid_dates = (
                    filtered_recommendations[
                        "timestamp"
                    ]
                    .dropna()
                )

                if not valid_dates.empty:

                    min_date = (
                        valid_dates
                        .min()
                        .date()
                    )

                    max_date = (
                        valid_dates
                        .max()
                        .date()
                    )

                    date_col1, date_col2 = (
                        st.columns(2)
                    )

                    with date_col1:

                        start_date = st.date_input(
                            "From",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="recommendation_start_date"
                        )

                    with date_col2:

                        end_date = st.date_input(
                            "To",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="recommendation_end_date"
                        )

                    filtered_recommendations = (
                        filter_by_date_range(
                            filtered_recommendations,
                            date_column="timestamp",
                            start_date=start_date,
                            end_date=end_date
                        )
                    )

            # ------------------------------------------
            # EMOTION FILTER
            # ------------------------------------------

            recommendation_emotions = [
                "All",
                "Joy",
                "Sadness",
                "Anger",
                "Fear",
                "Surprise",
                "Disgust"
            ]

            selected_recommendation_emotion = (
                st.selectbox(
                    "Emotion",
                    recommendation_emotions,
                    key="recommendation_emotion_filter"
                )
            )

            filtered_recommendations = (
                filter_by_emotion(
                    filtered_recommendations,
                    selected_recommendation_emotion
                )
            )

            # ------------------------------------------
            # INTENSITY FILTER
            # ------------------------------------------

            if "intensity_score" in (
                filtered_recommendations.columns
            ):

                filtered_recommendations[
                    "intensity_score"
                ] = pd.to_numeric(
                    filtered_recommendations[
                        "intensity_score"
                    ],
                    errors="coerce"
                )

                intensity_range = (
                    st.slider(
                        "Intensity Range",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.0, 1.0),
                        step=0.05,
                        key="recommendation_intensity_filter"
                    )
                )

                filtered_recommendations = (
                    filter_by_intensity(
                        filtered_recommendations,
                        intensity_column="intensity_score",
                        minimum_intensity=intensity_range[0],
                        maximum_intensity=intensity_range[1]
                    )
                )

            # ------------------------------------------
            # RECOMMENDATION TYPE
            # ------------------------------------------

            category_options = [
                "All"
            ]

            if "category" in (
                filtered_recommendations.columns
            ):

                categories = (
                    filtered_recommendations[
                        "category"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                category_options.extend(
                    sorted(categories)
                )

            selected_category = st.selectbox(
                "Recommendation Type / Category",
                category_options,
                key="recommendation_type_filter"
            )

            filtered_recommendations = (
                filter_by_recommendation_type(
                    filtered_recommendations,
                    selected_category,
                    column_name="category"
                )
            )

            st.metric(
                "Matching Recommendations",
                len(filtered_recommendations)
            )

            st.dataframe(
                filtered_recommendations,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            f"Recommendation filtering failed: {error}"
        )


# ==================================================
# WELLNESS CONTENT
# ==================================================

with wellness_tab:

    st.subheader(
        "🧘 Search Wellness Content"
    )

    try:

        wellness_content = pd.read_csv(
            "data/wellness_content.csv"
        )

        if wellness_content.empty:

            st.info(
                "No wellness content is available."
            )

        else:

            search_text = st.text_input(
                "Search wellness content",
                placeholder=(
                    "Search title, description, "
                    "activity..."
                ),
                key="wellness_search"
            )

            # ------------------------------------------
            # EMOTION OPTIONS
            # ------------------------------------------

            wellness_emotions = [
                "All",
                "Joy",
                "Sadness",
                "Anger",
                "Fear",
                "Surprise",
                "Disgust"
            ]

            selected_wellness_emotion = (
                st.selectbox(
                    "Target Emotion",
                    wellness_emotions,
                    key="wellness_emotion_filter"
                )
            )

            # ------------------------------------------
            # CATEGORY
            # ------------------------------------------

            category_options = [
                "All"
            ]

            if "category" in wellness_content.columns:

                category_values = (
                    wellness_content[
                        "category"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                category_options.extend(
                    sorted(category_values)
                )

            selected_wellness_category = (
                st.selectbox(
                    "Category",
                    category_options,
                    key="wellness_category_filter"
                )
            )

            # ------------------------------------------
            # ACTIVITY TYPE
            # ------------------------------------------

            activity_options = [
                "All"
            ]

            if (
                "activity_type"
                in wellness_content.columns
            ):

                activity_values = (
                    wellness_content[
                        "activity_type"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                activity_options.extend(
                    sorted(activity_values)
                )

            selected_activity = st.selectbox(
                "Activity Type",
                activity_options,
                key="wellness_activity_filter"
            )

            # ------------------------------------------
            # DURATION
            # ------------------------------------------

            minimum_duration = 0
            maximum_duration = 9999

            if (
                "duration_minutes"
                in wellness_content.columns
            ):

                duration_values = pd.to_numeric(
                    wellness_content[
                        "duration_minutes"
                    ],
                    errors="coerce"
                ).dropna()

                if not duration_values.empty:

                    minimum_duration = int(
                        duration_values.min()
                    )

                    maximum_duration = int(
                        duration_values.max()
                    )

            duration_range = st.slider(
                "Duration (minutes)",
                min_value=minimum_duration,
                max_value=maximum_duration,
                value=(
                    minimum_duration,
                    maximum_duration
                ),
                step=1,
                key="wellness_duration_filter"
            )

            filtered_wellness = (
                filter_wellness_content(
                    dataframe=wellness_content,
                    search_text=search_text,
                    emotion=selected_wellness_emotion,
                    category=selected_wellness_category,
                    activity_type=selected_activity,
                    minimum_duration=duration_range[0],
                    maximum_duration=duration_range[1]
                )
            )

            st.metric(
                "Matching Wellness Items",
                len(filtered_wellness)
            )

            st.dataframe(
                filtered_wellness,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            f"Wellness content filtering failed: {error}"
        )


# ==================================================
# FEEDBACK
# ==================================================

with feedback_tab:

    st.subheader(
        "📝 Search Feedback Records"
    )

    try:

        feedback_data = get_user_feedback(
            user_id
        )

        if feedback_data.empty:

            st.info(
                "No feedback records are available "
                "for this user."
            )

        else:

            search_text = st.text_input(
                "Search feedback",
                placeholder=(
                    "Search content, interaction, "
                    "emotion..."
                ),
                key="feedback_search"
            )

            filtered_feedback = filter_feedback(
                dataframe=feedback_data,
                search_text=search_text
            )

            # ------------------------------------------
            # DATE FILTER
            # ------------------------------------------

            if "timestamp" in filtered_feedback.columns:

                filtered_feedback[
                    "timestamp"
                ] = pd.to_datetime(
                    filtered_feedback[
                        "timestamp"
                    ],
                    errors="coerce"
                )

                valid_dates = (
                    filtered_feedback[
                        "timestamp"
                    ]
                    .dropna()
                )

                if not valid_dates.empty:

                    min_date = (
                        valid_dates
                        .min()
                        .date()
                    )

                    max_date = (
                        valid_dates
                        .max()
                        .date()
                    )

                    date_col1, date_col2 = (
                        st.columns(2)
                    )

                    with date_col1:

                        start_date = st.date_input(
                            "From",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="feedback_start_date"
                        )

                    with date_col2:

                        end_date = st.date_input(
                            "To",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="feedback_end_date"
                        )

                    filtered_feedback = (
                        filter_by_date_range(
                            filtered_feedback,
                            date_column="timestamp",
                            start_date=start_date,
                            end_date=end_date
                        )
                    )

            # ------------------------------------------
            # EMOTION FILTER
            # ------------------------------------------

            feedback_emotion = st.selectbox(
                "Emotion",
                [
                    "All",
                    "Joy",
                    "Sadness",
                    "Anger",
                    "Fear",
                    "Surprise",
                    "Disgust"
                ],
                key="feedback_emotion_filter"
            )

            filtered_feedback = filter_by_emotion(
                filtered_feedback,
                feedback_emotion
            )

            # ------------------------------------------
            # FEEDBACK STATUS
            # ------------------------------------------

            feedback_status = st.selectbox(
                "Feedback Status",
                [
                    "All",
                    "viewed",
                    "accepted",
                    "rejected"
                ],
                key="feedback_status_filter"
            )

            filtered_feedback = (
                filter_by_feedback_status(
                    filtered_feedback,
                    feedback_status
                )
            )

            st.metric(
                "Matching Feedback Records",
                len(filtered_feedback)
            )

            st.dataframe(
                filtered_feedback,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            f"Feedback filtering failed: {error}"
        )
# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Mood Mentor | Infosys Springboard Virtual Internship"
)
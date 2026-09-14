import streamlit as st
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.emotion_config import EMOTION_LABELS

from src.ingestion import ingest_file
from src.preprocessing import preprocess_texts
from src.sentiment import analyze_sentiment

from src.report import (
    generate_sentiment_report,
    generate_summary,
    compare_expected_results
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Mood Mentor",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# BERT MODEL CONFIGURATION
# --------------------------------------------------

BERT_MODEL_PATH = "models/bert_emotion"

BERT_THRESHOLD = 0.50


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


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧠 Mood Mentor")

st.subheader(
    "Milestone 1 & 2 — Sentiment and Emotion Analysis"
)

st.write(
    "Mood Mentor combines VADER sentiment analysis "
    "with BERT-based multi-label emotion classification."
)


# --------------------------------------------------
# SENTIMENT DISPLAY
# --------------------------------------------------

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


# --------------------------------------------------
# BERT EMOTION ANALYSIS
# --------------------------------------------------

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
        torch.argmax(probabilities).item()
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
        "primary_confidence": primary_confidence
    }


# --------------------------------------------------
# BERT EMOTION DISPLAY
# --------------------------------------------------

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
# TASK 4 — INITIAL SENTIMENT REPORT
# ==================================================

st.header(
    "📊 Initial Sentiment Report"
)

st.write(
    "Run the complete sample text corpus through "
    "ingestion, preprocessing and VADER sentiment analysis."
)


if st.button(
    "Generate Initial Sentiment Report",
    type="primary"
):

    # ------------------------------------------------
    # LOAD SAMPLE CORPUS
    # ------------------------------------------------

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

    # ------------------------------------------------
    # VALIDATE CORPUS
    # ------------------------------------------------

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

    # ------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------

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

    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    preprocessing_results = preprocess_texts(
        input_texts
    )

    if not preprocessing_results:

        st.error(
            "No valid text was available "
            "after preprocessing."
        )

        st.stop()

    # ------------------------------------------------
    # VADER SENTIMENT ANALYSIS
    # ------------------------------------------------

    sentiment_results = []

    for data in preprocessing_results:

        sentiment = analyze_sentiment(
            data["processed_text"]
        )

        if sentiment["success"]:

            combined_result = {

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

            sentiment_results.append(
                combined_result
            )

    # ------------------------------------------------
    # GENERATE REPORT
    # ------------------------------------------------

    report_df = generate_sentiment_report(
        sentiment_results
    )

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------

    summary = generate_summary(
        report_df
    )

    # ------------------------------------------------
    # COMPARE EXPECTED RESULTS
    # ------------------------------------------------

    comparison_df, comparison = (
        compare_expected_results(
            report_df,
            expected_labels
        )
    )

    # ------------------------------------------------
    # DISPLAY SUMMARY
    # ------------------------------------------------

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

    # ------------------------------------------------
    # COMPLETE REPORT
    # ------------------------------------------------

    st.subheader(
        "📋 Detailed Sentiment Report"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    # ------------------------------------------------
    # COMPARISON
    # ------------------------------------------------

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

    # ------------------------------------------------
    # DOWNLOAD REPORT
    # ------------------------------------------------

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
# MILESTONE 1 + 2 — INDIVIDUAL TEXT ANALYSIS
# ==================================================

st.divider()

st.header(
    "🔬 Individual Text Analysis"
)

st.write(
    "Enter text to analyze both sentiment "
    "and emotions."
)

user_text = st.text_area(
    "Enter a sentence:",
    placeholder=(
        "Example: I am feeling very happy today."
    )
)


if st.button(
    "Analyze Individual Text"
):

    from src.ingestion import ingest_direct_text

    # ------------------------------------------------
    # TEXT INGESTION
    # ------------------------------------------------

    ingestion_result = ingest_direct_text(
        user_text
    )

    if not ingestion_result["success"]:

        st.error(
            ingestion_result["message"]
        )

    else:

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

        else:

            data = preprocessing_results[0]

            # ------------------------------------------------
            # ORIGINAL TEXT
            # ------------------------------------------------

            st.write(
                "**Original Text:**"
            )

            st.write(
                data["original_text"]
            )

            # ------------------------------------------------
            # PROCESSED TEXT
            # ------------------------------------------------

            st.write(
                "**Processed Text:**"
            )

            st.write(
                data["processed_text"]
            )

            # ------------------------------------------------
            # VADER SENTIMENT
            # ------------------------------------------------

            st.subheader(
                "📊 VADER Sentiment Analysis"
            )

            sentiment_result = analyze_sentiment(
                data["processed_text"]
            )

            display_sentiment(
                sentiment_result
            )

            # ------------------------------------------------
            # BERT EMOTION ANALYSIS
            # ------------------------------------------------

            try:

                emotion_result = analyze_emotions(
                    data["original_text"]
                )

                display_emotions(
                    emotion_result
                )

            except Exception as error:

                st.error(
                    f"BERT emotion analysis failed: {error}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Mood Mentor | Infosys Springboard Virtual Internship"
)
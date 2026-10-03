import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ==================================================
# REPORT DATA PREPARATION
# ==================================================

def build_report_tables(
    user_id,
    user_text,
    sentiment_result,
    emotion_result,
    emotional_state,
    trend_analysis,
    ranking_result,
):
    """
    Build report sections from the current dynamic
    Mood Mentor analysis.

    No report values are hardcoded.
    """

    report_sections = {}

    # ==================================================
    # BASIC INFORMATION
    # ==================================================

    report_sections["basic"] = {
        "user_id": str(user_id),
        "generated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "input_text": str(user_text),
    }

    # ==================================================
    # SENTIMENT
    # ==================================================

    report_sections["sentiment"] = {
        "Sentiment": sentiment_result.get(
            "sentiment",
            "Unknown"
        ),
        "Positive Score": float(
            sentiment_result.get(
                "positive",
                0.0
            )
        ),
        "Negative Score": float(
            sentiment_result.get(
                "negative",
                0.0
            )
        ),
        "Neutral Score": float(
            sentiment_result.get(
                "neutral",
                0.0
            )
        ),
        "Compound Score": float(
            sentiment_result.get(
                "compound",
                0.0
            )
        ),
    }

    # ==================================================
    # EMOTIONS
    # ==================================================

    emotion_rows = []

    for emotion in emotion_result.get(
        "emotions",
        []
    ):
        emotion_rows.append(
            {
                "Emotion":
                    str(
                        emotion["emotion"]
                    ).title(),

                "Confidence":
                    float(
                        emotion["confidence"]
                    ),

                "Detected":
                    (
                        "Yes"
                        if emotion["detected"]
                        else "No"
                    ),
            }
        )

    report_sections["emotions"] = pd.DataFrame(
        emotion_rows
    )

    # ==================================================
    # EMOTIONAL STATE
    # ==================================================

    report_sections["state"] = {
        "Dominant Emotion":
            str(
                emotional_state.get(
                    "dominant_emotion",
                    "Unknown"
                )
            ).title(),

        "Emotion Intensity":
            float(
                emotional_state.get(
                    "emotion_intensity",
                    0.0
                )
            ),

        "Intensity Level":
            str(
                emotional_state.get(
                    "intensity_level",
                    "Unknown"
                )
            ),

        "Severity":
            str(
                emotional_state.get(
                    "emotion_severity",
                    "Unknown"
                )
            ),

        "Polarity":
            str(
                emotional_state.get(
                    "polarity",
                    "Unknown"
                )
            ),

        "Polarity Score":
            emotional_state.get(
                "polarity_score"
            ),

        "Mixed Emotional State":
            (
                "Yes"
                if emotional_state.get(
                    "is_mixed",
                    False
                )
                else "No"
            ),

        "Final Emotional State":
            str(
                emotional_state.get(
                    "emotional_state",
                    "Unknown"
                )
            ),
    }

    # ==================================================
    # TREND SUMMARY
    # ==================================================

    intensity_trend = trend_analysis.get(
        "intensity_trend",
        {}
    )

    polarity_trend = trend_analysis.get(
        "polarity_trend",
        {}
    )

    report_sections["trend"] = {
        "Historical Records":
            trend_analysis.get(
                "history_records",
                0
            ),

        "Intensity Trend":
            intensity_trend.get(
                "direction",
                "Unknown"
            ),

        "Polarity Trend":
            polarity_trend.get(
                "direction",
                "Unknown"
            ),

        "Historical Pattern Signal":
            float(
                trend_analysis.get(
                    "trend_signal",
                    0.0
                )
            ),
    }

    # ==================================================
    # RECOMMENDATIONS
    # ==================================================

    recommendations = ranking_result.get(
        "recommendations",
        pd.DataFrame()
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

        recommendation_df = pd.DataFrame(
            columns=[
                "Rank",
                "Content ID",
                "Recommendation",
                "Category",
                "Activity",
                "Duration",
                "Hybrid Score",
            ]
        )

    else:

        recommendation_rows = []

        for _, recommendation in (
            recommendations.iterrows()
        ):

            recommendation_rows.append(
                {
                    "Rank":
                        int(
                            recommendation[
                                "rank"
                            ]
                        ),

                    "Content ID":
                        recommendation[
                            "content_id"
                        ],

                    "Recommendation":
                        recommendation[
                            "title"
                        ],

                    "Category":
                        recommendation[
                            "category"
                        ],

                    "Activity":
                        recommendation[
                            "activity_type"
                        ],

                    "Duration":
                        (
                            f"{int(recommendation['duration_minutes'])} min"
                        ),

                    "Hybrid Score":
                        float(
                            recommendation[
                                "hybrid_score"
                            ]
                        ),
                }
            )

        recommendation_df = pd.DataFrame(
            recommendation_rows
        )

    report_sections["recommendations"] = (
        recommendation_df
    )

    return report_sections


# ==================================================
# CSV EXPORT
# ==================================================

def build_csv_report(
    report_sections
):
    """
    Build a complete CSV report containing
    all dynamic analysis sections.
    """

    output = io.StringIO()

    basic = report_sections["basic"]

    output.write(
        "MOOD MENTOR WELLNESS REPORT\n"
    )

    output.write(
        f"User ID,{basic['user_id']}\n"
    )

    output.write(
        f"Generated At,{basic['generated_at']}\n"
    )

    output.write(
        f"Input Text,\"{basic['input_text'].replace(chr(34), chr(34) * 2)}\"\n"
    )

    output.write("\n")
    output.write("SENTIMENT ANALYSIS\n")

    sentiment_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": value,
            }
            for key, value
            in report_sections[
                "sentiment"
            ].items()
        ]
    )

    output.write(
        sentiment_df.to_csv(
            index=False
        )
    )

    output.write("\n")
    output.write("EMOTION ANALYSIS\n")

    emotions_df = report_sections[
        "emotions"
    ].copy()

    if not emotions_df.empty:

        emotions_df[
            "Confidence"
        ] = emotions_df[
            "Confidence"
        ].map(
            lambda value:
                f"{value:.4f}"
        )

        output.write(
            emotions_df.to_csv(
                index=False
            )
        )

    output.write("\n")
    output.write("EMOTIONAL STATE\n")

    state_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": value,
            }
            for key, value
            in report_sections[
                "state"
            ].items()
        ]
    )

    output.write(
        state_df.to_csv(
            index=False
        )
    )

    output.write("\n")
    output.write("EMOTIONAL TRENDS\n")

    trend_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": value,
            }
            for key, value
            in report_sections[
                "trend"
            ].items()
        ]
    )

    output.write(
        trend_df.to_csv(
            index=False
        )
    )

    output.write("\n")
    output.write("RECOMMENDATIONS\n")

    recommendation_df = (
        report_sections[
            "recommendations"
        ].copy()
    )

    if not recommendation_df.empty:

        recommendation_df[
            "Hybrid Score"
        ] = recommendation_df[
            "Hybrid Score"
        ].map(
            lambda value:
                f"{value:.4f}"
        )

        output.write(
            recommendation_df.to_csv(
                index=False
            )
        )

    return output.getvalue()


# ==================================================
# PDF HELPERS
# ==================================================

def _pdf_table(
    dataframe,
    percentage_columns=None
):
    """
    Convert a DataFrame to a ReportLab table.
    """

    if dataframe is None:
        dataframe = pd.DataFrame()

    if dataframe.empty:

        return Table(
            [["No data available"]]
        )

    dataframe = dataframe.copy()

    if percentage_columns:

        for column in percentage_columns:

            if column in dataframe.columns:

                dataframe[
                    column
                ] = dataframe[
                    column
                ].map(
                    lambda value:
                        f"{float(value):.2%}"
                )

    data = [
        list(dataframe.columns)
    ]

    for row in dataframe.itertuples(
        index=False,
        name=None
    ):

        data.append(
            [
                str(value)
                for value in row
            ]
        )

    table = Table(
        data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#0C173A"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F5F7FA"
                        ),
                    ],
                ),
            ]
        )
    )

    return table


# ==================================================
# PDF EXPORT
# ==================================================

def build_pdf_report(
    report_sections
):
    """
    Generate a complete PDF wellness report.
    """

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MoodMentorTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=20,
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        "MoodMentorHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "MoodMentorBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=5,
    )

    story = []

    # ==================================================
    # TITLE
    # ==================================================

    story.append(
        Paragraph(
            "Mood Mentor Wellness Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-based emotion analysis and "
            "personalized wellness recommendation",
            body_style
        )
    )

    basic = report_sections[
        "basic"
    ]

    story.append(
        Paragraph(
            f"<b>User ID:</b> {basic['user_id']}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{basic['generated_at']}",
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    # ==================================================
    # INPUT
    # ==================================================

    story.append(
        Paragraph(
            "1. User Input",
            heading_style
        )
    )

    input_text = (
        basic["input_text"]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            input_text,
            body_style
        )
    )

    # ==================================================
    # SENTIMENT
    # ==================================================

    story.append(
        Paragraph(
            "2. Sentiment Analysis",
            heading_style
        )
    )

    sentiment_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": (
                    f"{value:.4f}"
                    if isinstance(
                        value,
                        float
                    )
                    else str(value)
                ),
            }
            for key, value
            in report_sections[
                "sentiment"
            ].items()
        ]
    )

    story.append(
        _pdf_table(
            sentiment_df
        )
    )

    # ==================================================
    # EMOTIONS
    # ==================================================

    story.append(
        Paragraph(
            "3. Emotion Analysis",
            heading_style
        )
    )

    emotions_df = (
        report_sections[
            "emotions"
        ].copy()
    )

    story.append(
        _pdf_table(
            emotions_df,
            percentage_columns=[
                "Confidence"
            ]
        )
    )

    # ==================================================
    # EMOTIONAL STATE
    # ==================================================

    story.append(
        Paragraph(
            "4. Emotional State",
            heading_style
        )
    )

    state_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": (
                    f"{value:.4f}"
                    if isinstance(
                        value,
                        float
                    )
                    else (
                        "N/A"
                        if value is None
                        else str(value)
                    )
                ),
            }
            for key, value
            in report_sections[
                "state"
            ].items()
        ]
    )

    story.append(
        _pdf_table(
            state_df
        )
    )

    # ==================================================
    # TRENDS
    # ==================================================

    story.append(
        Paragraph(
            "5. Emotional Trends",
            heading_style
        )
    )

    trend_df = pd.DataFrame(
        [
            {
                "Metric": key,
                "Value": (
                    f"{value:.4f}"
                    if isinstance(
                        value,
                        float
                    )
                    else str(value)
                ),
            }
            for key, value
            in report_sections[
                "trend"
            ].items()
        ]
    )

    story.append(
        _pdf_table(
            trend_df
        )
    )

    # ==================================================
    # RECOMMENDATIONS
    # ==================================================

    story.append(
        Paragraph(
            "6. Personalized Recommendations",
            heading_style
        )
    )

    recommendation_df = (
        report_sections[
            "recommendations"
        ].copy()
    )

    story.append(
        _pdf_table(
            recommendation_df,
            percentage_columns=[
                "Hybrid Score"
            ]
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    story.append(
        Paragraph(
            "This report contains dynamically generated "
            "analysis from the current Mood Mentor session.",
            body_style
        )
    )

    story.append(
        Paragraph(
            "Mood Mentor is an educational wellness-support "
            "prototype and does not provide clinical diagnosis "
            "or medical treatment.",
            body_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()
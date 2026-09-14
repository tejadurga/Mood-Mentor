import pandas as pd


def generate_sentiment_report(results):
    """
    Convert sentiment analysis results into a DataFrame.

    Each result should contain:
    - original_text
    - processed_text
    - sentiment scores
    - sentiment classification
    """

    report_data = []

    for result in results:

        report_data.append({
            "Input Text": result["original_text"],
            "Processed Text": result["processed_text"],
            "Sentiment": result["sentiment"],
            "Positive Score": result["positive"],
            "Negative Score": result["negative"],
            "Neutral Score": result["neutral"],
            "Compound Score": result["compound"]
        })

    return pd.DataFrame(report_data)


def generate_summary(report_df):
    """
    Generate summary statistics from the sentiment report.
    """

    total_samples = len(report_df)

    positive_count = (
        report_df["Sentiment"]
        .eq("Positive")
        .sum()
    )

    negative_count = (
        report_df["Sentiment"]
        .eq("Negative")
        .sum()
    )

    neutral_count = (
        report_df["Sentiment"]
        .eq("Neutral")
        .sum()
    )

    return {
        "total_samples": total_samples,
        "positive": int(positive_count),
        "negative": int(negative_count),
        "neutral": int(neutral_count)
    }


def compare_expected_results(
    report_df,
    expected_labels
):
    """
    Compare VADER predictions against expected
    sentiment labels.
    """

    report_df = report_df.copy()

    report_df["Expected Sentiment"] = expected_labels

    report_df["Result"] = (
        report_df["Sentiment"]
        == report_df["Expected Sentiment"]
    )

    correct_predictions = int(
        report_df["Result"].sum()
    )

    total_predictions = len(report_df)

    if total_predictions > 0:

        accuracy = (
            correct_predictions
            / total_predictions
        )

    else:

        accuracy = 0.0

    return report_df, {
        "correct": correct_predictions,
        "total": total_predictions,
        "accuracy": accuracy
    }
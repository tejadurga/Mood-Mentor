import nltk

from nltk.sentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# DOWNLOAD VADER LEXICON
# --------------------------------------------------

nltk.download(
    "vader_lexicon",
    quiet=True
)


# --------------------------------------------------
# INITIALIZE VADER
# --------------------------------------------------

VADER = SentimentIntensityAnalyzer()


# --------------------------------------------------
# SENTIMENT LABEL
# --------------------------------------------------

def get_sentiment_label(compound_score):
    """
    Convert the VADER compound score into
    Positive, Neutral or Negative.

    VADER standard thresholds:
        compound >= 0.05  -> Positive
        compound <= -0.05 -> Negative
        otherwise         -> Neutral
    """

    if compound_score >= 0.05:
        return "Positive"

    elif compound_score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


# --------------------------------------------------
# ANALYZE SENTIMENT
# --------------------------------------------------

def analyze_sentiment(text):
    """
    Analyze the sentiment of valid text using VADER.

    Returns:
        Dictionary containing:
        - sentiment label
        - positive score
        - negative score
        - neutral score
        - compound score
    """

    # Validate input
    if text is None:

        return {
            "success": False,
            "message": "Input is empty.",
            "sentiment": None,
            "positive": None,
            "negative": None,
            "neutral": None,
            "compound": None
        }

    if not isinstance(text, str):

        return {
            "success": False,
            "message": "Input must be text.",
            "sentiment": None,
            "positive": None,
            "negative": None,
            "neutral": None,
            "compound": None
        }

    if not text.strip():

        return {
            "success": False,
            "message": "Input contains no valid text.",
            "sentiment": None,
            "positive": None,
            "negative": None,
            "neutral": None,
            "compound": None
        }

    # ----------------------------------------------
    # VADER ANALYSIS
    # ----------------------------------------------

    scores = VADER.polarity_scores(text)

    positive_score = scores["pos"]
    negative_score = scores["neg"]
    neutral_score = scores["neu"]
    compound_score = scores["compound"]

    # ----------------------------------------------
    # DETERMINE SENTIMENT
    # ----------------------------------------------

    sentiment = get_sentiment_label(
        compound_score
    )

    return {
        "success": True,
        "message": "Sentiment successfully analyzed.",
        "sentiment": sentiment,
        "positive": positive_score,
        "negative": negative_score,
        "neutral": neutral_score,
        "compound": compound_score
    }
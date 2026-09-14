import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --------------------------------------------------
# NLTK RESOURCES
# --------------------------------------------------

# Make sure required resources are available.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# --------------------------------------------------
# INITIALIZE NLP TOOLS
# --------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


# Important words that should NOT be removed
# because they can change sentiment.
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor"
}


# Keep normal stopwords except important negation words.
STOP_WORDS = STOP_WORDS - NEGATION_WORDS


# --------------------------------------------------
# 1. BASIC VALIDATION
# --------------------------------------------------

def validate_preprocessing_input(text):
    """
    Validate text before preprocessing.
    """

    if text is None:
        return False, "Input is empty."

    if not isinstance(text, str):
        return False, "Input must be text."

    if not text.strip():
        return False, "Input contains no valid text."

    return True, "Valid input."


# --------------------------------------------------
# 2. NOISE FILTERING
# --------------------------------------------------

def remove_noise(text):
    """
    Remove unwanted noise such as:
    - URLs
    - Email addresses
    - HTML tags
    - User mentions
    - Hashtags
    """

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove # symbol but keep the hashtag word
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    return text


# --------------------------------------------------
# 3. SPECIAL CHARACTER HANDLING
# --------------------------------------------------

def handle_special_characters(text):
    """
    Remove unwanted special characters while keeping
    letters, numbers and normal spaces.
    """

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    return text


# --------------------------------------------------
# 4. PUNCTUATION HANDLING
# --------------------------------------------------

def remove_punctuation(text):
    """
    Remove punctuation marks.
    """

    return text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


# --------------------------------------------------
# 5. REPEATED SPACE HANDLING
# --------------------------------------------------

def normalize_spaces(text):
    """
    Replace multiple spaces, tabs and newlines
    with a single space.
    """

    return " ".join(text.split())


# --------------------------------------------------
# 6. TOKENIZATION
# --------------------------------------------------

def tokenize_text(text):
    """
    Convert text into individual tokens.
    """

    return word_tokenize(text)


# --------------------------------------------------
# 7. STOP-WORD REMOVAL
# --------------------------------------------------

def remove_stopwords(tokens):
    """
    Remove common English stopwords while
    preserving important negation words.
    """

    return [
        token
        for token in tokens
        if token.lower() not in STOP_WORDS
    ]


# --------------------------------------------------
# 8. LEMMATIZATION
# --------------------------------------------------

def lemmatize_tokens(tokens):
    """
    Convert words to their base/dictionary form.
    """

    return [
        LEMMATIZER.lemmatize(token)
        for token in tokens
    ]


# --------------------------------------------------
# 9. COMPLETE PREPROCESSING PIPELINE
# --------------------------------------------------

def preprocess_text(text):
    """
    Complete text preprocessing pipeline.

    Returns:
        dictionary containing:
        - original text
        - cleaned text
        - tokens
        - processed tokens
        - processed text
    """

    # Step 1: Validate
    is_valid, message = validate_preprocessing_input(text)

    if not is_valid:
        return {
            "success": False,
            "message": message,
            "original_text": text,
            "cleaned_text": "",
            "tokens": [],
            "processed_tokens": [],
            "processed_text": ""
        }

    original_text = text

    # Step 2: Noise filtering
    text = remove_noise(text)

    # Step 3: Special characters
    text = handle_special_characters(text)

    # Step 4: Punctuation
    text = remove_punctuation(text)

    # Step 5: Normalize spaces
    text = normalize_spaces(text)

    # Check whether text still exists
    if not text:
        return {
            "success": False,
            "message": "No valid text remains after cleaning.",
            "original_text": original_text,
            "cleaned_text": "",
            "tokens": [],
            "processed_tokens": [],
            "processed_text": ""
        }

    cleaned_text = text.lower()

    # Step 6: Tokenization
    tokens = tokenize_text(cleaned_text)

    # Step 7: Stop-word removal
    filtered_tokens = remove_stopwords(tokens)

    # Step 8: Lemmatization
    lemmatized_tokens = lemmatize_tokens(
        filtered_tokens
    )

    # Final processed text
    processed_text = " ".join(
        lemmatized_tokens
    )

    return {
        "success": True,
        "message": "Preprocessing completed successfully.",
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "processed_tokens": lemmatized_tokens,
        "processed_text": processed_text
    }


# --------------------------------------------------
# 10. PROCESS MULTIPLE TEXTS
# --------------------------------------------------

def preprocess_texts(texts):
    """
    Preprocess multiple text entries.
    """

    results = []

    for text in texts:

        result = preprocess_text(text)

        if result["success"]:
            results.append(result)

    return results
import pandas as pd


SUPPORTED_EXTENSIONS = ["txt", "csv"]


def validate_text(text):
    """
    Validate whether the given text contains useful content.

    Returns:
        tuple: (is_valid, message)
    """

    if text is None:
        return False, "Input is empty."

    if not isinstance(text, str):
        return False, "Input must be text."

    if not text.strip():
        return False, "Input contains no valid text."

    return True, "Valid text."


def ingest_direct_text(text):
    """
    Process text entered directly by the user.
    """

    is_valid, message = validate_text(text)

    if not is_valid:
        return {
            "success": False,
            "source": "direct_text",
            "message": message,
            "texts": []
        }

    return {
        "success": True,
        "source": "direct_text",
        "message": "Text successfully ingested.",
        "texts": [text.strip()]
    }


def ingest_txt(uploaded_file):
    """
    Read and validate a TXT file.
    """

    try:
        content = uploaded_file.read()

        # Convert bytes into string
        text = content.decode("utf-8")

        is_valid, message = validate_text(text)

        if not is_valid:
            return {
                "success": False,
                "source": "txt",
                "message": message,
                "texts": []
            }

        return {
            "success": True,
            "source": "txt",
            "message": "TXT file successfully ingested.",
            "texts": [text.strip()]
        }

    except UnicodeDecodeError:
        return {
            "success": False,
            "source": "txt",
            "message": "Unable to read TXT file. Please use UTF-8 encoding.",
            "texts": []
        }

    except Exception as e:
        return {
            "success": False,
            "source": "txt",
            "message": f"Error reading TXT file: {str(e)}",
            "texts": []
        }


def ingest_csv(uploaded_file):
    """
    Read and validate a CSV file.

    Expected format:

    text
    I am feeling happy today.
    Today was difficult.
    """

    try:
        df = pd.read_csv(uploaded_file)

        # Check whether CSV contains the required column
        if "text" not in df.columns:
            return {
                "success": False,
                "source": "csv",
                "message": "Invalid CSV format. Required column: 'text'.",
                "texts": []
            }

        # Remove empty values
        texts = (
            df["text"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        # Remove blank strings
        texts = texts[texts != ""]

        if len(texts) == 0:
            return {
                "success": False,
                "source": "csv",
                "message": "CSV file contains no valid text.",
                "texts": []
            }

        return {
            "success": True,
            "source": "csv",
            "message": "CSV file successfully ingested.",
            "texts": texts.tolist()
        }

    except pd.errors.EmptyDataError:
        return {
            "success": False,
            "source": "csv",
            "message": "CSV file is empty.",
            "texts": []
        }

    except Exception as e:
        return {
            "success": False,
            "source": "csv",
            "message": f"Error reading CSV file: {str(e)}",
            "texts": []
        }


def ingest_file(uploaded_file):
    """
    Identify the uploaded file type and send it
    to the appropriate ingestion function.
    """

    if uploaded_file is None:
        return {
            "success": False,
            "source": "unknown",
            "message": "No file uploaded.",
            "texts": []
        }

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return ingest_txt(uploaded_file)

    elif filename.endswith(".csv"):
        return ingest_csv(uploaded_file)

    else:
        return {
            "success": False,
            "source": "unknown",
            "message": "Unsupported file type. Please upload a .txt or .csv file.",
            "texts": []
        }
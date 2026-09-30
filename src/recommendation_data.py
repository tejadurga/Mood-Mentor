from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WELLNESS_CONTENT_PATH = PROJECT_ROOT / "data" / "wellness_content.csv"
USER_PROFILE_PATH = PROJECT_ROOT / "data" / "user_profiles.csv"
USER_HISTORY_PATH = PROJECT_ROOT / "data" / "user_history.csv"


def parse_list(value):
    """
    Convert a semicolon-separated CSV value into a list.
    """
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def load_wellness_content():
    """
    Load wellness recommendation content.
    """
    if not WELLNESS_CONTENT_PATH.exists():
        raise FileNotFoundError(
            f"Wellness content file not found: {WELLNESS_CONTENT_PATH}"
        )

    dataframe = pd.read_csv(WELLNESS_CONTENT_PATH)

    required_columns = {
        "content_id",
        "title",
        "description",
        "category",
        "target_emotions",
        "intensity_level",
        "activity_type",
        "duration_minutes",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing wellness content columns: {sorted(missing_columns)}"
        )

    dataframe["target_emotions"] = dataframe["target_emotions"].apply(parse_list)

    return dataframe


def load_user_profiles():
    """
    Load user preference profiles.
    """
    if not USER_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"User profile file not found: {USER_PROFILE_PATH}"
        )

    dataframe = pd.read_csv(USER_PROFILE_PATH)

    required_columns = {
        "user_id",
        "preferred_categories",
        "preferred_activity_types",
        "max_duration_minutes",
        "preferred_intensity_levels",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing user profile columns: {sorted(missing_columns)}"
        )

    dataframe["preferred_categories"] = dataframe[
        "preferred_categories"
    ].apply(parse_list)

    dataframe["preferred_activity_types"] = dataframe[
        "preferred_activity_types"
    ].apply(parse_list)

    dataframe["preferred_intensity_levels"] = dataframe[
        "preferred_intensity_levels"
    ].apply(parse_list)

    return dataframe


def load_user_history():
    """
    Load previous recommendation interactions.
    """
    if not USER_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"User history file not found: {USER_HISTORY_PATH}"
        )

    dataframe = pd.read_csv(USER_HISTORY_PATH)

    required_columns = {
        "user_id",
        "timestamp",
        "content_id",
        "interaction_type",
        "emotion_state",
        "intensity_score",
        "intensity_level",
        "was_completed",
        "helpfulness_rating",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing user history columns: {sorted(missing_columns)}"
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce"
    )

    return dataframe


def get_user_profile(user_id):
    """
    Return the profile for a specific user.
    """
    profiles = load_user_profiles()

    matches = profiles[profiles["user_id"] == user_id]

    if matches.empty:
        raise ValueError(f"User profile not found: {user_id}")

    return matches.iloc[0].to_dict()


def get_user_history(user_id):
    """
    Return recommendation history for a specific user.
    """
    history = load_user_history()

    user_history = history[
        history["user_id"] == user_id
    ].copy()

    if not user_history.empty:
        user_history = user_history.sort_values(
            by="timestamp",
            ascending=False
        )

    return user_history
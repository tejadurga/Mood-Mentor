from pathlib import Path

import pandas as pd

from src.emotion_config import EMOTION_LABELS


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = {
    "Train": DATA_DIR / "emotion_train.csv",
    "Validation": DATA_DIR / "emotion_validation.csv",
    "Test": DATA_DIR / "emotion_test.csv",
}


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_dataset(name, file_path):

    print("\n" + "=" * 60)
    print(f"{name.upper()} DATASET VALIDATION")
    print("=" * 60)

    if not file_path.exists():

        print(f"❌ File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)

    print(f"\nRows: {len(df)}")

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    required_columns = [
        "text",
        *EMOTION_LABELS
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "❌ Missing columns:",
            missing_columns
        )

        return False

    print("✓ Required columns present")

    # --------------------------------------------------
    # Empty text
    # --------------------------------------------------

    empty_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Empty text rows: {empty_text}")

    # --------------------------------------------------
    # Duplicate text
    # --------------------------------------------------

    duplicate_text = df["text"].duplicated().sum()

    print(f"Duplicate text rows: {duplicate_text}")

    # --------------------------------------------------
    # Validate emotion values
    # --------------------------------------------------

    invalid_values = {}

    for emotion in EMOTION_LABELS:

        invalid = ~df[emotion].isin([0, 1])

        count = invalid.sum()

        if count > 0:
            invalid_values[emotion] = count

    if invalid_values:

        print(
            "❌ Invalid emotion values:",
            invalid_values
        )

        return False

    print("✓ Emotion labels contain only 0/1")

    # --------------------------------------------------
    # At least one emotion
    # --------------------------------------------------

    emotion_sum = df[EMOTION_LABELS].sum(axis=1)

    no_emotion = (emotion_sum == 0).sum()

    print(
        f"Rows without any target emotion: "
        f"{no_emotion}"
    )

    if no_emotion > 0:

        print(
            "❌ Some rows contain no target emotion"
        )

        return False

    print("✓ Every row has at least one emotion")

    # --------------------------------------------------
    # Multi-label examples
    # --------------------------------------------------

    multi_label = (emotion_sum > 1).sum()

    print(
        f"Multi-label rows: {multi_label}"
    )

    # --------------------------------------------------
    # Emotion distribution
    # --------------------------------------------------

    print("\nEmotion distribution:")

    for emotion in EMOTION_LABELS:

        count = int(df[emotion].sum())

        print(
            f"{emotion:10}: {count}"
        )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    if empty_text > 0:

        print("\n⚠ Dataset contains empty text rows.")

    else:

        print("\n✓ No empty text rows.")

    print(f"\n✓ {name} dataset validation completed.")

    return True


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MOOD MENTOR - DATASET QUALITY VALIDATION")
    print("=" * 60)

    results = []

    for name, file_path in DATASETS.items():

        result = validate_dataset(
            name,
            file_path
        )

        results.append(result)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULT")
    print("=" * 60)

    if all(results):

        print("\n✅ ALL DATASETS PASSED VALIDATION")

    else:

        print("\n❌ DATASET VALIDATION FAILED")

    print("\nDataset quality check completed.")


if __name__ == "__main__":
    main()
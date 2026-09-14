from pathlib import Path

from src.emotion_data import prepare_emotion_dataset
from src.emotion_config import EMOTION_LABELS


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MOOD MENTOR - EMOTION DATASET PREPARATION")
    print("=" * 60)

    print("\nLoading and preparing GoEmotions...")

    train_df, validation_df, test_df = prepare_emotion_dataset()

    print("\nDataset preparation completed.")

    # --------------------------------------------------
    # Display dataset sizes
    # --------------------------------------------------

    print("\nDataset sizes:")
    print(f"Training   : {len(train_df)}")
    print(f"Validation : {len(validation_df)}")
    print(f"Test       : {len(test_df)}")

    # --------------------------------------------------
    # Display emotion distribution
    # --------------------------------------------------

    print("\nEmotion distribution:")

    for emotion in EMOTION_LABELS:

        print(
            f"{emotion:10} | "
            f"Train: {train_df[emotion].sum():4} | "
            f"Validation: {validation_df[emotion].sum():4} | "
            f"Test: {test_df[emotion].sum():4}"
        )

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    train_path = DATA_DIR / "emotion_train.csv"
    validation_path = DATA_DIR / "emotion_validation.csv"
    test_path = DATA_DIR / "emotion_test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("\nFiles created:")

    print(f"✓ {train_path}")
    print(f"✓ {validation_path}")
    print(f"✓ {test_path}")

    # --------------------------------------------------
    # Show sample records
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE TRAINING RECORDS")
    print("=" * 60)

    for _, row in train_df.head(5).iterrows():

        print("\nText:")
        print(row["text"])

        print("Labels:")

        detected = []

        for emotion in EMOTION_LABELS:

            if row[emotion] == 1:
                detected.append(emotion)

        print(", ".join(detected))

    # --------------------------------------------------
    # Verify multi-label examples
    # --------------------------------------------------

    multi_label_count = 0

    for _, row in train_df.iterrows():

        label_count = sum(
            row[emotion]
            for emotion in EMOTION_LABELS
        )

        if label_count > 1:
            multi_label_count += 1

    print("\n" + "=" * 60)
    print("MULTI-LABEL VALIDATION")
    print("=" * 60)

    print(
        f"Training examples with multiple emotions: "
        f"{multi_label_count}"
    )

    if multi_label_count > 0:
        print("✓ Multi-label examples successfully preserved.")
    else:
        print("⚠ No multi-label examples found.")

    print("\nDataset preparation finished successfully.")


if __name__ == "__main__":
    main()
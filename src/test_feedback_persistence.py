import sys
from datetime import datetime

import pandas as pd

from src.recommendation_feedback import (
    load_feedback,
    save_feedback_record,
    get_user_feedback,
    calculate_feedback_signal,
    get_feedback_summary,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_USER_ID = "task10_test_user"
TEST_CONTENT_ID = "TASK10_TEST_CONTENT"

TEST_EMOTION_STATE = (
    "Very High-intensity fear state"
)

TEST_INTENSITY = 0.88

TEST_RATING = 5

VALID_INTERACTION = "accepted"

INVALID_INTERACTION = "invalid_type"

INVALID_RATING = 10


# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_pass(message):
    print(f"[PASS] {message}")


def print_fail(message):
    print(f"[FAIL] {message}")


def print_info(message):
    print(f"[INFO] {message}")


# ============================================================
# IDENTIFY FEEDBACK FILE
# ============================================================

def get_feedback_file():
    """
    Obtain the actual feedback file location used by
    the recommendation_feedback module.

    We inspect the module rather than hardcoding a new path.
    """

    import src.recommendation_feedback as feedback_module

    feedback_path = getattr(
        feedback_module,
        "FEEDBACK_PATH",
        None
    )

    if feedback_path is None:
        raise RuntimeError(
            "FEEDBACK_PATH was not found in "
            "recommendation_feedback.py"
        )

    return feedback_path


# ============================================================
# REMOVE ONLY OUR TEST RECORDS
# ============================================================

def cleanup_test_records(
    feedback_path
):
    """
    Remove records created by this test only.

    Existing feedback records are preserved.
    """

    if not feedback_path.exists():
        return

    try:

        dataframe = pd.read_csv(
            feedback_path
        )

    except Exception:
        return

    if dataframe.empty:
        return

    required_columns = [
        "user_id",
        "content_id"
    ]

    if not all(
        column in dataframe.columns
        for column in required_columns
    ):
        return

    mask = ~(
        (
            dataframe["user_id"].astype(str)
            ==
            TEST_USER_ID
        )
        &
        (
            dataframe["content_id"].astype(str)
            ==
            TEST_CONTENT_ID
        )
    )

    cleaned = dataframe[
        mask
    ].copy()

    cleaned.to_csv(
        feedback_path,
        index=False
    )


# ============================================================
# ORIGINAL FEEDBACK SNAPSHOT
# ============================================================

def snapshot_original_test_records(
    feedback_path
):
    """
    Preserve the number of pre-existing test records,
    allowing us to make sure only our generated records
    are removed during cleanup.
    """

    if not feedback_path.exists():
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(
            feedback_path
        )

    except Exception:
        return pd.DataFrame()

    if dataframe.empty:
        return dataframe

    if not {
        "user_id",
        "content_id"
    }.issubset(
        dataframe.columns
    ):
        return pd.DataFrame()

    return dataframe[
        (
            dataframe["user_id"].astype(str)
            ==
            TEST_USER_ID
        )
        &
        (
            dataframe["content_id"].astype(str)
            ==
            TEST_CONTENT_ID
        )
    ].copy()


# ============================================================
# 1. LOAD EXISTING FEEDBACK
# ============================================================

def test_load_feedback():

    print_section(
        "1. FEEDBACK LOADING"
    )

    feedback = load_feedback()

    if not isinstance(
        feedback,
        pd.DataFrame
    ):

        print_fail(
            "load_feedback() did not return a DataFrame."
        )

        return False

    print_pass(
        "Feedback data loaded successfully."
    )

    print_info(
        f"Existing feedback records: {len(feedback)}"
    )

    return True


# ============================================================
# 2. SAVE FEEDBACK
# ============================================================

def test_save_feedback():

    print_section(
        "2. FEEDBACK STORAGE"
    )

    record = save_feedback_record(
        user_id=TEST_USER_ID,
        content_id=TEST_CONTENT_ID,
        interaction_type=VALID_INTERACTION,
        rating=TEST_RATING,
        emotion_state=TEST_EMOTION_STATE,
        intensity_score=TEST_INTENSITY,
        preference_key="test_preference",
        preference_value="test_value",
    )

    if not isinstance(
        record,
        dict
    ):

        print_fail(
            "save_feedback_record() did not return a dictionary."
        )

        return False

    required_fields = [
        "user_id",
        "timestamp",
        "content_id",
        "interaction_type",
        "rating",
        "emotion_state",
        "intensity_score",
        "preference_key",
        "preference_value",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing_fields:

        print_fail(
            "Saved record is missing fields: "
            f"{missing_fields}"
        )

        return False

    if record["user_id"] != TEST_USER_ID:

        print_fail(
            "Saved user_id does not match test user."
        )

        return False

    if record["content_id"] != TEST_CONTENT_ID:

        print_fail(
            "Saved content_id does not match test content."
        )

        return False

    if record["interaction_type"] != VALID_INTERACTION:

        print_fail(
            "Saved interaction_type does not match."
        )

        return False

    print_pass(
        "Feedback record was created successfully."
    )

    print_info(
        f"Interaction: {record['interaction_type']}"
    )

    print_info(
        f"Rating: {record['rating']}"
    )

    return True


# ============================================================
# 3. VERIFY PERSISTENCE
# ============================================================

def test_feedback_persistence():

    print_section(
        "3. FEEDBACK PERSISTENCE"
    )

    feedback = get_user_feedback(
        TEST_USER_ID
    )

    if not isinstance(
        feedback,
        pd.DataFrame
    ):

        print_fail(
            "get_user_feedback() did not return a DataFrame."
        )

        return False

    if feedback.empty:

        print_fail(
            "Saved feedback could not be found."
        )

        return False

    matching = feedback[
        feedback["content_id"].astype(str)
        ==
        TEST_CONTENT_ID
    ]

    if matching.empty:

        print_fail(
            "Test feedback record is missing after save."
        )

        return False

    saved = matching.iloc[-1]

    checks = {
        "user_id":
            str(saved["user_id"])
            ==
            TEST_USER_ID,

        "content_id":
            str(saved["content_id"])
            ==
            TEST_CONTENT_ID,

        "interaction_type":
            str(saved["interaction_type"]).lower()
            ==
            VALID_INTERACTION,

        "rating":
            float(saved["rating"])
            ==
            float(TEST_RATING),

        "emotion_state":
            str(saved["emotion_state"])
            ==
            TEST_EMOTION_STATE,

        "intensity_score":
            abs(
                float(saved["intensity_score"])
                -
                TEST_INTENSITY
            )
            < 0.000001,
    }

    failed_checks = [
        field
        for field, passed
        in checks.items()
        if not passed
    ]

    if failed_checks:

        print_fail(
            "Persistence validation failed for: "
            f"{failed_checks}"
        )

        return False

    print_pass(
        "Saved feedback was correctly reloaded."
    )

    return True


# ============================================================
# 4. FEEDBACK SIGNAL
# ============================================================

def test_feedback_signal():

    print_section(
        "4. FEEDBACK SIGNAL"
    )

    signal = calculate_feedback_signal(
        user_id=TEST_USER_ID,
        content_id=TEST_CONTENT_ID
    )

    try:

        signal = float(
            signal
        )

    except (
        TypeError,
        ValueError
    ):

        print_fail(
            "Feedback signal is not numeric."
        )

        return False

    print_info(
        f"Calculated feedback signal: {signal:.4f}"
    )

    if not (
        0.0
        <= signal
        <= 1.0
    ):

        print_fail(
            "Feedback signal is outside the [0, 1] range."
        )

        return False

    if signal <= 0.5:

        print_fail(
            "Accepted rating-5 feedback did not "
            "produce an elevated feedback signal."
        )

        return False

    print_pass(
        "Feedback signal calculated correctly "
        "and remains within [0, 1]."
    )

    return True


# ============================================================
# 5. FEEDBACK SUMMARY
# ============================================================

def test_feedback_summary():

    print_section(
        "5. FEEDBACK SUMMARY"
    )

    summary = get_feedback_summary(
        TEST_USER_ID
    )

    if not isinstance(
        summary,
        dict
    ):

        print_fail(
            "get_feedback_summary() did not return a dictionary."
        )

        return False

    print_info(
        f"Summary: {summary}"
    )

    required_fields = [
        "total_feedback",
        "viewed",
        "accepted",
        "rejected",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in summary
    ]

    if missing_fields:

        print_fail(
            "Feedback summary is missing: "
            f"{missing_fields}"
        )

        return False

    if int(
        summary["accepted"]
    ) < 1:

        print_fail(
            "Accepted feedback was not reflected "
            "in the summary."
        )

        return False

    if int(
        summary["total_feedback"]
    ) < 1:

        print_fail(
            "Total feedback count was not updated."
        )

        return False

    print_pass(
        "Feedback summary reflects the stored interaction."
    )

    return True


# ============================================================
# 6. INVALID INTERACTION TYPE
# ============================================================

def test_invalid_interaction():

    print_section(
        "6. ERROR HANDLING — INVALID INTERACTION"
    )

    try:

        save_feedback_record(
            user_id=TEST_USER_ID,
            content_id=TEST_CONTENT_ID,
            interaction_type=INVALID_INTERACTION,
            rating=TEST_RATING,
            emotion_state=TEST_EMOTION_STATE,
            intensity_score=TEST_INTENSITY,
        )

    except ValueError as error:

        print_pass(
            "Invalid interaction type was rejected."
        )

        print_info(
            f"Error: {error}"
        )

        return True

    except Exception as error:

        print_fail(
            "Unexpected exception type: "
            f"{type(error).__name__}: {error}"
        )

        return False

    print_fail(
        "Invalid interaction type was accepted."
    )

    return False


# ============================================================
# 7. INVALID RATING
# ============================================================

def test_invalid_rating():

    print_section(
        "7. ERROR HANDLING — INVALID RATING"
    )

    try:

        save_feedback_record(
            user_id=TEST_USER_ID,
            content_id=TEST_CONTENT_ID,
            interaction_type=VALID_INTERACTION,
            rating=INVALID_RATING,
            emotion_state=TEST_EMOTION_STATE,
            intensity_score=TEST_INTENSITY,
        )

    except ValueError as error:

        print_pass(
            "Invalid rating was rejected."
        )

        print_info(
            f"Error: {error}"
        )

        return True

    except Exception as error:

        print_fail(
            "Unexpected exception type: "
            f"{type(error).__name__}: {error}"
        )

        return False

    print_fail(
        "Invalid rating was accepted."
    )

    return False


# ============================================================
# 8. MISSING CONTENT FEEDBACK SIGNAL
# ============================================================

def test_missing_content_signal():

    print_section(
        "8. MISSING FEEDBACK HANDLING"
    )

    signal = calculate_feedback_signal(
        user_id=TEST_USER_ID,
        content_id="NON_EXISTENT_CONTENT"
    )

    try:

        signal = float(
            signal
        )

    except (
        TypeError,
        ValueError
    ):

        print_fail(
            "Missing-content signal is not numeric."
        )

        return False

    print_info(
        f"Missing-content signal: {signal:.4f}"
    )

    if not (
        0.0
        <= signal
        <= 1.0
    ):

        print_fail(
            "Missing-content signal is outside [0, 1]."
        )

        return False

    print_pass(
        "Missing-content feedback is handled safely."
    )

    return True


# ============================================================
# 9. CLEANUP
# ============================================================

def test_cleanup(
    feedback_path
):

    print_section(
        "9. TEST DATA CLEANUP"
    )

    try:

        cleanup_test_records(
            feedback_path
        )

        if not feedback_path.exists():

            print_pass(
                "Temporary feedback data cleaned up."
            )

            return True

        dataframe = pd.read_csv(
            feedback_path
        )

        if dataframe.empty:

            print_pass(
                "Temporary feedback data cleaned up."
            )

            return True

        remaining = dataframe[
            (
                dataframe["user_id"].astype(str)
                ==
                TEST_USER_ID
            )
            &
            (
                dataframe["content_id"].astype(str)
                ==
                TEST_CONTENT_ID
            )
        ]

        if remaining.empty:

            print_pass(
                "Temporary feedback record removed."
            )

            return True

        print_fail(
            "Test feedback record still exists."
        )

        return False

    except Exception as error:

        print_fail(
            f"Cleanup failed: {error}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "MOOD MENTOR — TASK 10.2 FEEDBACK & ERROR HANDLING TEST"
    )

    print_info(
        f"Python version: {sys.version.split()[0]}"
    )

    feedback_path = None

    original_test_records = pd.DataFrame()

    results = []

    try:

        # ----------------------------------------------------
        # Get actual feedback path
        # ----------------------------------------------------

        feedback_path = get_feedback_file()

        print_info(
            f"Feedback file: {feedback_path}"
        )

        # ----------------------------------------------------
        # Snapshot existing matching test records
        # ----------------------------------------------------

        original_test_records = (
            snapshot_original_test_records(
                feedback_path
            )
        )

        # ----------------------------------------------------
        # Remove stale test records before starting
        # ----------------------------------------------------

        cleanup_test_records(
            feedback_path
        )

        # ----------------------------------------------------
        # Run tests
        # ----------------------------------------------------

        results.append(
            (
                "Load Feedback",
                test_load_feedback()
            )
        )

        results.append(
            (
                "Save Feedback",
                test_save_feedback()
            )
        )

        results.append(
            (
                "Persistence",
                test_feedback_persistence()
            )
        )

        results.append(
            (
                "Feedback Signal",
                test_feedback_signal()
            )
        )

        results.append(
            (
                "Feedback Summary",
                test_feedback_summary()
            )
        )

        results.append(
            (
                "Invalid Interaction",
                test_invalid_interaction()
            )
        )

        results.append(
            (
                "Invalid Rating",
                test_invalid_rating()
            )
        )

        results.append(
            (
                "Missing Content",
                test_missing_content_signal()
            )
        )

    except Exception as error:

        print_fail(
            f"Unexpected Task 10.2 error: {error}"
        )

        results.append(
            (
                "Unexpected Error",
                False
            )
        )

    finally:

        # ----------------------------------------------------
        # Cleanup our temporary records
        # ----------------------------------------------------

        if feedback_path is not None:

            # First remove all records generated by this test.
            cleanup_test_records(
                feedback_path
            )

            # Restore any matching records that existed before
            # this test started.
            if (
                not original_test_records.empty
                and feedback_path.exists()
            ):

                try:

                    current = pd.read_csv(
                        feedback_path
                    )

                    # Avoid accidental duplication.
                    current = current[
                        ~(
                            (
                                current[
                                    "user_id"
                                ].astype(str)
                                ==
                                TEST_USER_ID
                            )
                            &
                            (
                                current[
                                    "content_id"
                                ].astype(str)
                                ==
                                TEST_CONTENT_ID
                            )
                        )
                    ].copy()

                    restored = pd.concat(
                        [
                            current,
                            original_test_records
                        ],
                        ignore_index=True
                    )

                    restored.to_csv(
                        feedback_path,
                        index=False
                    )

                except Exception as error:

                    print_fail(
                        "Could not restore pre-existing matching "
                        f"records: {error}"
                    )

            cleanup_result = (
                test_cleanup(
                    feedback_path
                )
            )

            results.append(
                (
                    "Cleanup",
                    cleanup_result
                )
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "TASK 10.2 FINAL SUMMARY"
    )

    passed_count = sum(
        1
        for _, passed in results
        if passed
    )

    total_count = len(
        results
    )

    for test_name, passed in results:

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{test_name:<25} {status}"
        )

    print()

    print(
        f"Tests passed: "
        f"{passed_count}/{total_count}"
    )

    all_passed = (
        total_count > 0
        and passed_count == total_count
    )

    print()

    if all_passed:

        print(
            "=" * 80
        )

        print(
            "TASK 10.2 STATUS: PASSED"
        )

        print(
            "=" * 80
        )

    else:

        print(
            "=" * 80
        )

        print(
            "TASK 10.2 STATUS: NEEDS FIXES"
        )

        print(
            "=" * 80
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
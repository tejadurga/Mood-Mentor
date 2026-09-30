import sys
import traceback

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.emotion_config import EMOTION_LABELS
from src.ingestion import ingest_direct_text
from src.preprocessing import preprocess_texts
from src.sentiment import analyze_sentiment
from src.emotion_state import analyze_emotional_state

from src.recommendation_data import (
    get_user_profile,
    get_user_history
)

from src.recommendation_engine import (
    build_hybrid_recommendations
)

from src.recommendation_ranking import (
    rank_recommendations
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/bert_emotion"

BERT_THRESHOLD = 0.50

DEFAULT_USER_ID = "demo_user_001"

MAX_LENGTH = 128


# ============================================================
# TEST INPUTS
# ============================================================

TEST_CASES = [
    {
        "name": "Fear Case",
        "user_id": "demo_user_001",
        "text": (
            "I feel scared and overwhelmed "
            "about everything happening right now."
        )
    },
    {
        "name": "Joy Case",
        "user_id": "demo_user_002",
        "text": (
            "I am feeling very happy and excited "
            "about the wonderful surprise I received today."
        )
    }
]


# ============================================================
# RESULT HELPERS
# ============================================================

def print_section(
    title
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_pass(
    message
):
    print(f"[PASS] {message}")


def print_fail(
    message
):
    print(f"[FAIL] {message}")


def print_info(
    message
):
    print(f"[INFO] {message}")


# ============================================================
# BERT MODEL
# ============================================================

def load_bert_model():

    print_info(
        "Loading trained BERT model..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )
    )

    model.eval()

    print_pass(
        "Trained BERT model loaded."
    )

    return tokenizer, model


# ============================================================
# BERT PREDICTION
# ============================================================

def analyze_bert_emotions(
    text,
    tokenizer,
    model
):
    """
    Run multi-label BERT emotion prediction.
    """

    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
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

    probability_array = (
        probabilities
        .cpu()
        .numpy()
    )

    emotion_results = []

    for emotion, probability in zip(
        EMOTION_LABELS,
        probability_array
    ):

        confidence = float(
            probability
        )

        emotion_results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "detected": (
                    confidence
                    >= BERT_THRESHOLD
                )
            }
        )

    detected_emotions = [
        item["emotion"]
        for item in emotion_results
        if item["detected"]
    ]

    primary_index = int(
        np.argmax(
            probability_array
        )
    )

    primary_emotion = (
        EMOTION_LABELS[
            primary_index
        ]
    )

    primary_confidence = float(
        probability_array[
            primary_index
        ]
    )

    return {
        "probabilities": probability_array,
        "emotion_results": emotion_results,
        "detected_emotions": detected_emotions,
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_confidence
    }


# ============================================================
# SINGLE CASE WORKFLOW
# ============================================================

def run_workflow_case(
    case,
    tokenizer,
    model
):

    case_name = case["name"]

    user_id = case["user_id"]

    original_text = case["text"]

    print_section(
        f"WORKFLOW TEST — {case_name}"
    )

    print_info(
        f"User ID: {user_id}"
    )

    print_info(
        f"Input: {original_text}"
    )

    # --------------------------------------------------------
    # 1. USER PROFILE
    # --------------------------------------------------------

    try:

        profile = get_user_profile(
            user_id
        )

        if profile is None:

            raise ValueError(
                f"User profile not found: {user_id}"
            )

        print_pass(
            "User profile loaded."
        )

    except Exception as error:

        print_fail(
            f"User profile failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 2. TEXT INGESTION
    # --------------------------------------------------------

    try:

        ingestion_result = (
            ingest_direct_text(
                original_text
            )
        )

        if not ingestion_result["success"]:

            raise ValueError(
                ingestion_result["message"]
            )

        if not ingestion_result["texts"]:

            raise ValueError(
                "No text returned after ingestion."
            )

        print_pass(
            "Text ingestion completed."
        )

    except Exception as error:

        print_fail(
            f"Text ingestion failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 3. PREPROCESSING
    # --------------------------------------------------------

    try:

        preprocessing_results = (
            preprocess_texts(
                ingestion_result["texts"]
            )
        )

        if not preprocessing_results:

            raise ValueError(
                "Preprocessing returned no results."
            )

        processed_data = (
            preprocessing_results[0]
        )

        processed_text = (
            processed_data["processed_text"]
        )

        if not processed_text.strip():

            raise ValueError(
                "Processed text is empty."
            )

        print_pass(
            "Text preprocessing completed."
        )

    except Exception as error:

        print_fail(
            f"Preprocessing failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 4. VADER SENTIMENT
    # --------------------------------------------------------

    try:

        sentiment_result = (
            analyze_sentiment(
                processed_text
            )
        )

        if not sentiment_result["success"]:

            raise ValueError(
                sentiment_result["message"]
            )

        print_pass(
            "VADER sentiment analysis completed."
        )

        print_info(
            "Sentiment: "
            f"{sentiment_result['sentiment']}"
        )

        print_info(
            "Compound: "
            f"{sentiment_result['compound']:.4f}"
        )

    except Exception as error:

        print_fail(
            f"VADER analysis failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 5. BERT EMOTION ANALYSIS
    # --------------------------------------------------------

    try:

        emotion_result = (
            analyze_bert_emotions(
                original_text,
                tokenizer,
                model
            )
        )

        if len(
            emotion_result["probabilities"]
        ) != len(
            EMOTION_LABELS
        ):

            raise ValueError(
                "BERT probability count does not "
                "match configured emotion labels."
            )

        print_pass(
            "BERT emotion prediction completed."
        )

        print_info(
            "Primary emotion: "
            f"{emotion_result['primary_emotion']}"
        )

        print_info(
            "Primary confidence: "
            f"{emotion_result['primary_confidence']:.4f}"
        )

        print_info(
            "Detected emotions: "
            f"{emotion_result['detected_emotions']}"
        )

    except Exception as error:

        print_fail(
            f"BERT emotion analysis failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 6. EMOTIONAL STATE
    # --------------------------------------------------------

    try:

        emotional_state = (
            analyze_emotional_state(
                probabilities=(
                    emotion_result[
                        "probabilities"
                    ]
                ),
                sentiment_result=(
                    sentiment_result
                ),
                threshold=BERT_THRESHOLD
            )
        )

        required_state_keys = [
            "dominant_emotion",
            "emotion_intensity",
            "intensity_level",
            "emotion_severity",
            "detected_emotions",
            "emotional_state",
            "polarity"
        ]

        missing_keys = [
            key
            for key in required_state_keys
            if key not in emotional_state
        ]

        if missing_keys:

            raise ValueError(
                "Emotional state result is missing: "
                f"{missing_keys}"
            )

        print_pass(
            "Emotional state analysis completed."
        )

        print_info(
            "Dominant emotion: "
            f"{emotional_state['dominant_emotion']}"
        )

        print_info(
            "Intensity: "
            f"{emotional_state['emotion_intensity']:.4f}"
        )

        print_info(
            "Intensity level: "
            f"{emotional_state['intensity_level']}"
        )

        print_info(
            "Emotional state: "
            f"{emotional_state['emotional_state']}"
        )

    except Exception as error:

        print_fail(
            f"Emotional state analysis failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 7. USER HISTORY
    # --------------------------------------------------------

    try:

        history = get_user_history(
            user_id
        )

        if history is None:

            raise ValueError(
                "User history returned None."
            )

        if not isinstance(
            history,
            pd.DataFrame
        ):

            raise TypeError(
                "User history must be a Pandas DataFrame."
            )

        print_pass(
            "User emotional history loaded."
        )

        print_info(
            f"History records: {len(history)}"
        )

    except Exception as error:

        print_fail(
            f"User history failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 8. HYBRID RECOMMENDATION MODEL
    # --------------------------------------------------------

    try:

        detected_emotions = (
            emotional_state[
                "detected_emotions"
            ]
        )

        hybrid_recommendations = (
            build_hybrid_recommendations(
                user_id=user_id,
                user_text=original_text,
                detected_emotions=(
                    detected_emotions
                ),
                emotion_intensity=float(
                    emotional_state[
                        "emotion_intensity"
                    ]
                ),
                emotional_state=(
                    emotional_state[
                        "emotional_state"
                    ]
                )
            )
        )

        if hybrid_recommendations is None:

            raise ValueError(
                "Hybrid engine returned None."
            )

        if not isinstance(
            hybrid_recommendations,
            pd.DataFrame
        ):

            raise TypeError(
                "Hybrid engine must return "
                "a Pandas DataFrame."
            )

        if hybrid_recommendations.empty:

            raise ValueError(
                "Hybrid engine returned "
                "zero recommendation candidates."
            )

        if "content_id" not in (
            hybrid_recommendations.columns
        ):

            raise ValueError(
                "Hybrid recommendations do not "
                "contain 'content_id'."
            )

        if "hybrid_score" not in (
            hybrid_recommendations.columns
        ):

            raise ValueError(
                "Hybrid recommendations do not "
                "contain 'hybrid_score'."
            )

        print_pass(
            "Hybrid recommendation model completed."
        )

        print_info(
            "Candidate recommendations: "
            f"{len(hybrid_recommendations)}"
        )

    except Exception as error:

        print_fail(
            f"Hybrid recommendation failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 9. SCORE VALIDATION
    # --------------------------------------------------------

    try:

        scores = pd.to_numeric(
            hybrid_recommendations[
                "hybrid_score"
            ],
            errors="coerce"
        )

        if scores.isna().any():

            raise ValueError(
                "Hybrid score contains NaN values."
            )

        if (
            (scores < 0.0)
            |
            (scores > 1.0)
        ).any():

            raise ValueError(
                "Hybrid score contains values outside [0, 1]."
            )

        print_pass(
            "Hybrid scores are valid and within [0, 1]."
        )

    except Exception as error:

        print_fail(
            f"Hybrid score validation failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 10. RANKING
    # --------------------------------------------------------

    try:

        ranking_result = (
            rank_recommendations(
                recommendations=(
                    hybrid_recommendations
                ),
                top_k=5,
                minimum_score=0.30
            )
        )

        if not isinstance(
            ranking_result,
            dict
        ):

            raise TypeError(
                "Ranking result must be a dictionary."
            )

        if "recommendations" not in (
            ranking_result
        ):

            raise ValueError(
                "Ranking result does not contain "
                "'recommendations'."
            )

        ranked_recommendations = (
            ranking_result[
                "recommendations"
            ]
        )

        if ranked_recommendations is None:

            raise ValueError(
                "Ranking returned None."
            )

        if not isinstance(
            ranked_recommendations,
            pd.DataFrame
        ):

            raise TypeError(
                "Ranked recommendations must "
                "be a Pandas DataFrame."
            )

        if ranked_recommendations.empty:

            raise ValueError(
                "Ranking returned zero recommendations."
            )

        if "rank" not in (
            ranked_recommendations.columns
        ):

            raise ValueError(
                "Ranked recommendations do not "
                "contain 'rank'."
            )

        print_pass(
            "Recommendation ranking completed."
        )

    except Exception as error:

        print_fail(
            f"Recommendation ranking failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # 11. DYNAMIC TOP RECOMMENDATION
    # --------------------------------------------------------

    try:

        top_row = (
            ranked_recommendations.iloc[0]
        )

        top_content_id = (
            str(
                top_row[
                    "content_id"
                ]
            )
        )

        top_score = float(
            top_row[
                "hybrid_score"
            ]
        )

        print_pass(
            "Dynamic top recommendation generated."
        )

        print_info(
            f"Top content ID: {top_content_id}"
        )

        print_info(
            f"Top hybrid score: {top_score:.4f}"
        )

    except Exception as error:

        print_fail(
            f"Top recommendation validation failed: {error}"
        )

        raise

    # --------------------------------------------------------
    # CASE RESULT
    # --------------------------------------------------------

    return {
        "case_name": case_name,
        "user_id": user_id,
        "text": original_text,
        "sentiment": sentiment_result[
            "sentiment"
        ],
        "compound": float(
            sentiment_result[
                "compound"
            ]
        ),
        "detected_emotions": (
            emotion_result[
                "detected_emotions"
            ]
        ),
        "primary_emotion": (
            emotion_result[
                "primary_emotion"
            ]
        ),
        "primary_confidence": (
            emotion_result[
                "primary_confidence"
            ]
        ),
        "emotion_intensity": float(
            emotional_state[
                "emotion_intensity"
            ]
        ),
        "emotional_state": (
            emotional_state[
                "emotional_state"
            ]
        ),
        "history_records": len(
            history
        ),
        "candidate_count": len(
            hybrid_recommendations
        ),
        "ranked_count": len(
            ranked_recommendations
        ),
        "top_content_id": top_content_id,
        "top_score": top_score
    }


# ============================================================
# DYNAMIC BEHAVIOR TEST
# ============================================================

def validate_dynamic_behavior(
    results
):

    print_section(
        "DYNAMIC ML BEHAVIOR VALIDATION"
    )

    if len(results) < 2:

        print_fail(
            "Not enough test cases for dynamic comparison."
        )

        return False

    first = results[0]

    second = results[1]

    # --------------------------------------------------------
    # Emotion difference
    # --------------------------------------------------------

    emotion_changed = (
        first["detected_emotions"]
        !=
        second["detected_emotions"]
    )

    # --------------------------------------------------------
    # Primary emotion difference
    # --------------------------------------------------------

    primary_changed = (
        first["primary_emotion"]
        !=
        second["primary_emotion"]
    )

    # --------------------------------------------------------
    # Recommendation difference
    # --------------------------------------------------------

    recommendation_changed = (
        first["top_content_id"]
        !=
        second["top_content_id"]
    )

    # --------------------------------------------------------
    # Score difference
    # --------------------------------------------------------

    score_changed = (
        abs(
            first["top_score"]
            -
            second["top_score"]
        )
        > 0.000001
    )

    print_info(
        f"Case 1 emotions: "
        f"{first['detected_emotions']}"
    )

    print_info(
        f"Case 2 emotions: "
        f"{second['detected_emotions']}"
    )

    print_info(
        f"Case 1 top recommendation: "
        f"{first['top_content_id']}"
    )

    print_info(
        f"Case 2 top recommendation: "
        f"{second['top_content_id']}"
    )

    print_info(
        f"Emotion output changed: "
        f"{emotion_changed}"
    )

    print_info(
        f"Primary emotion changed: "
        f"{primary_changed}"
    )

    print_info(
        f"Top recommendation changed: "
        f"{recommendation_changed}"
    )

    print_info(
        f"Top score changed: "
        f"{score_changed}"
    )

    # At least one output characteristic should change.
    dynamic = (
        emotion_changed
        or
        primary_changed
        or
        recommendation_changed
        or
        score_changed
    )

    if dynamic:

        print_pass(
            "ML workflow demonstrates dynamic behavior."
        )

        return True

    print_fail(
        "Two different inputs produced identical "
        "emotion and recommendation outputs."
    )

    return False


# ============================================================
# EMPTY INPUT VALIDATION
# ============================================================

def validate_empty_input():

    print_section(
        "ERROR HANDLING — EMPTY INPUT"
    )

    try:

        result = (
            ingest_direct_text("")
        )

        if result["success"]:

            print_fail(
                "Empty input was incorrectly accepted."
            )

            return False

        print_pass(
            "Empty input handled correctly."
        )

        print_info(
            f"Message: {result['message']}"
        )

        return True

    except Exception as error:

        print_fail(
            f"Unexpected exception for empty input: {error}"
        )

        return False


# ============================================================
# SUMMARY
# ============================================================

def print_final_summary(
    results,
    dynamic_passed,
    empty_input_passed
):

    print_section(
        "TASK 10.1 FINAL SUMMARY"
    )

    print(
        f"Workflow cases passed : {len(results)}"
    )

    print(
        f"Dynamic behavior      : "
        f"{'PASS' if dynamic_passed else 'FAIL'}"
    )

    print(
        f"Empty input handling  : "
        f"{'PASS' if empty_input_passed else 'FAIL'}"
    )

    print()

    if results:

        summary_rows = []

        for result in results:

            summary_rows.append(
                {
                    "Case":
                        result["case_name"],

                    "User":
                        result["user_id"],

                    "Sentiment":
                        result["sentiment"],

                    "Primary Emotion":
                        result["primary_emotion"],

                    "Confidence":
                        round(
                            result[
                                "primary_confidence"
                            ],
                            4
                        ),

                    "Intensity":
                        round(
                            result[
                                "emotion_intensity"
                            ],
                            4
                        ),

                    "Top Recommendation":
                        result[
                            "top_content_id"
                        ],

                    "Top Score":
                        round(
                            result[
                                "top_score"
                            ],
                            4
                        )
                }
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

        print()
        print(
            summary_df.to_string(
                index=False
            )
        )

    print()

    overall_passed = (
        len(results) == len(TEST_CASES)
        and dynamic_passed
        and empty_input_passed
    )

    if overall_passed:

        print(
            "=" * 80
        )

        print(
            "TASK 10.1 STATUS: PASSED"
        )

        print(
            "=" * 80
        )

    else:

        print(
            "=" * 80
        )

        print(
            "TASK 10.1 STATUS: NEEDS FIXES"
        )

        print(
            "=" * 80
        )

    return overall_passed


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "MOOD MENTOR — TASK 10.1 COMPLETE ML WORKFLOW TEST"
    )

    print_info(
        f"Python version: {sys.version.split()[0]}"
    )

    print_info(
        f"PyTorch version: {torch.__version__}"
    )

    print_info(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    results = []

    try:

        tokenizer, model = (
            load_bert_model()
        )

    except Exception:

        print_fail(
            "Cannot continue because BERT model failed to load."
        )

        traceback.print_exc()

        sys.exit(1)

    # --------------------------------------------------------
    # RUN WORKFLOW CASES
    # --------------------------------------------------------

    for case in TEST_CASES:

        try:

            result = run_workflow_case(
                case=case,
                tokenizer=tokenizer,
                model=model
            )

            results.append(
                result
            )

        except Exception:

            print()
            print(
                traceback.format_exc()
            )

    # --------------------------------------------------------
    # DYNAMIC VALIDATION
    # --------------------------------------------------------

    dynamic_passed = (
        validate_dynamic_behavior(
            results
        )
        if len(results) >= 2
        else False
    )

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    empty_input_passed = (
        validate_empty_input()
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    overall_passed = (
        print_final_summary(
            results=results,
            dynamic_passed=dynamic_passed,
            empty_input_passed=empty_input_passed
        )
    )

    if not overall_passed:

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
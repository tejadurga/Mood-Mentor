import time
import math
import statistics

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "models/bert_emotion"

TEST_TEXTS = [
    "I am feeling very happy and excited today.",
    "I feel scared and worried about what might happen.",
    "I am angry and frustrated because everything went wrong.",
    "I feel sad, tired, and alone today.",
    "I am happy but also nervous about tomorrow.",
    "I received wonderful news and I feel joyful.",
    "I am afraid that something bad may happen.",
    "I am extremely angry about the situation.",
    "I feel disgusted by what happened.",
    "I am surprised by the unexpected result.",
]


TOTAL_REQUESTS = 100


def run_stress_test():

    print("=" * 60)
    print("MOOD MENTOR — BERT STRESS TEST")
    print("=" * 60)

    print("\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    print("Model loaded successfully.")
    print(f"Device: {device}")
    print(f"Total requests: {TOTAL_REQUESTS}")

    # ------------------------------------------------
    # WARM-UP
    # ------------------------------------------------

    warmup_inputs = tokenizer(
        TEST_TEXTS[0],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    warmup_inputs = {
        key: value.to(device)
        for key, value in warmup_inputs.items()
    }

    with torch.no_grad():
        model(**warmup_inputs)

    print("Warm-up complete.")

    # ------------------------------------------------
    # STRESS TEST
    # ------------------------------------------------

    latencies = []
    successful_requests = 0
    failed_requests = 0

    start_total = time.perf_counter()

    for index in range(TOTAL_REQUESTS):

        text = TEST_TEXTS[
            index % len(TEST_TEXTS)
        ]

        try:

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.sigmoid(
                outputs.logits
            )

            elapsed = (
                time.perf_counter() - start_time
            ) * 1000

            # ----------------------------------------
            # OUTPUT VALIDATION
            # ----------------------------------------

            if probabilities.shape != (1, 6):
                raise ValueError(
                    f"Unexpected output shape: "
                    f"{probabilities.shape}"
                )

            if not torch.isfinite(
                probabilities
            ).all():
                raise ValueError(
                    "Prediction contains NaN or infinity."
                )

            values = probabilities.cpu().numpy()[0]

            if any(
                math.isnan(float(value))
                or math.isinf(float(value))
                for value in values
            ):
                raise ValueError(
                    "Invalid numerical prediction."
                )

            latencies.append(elapsed)
            successful_requests += 1

        except Exception as error:

            failed_requests += 1

            print(
                f"Request {index + 1} FAILED: "
                f"{error}"
            )

    total_time = (
        time.perf_counter() - start_total
    )

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)

    print(
        f"Total requests      : {TOTAL_REQUESTS}"
    )

    print(
        f"Successful requests : {successful_requests}"
    )

    print(
        f"Failed requests     : {failed_requests}"
    )

    print(
        f"Total execution     : {total_time:.2f} sec"
    )

    if latencies:

        print(
            f"Average latency     : "
            f"{statistics.mean(latencies):.2f} ms"
        )

        print(
            f"Median latency      : "
            f"{statistics.median(latencies):.2f} ms"
        )

        print(
            f"Minimum latency     : "
            f"{min(latencies):.2f} ms"
        )

        print(
            f"Maximum latency     : "
            f"{max(latencies):.2f} ms"
        )

    # ------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------

    if (
        successful_requests == TOTAL_REQUESTS
        and failed_requests == 0
    ):
        print("\nStress test status: PASSED")
        print(
            "All 100 requests completed successfully."
        )
    else:
        print("\nStress test status: FAILED")

    print("=" * 60)


if __name__ == "__main__":
    run_stress_test()
import time
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


def benchmark_model():
    print("=" * 60)
    print("MOOD MENTOR — BERT PERFORMANCE TEST")
    print("=" * 60)

    print("\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    print("Model loaded successfully.")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    # ------------------------------------------------
    # WARM-UP
    # ------------------------------------------------

    print("\nRunning warm-up...")

    sample = tokenizer(
        TEST_TEXTS[0],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    sample = {
        key: value.to(device)
        for key, value in sample.items()
    }

    with torch.no_grad():
        model(**sample)

    print("Warm-up complete.")

    # ------------------------------------------------
    # LATENCY TEST
    # ------------------------------------------------

    print("\nRunning latency benchmark...")

    times = []

    for text in TEST_TEXTS:

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

        times.append(elapsed)

        print(
            f"{elapsed:8.2f} ms | "
            f"{text[:55]}"
        )

        # Validate output shape
        if probabilities.shape[-1] != 6:
            raise ValueError(
                f"Expected 6 emotions, "
                f"got {probabilities.shape[-1]}"
            )

    # ------------------------------------------------
    # STATISTICS
    # ------------------------------------------------

    average_time = statistics.mean(times)
    median_time = statistics.median(times)

    if len(times) >= 2:
        p95_time = sorted(times)[
            int(len(times) * 0.95) - 1
        ]
    else:
        p95_time = times[0]

    minimum_time = min(times)
    maximum_time = max(times)

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print(f"Requests tested : {len(times)}")
    print(f"Minimum latency : {minimum_time:.2f} ms")
    print(f"Average latency : {average_time:.2f} ms")
    print(f"Median latency  : {median_time:.2f} ms")
    print(f"P95 latency     : {p95_time:.2f} ms")
    print(f"Maximum latency : {maximum_time:.2f} ms")

    print("\nOutput validation: PASSED")
    print("All predictions returned 6 emotion probabilities.")

    print("\nPerformance test completed successfully.")


if __name__ == "__main__":
    benchmark_model()
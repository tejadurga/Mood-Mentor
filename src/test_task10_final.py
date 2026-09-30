from pathlib import Path
import ast


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "app.py",
    "src/ingestion.py",
    "src/preprocessing.py",
    "src/sentiment.py",
    "src/report.py",
    "src/emotion_config.py",
    "src/emotion_state.py",
    "src/recommendation_data.py",
    "src/recommendation_rules.py",
    "src/recommendation_engine.py",
    "src/recommendation_ranking.py",
    "src/semantic_matching.py",
    "src/emotion_trends.py",
    "src/recommendation_feedback.py",
    "src/recommendation_explainability.py",
    "data/recommendation_feedback.csv",
]


def check_required_files():
    print("\n" + "=" * 80)
    print("1. REQUIRED PROJECT FILES")
    print("=" * 80)

    missing = []

    for file_name in REQUIRED_FILES:
        path = PROJECT_ROOT / file_name

        if path.exists():
            print(f"[PASS] {file_name}")
        else:
            print(f"[FAIL] Missing: {file_name}")
            missing.append(file_name)

    return len(missing) == 0


def check_model():
    print("\n" + "=" * 80)
    print("2. TRAINED BERT MODEL")
    print("=" * 80)

    model_path = PROJECT_ROOT / "models" / "bert_emotion"

    if model_path.exists():
        files = list(model_path.iterdir())

        if files:
            print("[PASS] BERT emotion model directory exists.")
            print(f"[INFO] Model files found: {len(files)}")
            return True

    print("[FAIL] Trained BERT model not found.")
    return False


def check_python_files():
    print("\n" + "=" * 80)
    print("3. PYTHON SOURCE VALIDATION")
    print("=" * 80)

    source_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "src",
    ]

    failures = []

    for source in source_files:
        files = [source] if source.is_file() else list(source.glob("*.py"))

        for file_path in files:
            try:
                source_text = file_path.read_text(encoding="utf-8")
                ast.parse(source_text)
                print(f"[PASS] Syntax valid: {file_path.relative_to(PROJECT_ROOT)}")
            except Exception as error:
                print(f"[FAIL] {file_path}: {error}")
                failures.append(file_path)

    return len(failures) == 0


def check_for_debug_artifacts():
    print("\n" + "=" * 80)
    print("4. DEBUG / TEMPORARY ARTIFACT CHECK")
    print("=" * 80)

    suspicious_terms = [
        "breakpoint()",
        "pdb.set_trace()",
        "TODO_REMOVE",
        "TEMP_DEBUG",
    ]

    found = []

    for folder in [PROJECT_ROOT / "src", PROJECT_ROOT]:
        if not folder.exists():
            continue

        files = (
            [folder]
            if folder.is_file()
            else [
                p for p in folder.rglob("*.py")
                if ".venv" not in str(p)
                and "venv" not in str(p)
                and "__pycache__" not in str(p)
            ]
        )

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")

                for term in suspicious_terms:
                    if term in text:
                        found.append((file_path, term))

            except Exception:
                pass

    if not found:
        print("[PASS] No obvious debug artifacts found.")
        return True

    for file_path, term in found:
        print(f"[FAIL] Found {term} in {file_path}")

    return False


def check_hardcoded_ml_outputs():
    print("\n" + "=" * 80)
    print("5. HARDCODED ML OUTPUT CHECK")
    print("=" * 80)

    engine_path = PROJECT_ROOT / "src" / "recommendation_engine.py"
    app_path = PROJECT_ROOT / "app.py"

    suspicious = [
        'return "fear"',
        'return "joy"',
        'return "sadness"',
        'return "anger"',
        'return "surprise"',
        'return "disgust"',
    ]

    found = []

    for path in [engine_path, app_path]:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        for pattern in suspicious:
            if pattern in text:
                found.append((path.name, pattern))

    if not found:
        print("[PASS] No obvious hardcoded emotion outputs found.")
        return True

    for file_name, pattern in found:
        print(f"[FAIL] Possible hardcoded output: {file_name} -> {pattern}")

    return False


def main():
    print("\n" + "=" * 80)
    print("MOOD MENTOR — TASK 10 FINAL REGRESSION TEST")
    print("=" * 80)

    results = []

    results.append(("Required Files", check_required_files()))
    results.append(("BERT Model", check_model()))
    results.append(("Python Source Validation", check_python_files()))
    results.append(("Debug Artifact Check", check_for_debug_artifacts()))
    results.append(("Hardcoded ML Output Check", check_hardcoded_ml_outputs()))

    print("\n" + "=" * 80)
    print("TASK 10 FINAL SUMMARY")
    print("=" * 80)

    passed = 0

    for name, result in results:
        status = "PASS" if result else "FAIL"

        print(f"{name:<35} {status}")

        if result:
            passed += 1

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {passed}/{len(results)} CHECKS PASSED")
    print("=" * 80)

    if passed == len(results):
        print("\nTASK 10 STATUS: PASSED")
    else:
        print("\nTASK 10 STATUS: REVIEW REQUIRED")


if __name__ == "__main__":
    main()
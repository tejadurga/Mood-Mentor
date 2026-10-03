from pathlib import Path
import ast
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = PROJECT_ROOT / "data"

# Files used for testing, training, evaluation, benchmarking,
# inspection, and validation are intentionally excluded from
# the production debug-output scan.
EXCLUDED_PREFIXES = (
    "test_",
    "train_",
    "distilbert_training",
    "evaluate_",
    "validate_",
    "inspect_",
    "prepare_",
    "performance_",
    "stress_",
    "security_validation",
)

SENSITIVE_PATTERNS = [
    r"api[_-]?key\s*=",
    r"secret[_-]?key\s*=",
    r"password\s*=",
    r"access[_-]?token\s*=",
    r"auth[_-]?token\s*=",
]


def get_production_python_files():
    """
    Return production Python files only.
    Includes app.py and operational modules in src/.
    Excludes test/benchmark/training/validation utilities.
    """

    files = []

    app_path = PROJECT_ROOT / "app.py"

    if app_path.exists():
        files.append(app_path)

    src_directory = PROJECT_ROOT / "src"

    if src_directory.exists():

        for file_path in src_directory.glob("*.py"):

            filename = file_path.name

            if filename.startswith(
                EXCLUDED_PREFIXES
            ):
                continue

            files.append(file_path)

    return files


def scan_source_files():

    print("=" * 60)
    print("MOOD MENTOR — SECURITY VALIDATION")
    print("=" * 60)

    python_files = get_production_python_files()

    findings = []

    for file_path in python_files:

        try:

            content = file_path.read_text(
                encoding="utf-8"
            )

        except Exception as error:

            findings.append(
                f"{file_path.name}: "
                f"unable to read file ({error})"
            )

            continue

        # --------------------------------------------
        # Credential pattern scan
        # --------------------------------------------

        for pattern in SENSITIVE_PATTERNS:

            if re.search(
                pattern,
                content,
                flags=re.IGNORECASE
            ):

                findings.append(
                    f"{file_path.name}: "
                    "possible sensitive credential pattern"
                )

        # --------------------------------------------
        # Actual Python debug-call scan
        # --------------------------------------------

        try:

            tree = ast.parse(
                content,
                filename=str(file_path)
            )

        except SyntaxError as error:

            findings.append(
                f"{file_path.name}: "
                f"syntax error ({error})"
            )

            continue

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call
            ):
                continue

            function_name = ""

            # print(...)
            if isinstance(
                node.func,
                ast.Name
            ):

                function_name = node.func.id

            # pdb.set_trace(...)
            elif isinstance(
                node.func,
                ast.Attribute
            ):

                if (
                    isinstance(
                        node.func.value,
                        ast.Name
                    )
                    and node.func.value.id == "pdb"
                    and node.func.attr == "set_trace"
                ):

                    function_name = "pdb.set_trace"

                elif node.func.attr == "breakpoint":

                    function_name = "breakpoint"

            if function_name in (
                "print",
                "pdb.set_trace",
                "breakpoint",
            ):

                findings.append(
                    f"{file_path.name}: "
                    f"actual {function_name}() call "
                    f"at line {node.lineno}"
                )

    print(
        f"\nProduction Python files scanned: "
        f"{len(python_files)}"
    )

    credential_findings = [
        item
        for item in findings
        if "credential" in item
        or "key" in item.lower()
        or "token" in item.lower()
        or "password" in item.lower()
    ]

    debug_findings = [
        item
        for item in findings
        if "actual print" in item
        or "actual pdb.set_trace" in item
        or "actual breakpoint" in item
    ]

    syntax_findings = [
        item
        for item in findings
        if "syntax error" in item
        or "unable to read" in item
    ]

    if credential_findings:

        print("\nCredential scan: REVIEW REQUIRED")

        for finding in credential_findings:
            print(f"- {finding}")

    else:

        print("\nCredential scan: PASSED")
        print(
            "No obvious hardcoded credential patterns found."
        )

    if syntax_findings:

        print(
            "\nSource validation: REVIEW REQUIRED"
        )

        for finding in syntax_findings:
            print(f"- {finding}")

    else:

        print(
            "\nSource syntax validation: PASSED"
        )

    if debug_findings:

        print(
            "\nDebug-output validation: REVIEW REQUIRED"
        )

        for finding in debug_findings:
            print(f"- {finding}")

    else:

        print(
            "\nDebug-output validation: PASSED"
        )

        print(
            "No actual print(), pdb.set_trace(), "
            "or breakpoint() calls found in production code."
        )


def check_expected_data_paths():

    print("\nData path validation:")

    expected_files = [
        "emotion_history.csv",
        "recommendation_history.csv",
        "recommendation_feedback.csv",
        "wellness_content.csv",
        "user_profiles.csv",
        "user_history.csv",
    ]

    outside_data = []

    data_directory_resolved = (
        DATA_DIRECTORY.resolve()
    )

    for filename in expected_files:

        path = DATA_DIRECTORY / filename

        if not path.exists():
            continue

        resolved = path.resolve()

        try:

            resolved.relative_to(
                data_directory_resolved
            )

        except ValueError:

            outside_data.append(
                str(path)
            )

    if outside_data:

        print(
            "Data path validation: FAILED"
        )

        for path in outside_data:
            print(f"- {path}")

    else:

        print(
            "Data path validation: PASSED"
        )

        print(
            "Expected project data files remain "
            "inside the data directory."
        )


if __name__ == "__main__":

    scan_source_files()

    check_expected_data_paths()

    print("\n" + "=" * 60)
    print("Security validation completed.")
    print("=" * 60)
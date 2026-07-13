from __future__ import annotations

import re


def code_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def compact_text(text: str) -> str:
    return "".join(text.split()).casefold()


def markdown_headings(notebook: dict) -> set[str]:
    headings = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for line in source.splitlines():
            match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
            if match:
                headings.add(compact_text(match.group(1)))
    return headings


def validate_notebook(notebook: dict) -> list[str]:
    code = compact_text(code_text(notebook))
    violations = []
    if "validation_data=(x_test,y_test)" in code:
        violations.append("test-data-used-for-validation")
    if "share=true" in code:
        violations.append("public-share-enabled")
    if "debug=true" in code:
        violations.append("debug-enabled")
    if "[ignoringloopdetection]" in code:
        violations.append("prompt-artifact")
    if any(
        phrase in code
        for phrase in ("絕對不報錯", "終極", "創意神經網路")
    ):
        violations.append("unsupported-hype")
    headings = markdown_headings(notebook)
    required_sections = (
        ({"problem", "問題"}, "missing-problem-section"),
        ({"method", "方法"}, "missing-method-section"),
        ({"results", "結果"}, "missing-results-section"),
        ({"limitations", "限制"}, "missing-limitations-section"),
    )
    for accepted_headings, violation_id in required_sections:
        if headings.isdisjoint(accepted_headings):
            violations.append(violation_id)
    supported_seed_calls = (
        "tf.keras.utils.set_random_seed(",
        "tf.random.set_seed(",
        "np.random.seed(",
    )
    if "model.fit(" in code and not any(
        seed_call in code for seed_call in supported_seed_calls
    ):
        violations.append("missing-deterministic-seed")
    return violations

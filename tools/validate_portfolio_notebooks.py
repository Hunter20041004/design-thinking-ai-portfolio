from __future__ import annotations

import io
import re
import tokenize


def code_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def compact_text(text: str) -> str:
    return "".join(text.split()).casefold()


def executable_tokens(source: str) -> tuple[str, ...]:
    tokens = []
    try:
        generated = tokenize.generate_tokens(io.StringIO(source).readline)
        for token_info in generated:
            if token_info.type in {tokenize.NAME, tokenize.OP}:
                tokens.append(token_info.string.casefold())
    except (IndentationError, tokenize.TokenError):
        pass
    return tuple(tokens)


def contains_token_sequence(
    tokens: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(len(tokens) - width + 1)
    )


def has_true_keyword_argument(
    tokens: tuple[str, ...], argument_name: str
) -> bool:
    target = (argument_name, "=", "true")
    return any(
        tokens[index : index + len(target)] == target
        and index > 0
        and tokens[index - 1] in {"(", ","}
        for index in range(len(tokens) - len(target) + 1)
    )


def markdown_headings(notebook: dict) -> set[str]:
    headings = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        fence_character = None
        fence_width = 0
        for line in source.splitlines():
            stripped = line.strip()
            if fence_character is not None:
                if (
                    len(stripped) >= fence_width
                    and set(stripped) == {fence_character}
                ):
                    fence_character = None
                    fence_width = 0
                continue
            fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
            if fence_match:
                marker = fence_match.group(1)
                fence_character = marker[0]
                fence_width = len(marker)
                continue
            match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
            if match:
                title = re.sub(r"\s+#+\s*$", "", match.group(1))
                headings.add(compact_text(title))
    return headings


def validate_notebook(notebook: dict) -> list[str]:
    source = code_text(notebook)
    code = compact_text(source)
    tokens = executable_tokens(source)
    violations = []
    if "validation_data=(x_test,y_test)" in code:
        violations.append("test-data-used-for-validation")
    if has_true_keyword_argument(tokens, "share"):
        violations.append("public-share-enabled")
    if has_true_keyword_argument(tokens, "debug"):
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
    model_fit_call = ("model", ".", "fit", "(")
    supported_seed_calls = (
        ("tf", ".", "keras", ".", "utils", ".", "set_random_seed", "("),
        ("tf", ".", "random", ".", "set_seed", "("),
        ("np", ".", "random", ".", "seed", "("),
    )
    if contains_token_sequence(tokens, model_fit_call) and not any(
        contains_token_sequence(tokens, seed_call)
        for seed_call in supported_seed_calls
    ):
        violations.append("missing-deterministic-seed")
    return violations

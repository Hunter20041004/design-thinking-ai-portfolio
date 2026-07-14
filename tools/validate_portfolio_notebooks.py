from __future__ import annotations

import ast
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


def markdown_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


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


def executable_trees(notebook: dict) -> tuple[ast.AST, ...]:
    trees = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source_lines = []
        for line in "".join(cell.get("source", [])).splitlines(True):
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                indentation = line[: len(line) - len(stripped)]
                source_lines.append(f"{indentation}pass\n")
            else:
                source_lines.append(line)
        try:
            trees.append(ast.parse("".join(source_lines)))
        except SyntaxError:
            continue
    return tuple(trees)


def launch_calls_use_literal_keyword(
    notebook: dict, argument_name: str, expected_value: bool
) -> bool:
    launch_calls = []
    for tree in executable_trees(notebook):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            is_launch = (
                isinstance(function, ast.Name) and function.id == "launch"
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "launch"
            )
            if is_launch:
                launch_calls.append(node)

    if not launch_calls:
        return False
    return all(
        any(
            keyword.arg == argument_name
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is expected_value
            for keyword in call.keywords
        )
        for call in launch_calls
    )


def dotted_name(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return (*parent, node.attr) if parent else ()
    return ()


def has_executable_call(notebook: dict, expected_name: tuple[str, ...]) -> bool:
    return any(
        isinstance(node, ast.Call) and dotted_name(node.func) == expected_name
        for tree in executable_trees(notebook)
        for node in ast.walk(tree)
    )


def contains_token_sequence(
    tokens: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    width = len(sequence)
    return any(
        tokens[index : index + width] == sequence
        for index in range(len(tokens) - width + 1)
    )


def launch_has_true_keyword_argument(
    tokens: tuple[str, ...], argument_name: str
) -> bool:
    target = (argument_name, "=", "true")
    for launch_index in range(len(tokens) - 1):
        if tokens[launch_index : launch_index + 2] != ("launch", "("):
            continue
        if launch_index > 0 and tokens[launch_index - 1] in {"class", "def"}:
            continue

        open_parenthesis = launch_index + 1
        nested_depth = 0
        for index in range(open_parenthesis + 1, len(tokens)):
            token = tokens[index]
            if token in {"(", "[", "{"}:
                nested_depth += 1
                continue
            if token in {")", "]", "}"}:
                if nested_depth == 0:
                    break
                nested_depth -= 1
                continue
            if (
                nested_depth == 0
                and tokens[index : index + len(target)] == target
                and (
                    index == open_parenthesis + 1
                    or tokens[index - 1] == ","
                )
            ):
                return True
    return False


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
            match = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$", line)
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
    if launch_has_true_keyword_argument(tokens, "share"):
        violations.append("public-share-enabled")
    if launch_has_true_keyword_argument(tokens, "debug"):
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

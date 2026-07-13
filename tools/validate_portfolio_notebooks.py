from __future__ import annotations


def code_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def validate_notebook(notebook: dict) -> list[str]:
    code = code_text(notebook).replace(" ", "")
    violations = []
    if "validation_data=(x_test,y_test)" in code:
        violations.append("test-data-used-for-validation")
    return violations

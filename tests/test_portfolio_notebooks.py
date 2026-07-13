import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "validate_portfolio_notebooks.py"
SPEC = importlib.util.spec_from_file_location("validate_portfolio_notebooks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def notebook_with(source: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": source.splitlines(True),
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_validator_rejects_test_data_used_as_validation():
    violations = MODULE.validate_notebook(
        notebook_with(
            "model.fit(x_train, y_train, validation_data=(x_test, y_test))\n"
        )
    )
    assert "test-data-used-for-validation" in violations

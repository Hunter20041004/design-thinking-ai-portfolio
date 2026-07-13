import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
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


def add_required_sections(notebook: dict) -> dict:
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 問題\n", "# 方法\n", "# 結果\n", "# 限制\n"],
        },
    )
    return notebook


def load_notebook(name: str) -> dict:
    return json.loads(
        (ROOT / "notebooks" / name).read_text(encoding="utf-8")
    )


def test_validator_rejects_test_data_used_as_validation():
    violations = MODULE.validate_notebook(
        notebook_with(
            "model.fit(x_train, y_train, validation_data=(x_test, y_test))\n"
        )
    )
    assert "test-data-used-for-validation" in violations


def test_validator_rejects_public_share_and_debug_defaults():
    violations = MODULE.validate_notebook(
        notebook_with("demo.launch(\n  ShArE \t= TrUe,\n  DeBuG=TRUE\n)\n")
    )
    assert "public-share-enabled" in violations
    assert "debug-enabled" in violations


def test_validator_rejects_prompt_artifacts_and_absolute_hype():
    notebook = notebook_with(
        "# [ Ignoring Loop Detection ]\n"
        "print('絕對 不報錯的終極模型：創意 神經網路')\n"
    )
    violations = MODULE.validate_notebook(notebook)
    assert "prompt-artifact" in violations
    assert "unsupported-hype" in violations


def test_validator_requires_method_results_and_limitations_sections():
    notebook = notebook_with("print('coursework evidence')\n")
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["  ##   pRoBlEm   \n"],
        },
    )

    assert MODULE.validate_notebook(notebook) == [
        "missing-method-section",
        "missing-results-section",
        "missing-limitations-section",
    ]


def test_validator_requires_a_supported_seed_for_model_training():
    unseeded = add_required_sections(
        notebook_with("MoDeL . FiT (x_train, y_train)\n")
    )
    assert MODULE.validate_notebook(unseeded) == [
        "missing-deterministic-seed"
    ]

    for seed_call in (
        "tf.keras.utils.set_random_seed(2026)",
        "tf.random.set_seed(2026)",
        "np.random.seed(2026)",
    ):
        seeded = add_required_sections(
            notebook_with(f"{seed_call}\nmodel.fit(x_train, y_train)\n")
        )
        assert "missing-deterministic-seed" not in MODULE.validate_notebook(
            seeded
        )


def test_seed_detection_uses_executable_calls_not_comments_or_strings():
    commented_seed = add_required_sections(
        notebook_with(
            "# tf.keras.utils.set_random_seed(2026)\n"
            "model.fit(x_train, y_train)\n"
        )
    )
    assert "missing-deterministic-seed" in MODULE.validate_notebook(
        commented_seed
    )

    string_seed = add_required_sections(
        notebook_with(
            "seed_example = 'np.random.seed(2026)'\n"
            "model.fit(x_train, y_train)\n"
        )
    )
    assert "missing-deterministic-seed" in MODULE.validate_notebook(
        string_seed
    )

    string_fit = add_required_sections(
        notebook_with("example = 'model.fit(x_train, y_train)'\n")
    )
    assert "missing-deterministic-seed" not in MODULE.validate_notebook(
        string_fit
    )

    for seed_call in (
        "tf.keras.utils.set_random_seed(2026)",
        "tf.random.set_seed(2026)",
        "np.random.seed(2026)",
    ):
        seeded = add_required_sections(
            notebook_with(f"{seed_call}\nmodel.fit(x_train, y_train)\n")
        )
        assert "missing-deterministic-seed" not in MODULE.validate_notebook(
            seeded
        )


def test_required_headings_ignore_backtick_and_tilde_fenced_blocks():
    for fence in ("```", "~~~"):
        notebook = notebook_with("print('coursework evidence')\n")
        notebook["cells"].insert(
            0,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Problem\n",
                    f"{fence}markdown\n",
                    "# Method\n",
                    "# Results\n",
                    "# Limitations\n",
                    f"{fence}\n",
                ],
            },
        )

        assert MODULE.validate_notebook(notebook) == [
            "missing-method-section",
            "missing-results-section",
            "missing-limitations-section",
        ]


def test_required_headings_accept_optional_closing_hashes():
    notebook = notebook_with("print('coursework evidence')\n")
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Problem #\n",
                "## Method ##\n",
                "### Results ###\n",
                "#### Limitations ####\n",
            ],
        },
    )

    assert MODULE.validate_notebook(notebook) == []


def test_public_defaults_require_exact_keyword_argument_names():
    notebook = add_required_sections(
        notebook_with("demo.launch(not_share=True, nodebug=True)\n")
    )

    violations = MODULE.validate_notebook(notebook)
    assert "public-share-enabled" not in violations
    assert "debug-enabled" not in violations


def test_required_headings_ignore_four_space_indented_code():
    notebook = notebook_with("print('coursework evidence')\n")
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Problem\n",
                "    # Method\n",
                "    # Results\n",
                "    # Limitations\n",
            ],
        },
    )

    assert MODULE.validate_notebook(notebook) == [
        "missing-method-section",
        "missing-results-section",
        "missing-limitations-section",
    ]


def test_public_defaults_apply_only_to_launch_call_arguments():
    notebook = add_required_sections(
        notebook_with(
            "def configure(share=True, debug=True):\n"
            "    return share, debug\n"
            "options = dict(share=True, debug=True)\n"
        )
    )

    violations = MODULE.validate_notebook(notebook)
    assert "public-share-enabled" not in violations
    assert "debug-enabled" not in violations


def test_dnn_notebook_meets_portfolio_contract():
    notebook = load_notebook("02-mnist-neural-network-gradio.ipynb")

    assert MODULE.validate_notebook(notebook) == []

    code = MODULE.code_text(notebook)
    compact_code = MODULE.compact_text(code)
    assert "validation_split=0.1" in compact_code or "x_val" in code
    assert code.count("tf.keras.utils.set_random_seed(2026)") == 1
    assert code.count("model.evaluate(") == 1
    assert code.count("model.evaluate(x_test, y_test") == 1
    assert code.index("model.fit(") < code.index(
        "model.evaluate(x_test, y_test"
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    assert "終極" not in notebook_text
    assert "徹底消除" not in notebook_text
    assert "完美地貼到" not in notebook_text
    assert "現在就算特別把字畫在角落" not in notebook_text

    private_metadata = {"colab", "executionInfo", "outputId"}
    assert "colab" not in notebook.get("metadata", {})
    for cell in notebook.get("cells", []):
        assert private_metadata.isdisjoint(cell.get("metadata", {}))
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []

    rewrite_script = ROOT / "tools" / "rewrite_portfolio_notebooks.py"
    rewrite_spec = importlib.util.spec_from_file_location(
        "rewrite_portfolio_notebooks", rewrite_script
    )
    rewrite_module = importlib.util.module_from_spec(rewrite_spec)
    rewrite_spec.loader.exec_module(rewrite_module)

    assert rewrite_module.rewrite_dnn(notebook) == notebook
    unrelated = notebook_with("print('not the DNN notebook')\n")
    assert rewrite_module.rewrite_dnn(unrelated) == unrelated

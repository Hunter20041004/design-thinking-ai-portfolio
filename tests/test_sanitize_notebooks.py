import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "sanitize_notebooks.py"
SPEC = importlib.util.spec_from_file_location("sanitize_notebooks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sanitize_notebook_removes_private_metadata_outputs_and_hardcoded_key():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {
                    "executionInfo": {
                        "user": {"displayName": "Student", "userId": "123"}
                    }
                },
                "outputs": [{"output_type": "stream", "text": ["private output"]}],
                "source": [
                    "import os\n",
                    "os.environ['GROQ_API_KEY'] = 'gsk_example-secret-value'\n",
                ],
            }
        ],
        "metadata": {
            "colab": {"authorship_tag": "private", "mount_file_id": "drive-id"},
            "kernelspec": {"name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    result = MODULE.sanitize_notebook(notebook)

    cell = result["cells"][0]
    assert cell["execution_count"] is None
    assert cell["outputs"] == []
    assert "executionInfo" not in cell["metadata"]
    assert "gsk_example-secret-value" not in "".join(cell["source"])
    assert "os.environ.get('GROQ_API_KEY')" in "".join(cell["source"])
    assert result["metadata"] == {"kernelspec": {"name": "python3"}}

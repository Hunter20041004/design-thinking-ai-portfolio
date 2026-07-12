"""Remove private Colab metadata, outputs, and hardcoded API credentials."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


HARDCODED_ENV_ASSIGNMENT = re.compile(
    r"^\s*os\.environ\[['\"](?P<name>[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*)['\"]\]"
    r"\s*=\s*['\"][^'\"]+['\"]\s*$"
)


def sanitize_notebook(notebook: dict) -> dict:
    cleaned = copy.deepcopy(notebook)
    cleaned["metadata"].pop("colab", None)

    for cell in cleaned.get("cells", []):
        metadata = cell.setdefault("metadata", {})
        for private_key in ("executionInfo", "outputId", "colab"):
            metadata.pop(private_key, None)

        if cell.get("cell_type") != "code":
            continue

        cell["execution_count"] = None
        cell["outputs"] = []
        sanitized_source = []
        for line in cell.get("source", []):
            match = HARDCODED_ENV_ASSIGNMENT.match(line.rstrip("\n"))
            if not match:
                sanitized_source.append(line)
                continue

            name = match.group("name")
            sanitized_source.extend(
                [
                    f"# Set {name} in the environment before running this cell.\n",
                    f"{name.lower()} = os.environ.get('{name}')\n",
                    f"if not {name.lower()}:\n",
                    f"    raise ValueError('{name} is required')\n",
                ]
            )
        cell["source"] = sanitized_source

    return cleaned


def sanitize_file(source: Path, destination: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cleaned = sanitize_notebook(notebook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize_file(args.source, args.destination)


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from difflib import SequenceMatcher
import io
import re
import tokenize


@dataclass(frozen=True)
class Comparison:
    score: float
    is_duplicate: bool


def compare_notebooks(candidate, existing):
    candidate_code = _normalized_code(candidate)
    existing_code = _normalized_code(existing)
    score = SequenceMatcher(None, candidate_code, existing_code).ratio()
    candidate_architecture = _architecture_signature(candidate_code)
    existing_architecture = _architecture_signature(existing_code)
    same_architecture = candidate_architecture == existing_architecture
    same_dataset = _dataset_signature(candidate_code) == _dataset_signature(existing_code)
    semantic_duplicate = (
        bool(candidate_architecture)
        and bool(_dataset_signature(candidate_code))
        and same_architecture
        and same_dataset
    )
    return Comparison(
        score=score,
        is_duplicate=(score >= 0.9 and same_architecture) or semantic_duplicate,
    )


def _normalized_code(notebook):
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    ignored = {
        tokenize.COMMENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return "".join(token.string for token in tokens if token.type not in ignored)


def _architecture_signature(code):
    layer_names = (
        "Conv1D",
        "Conv2D",
        "Dense",
        "Embedding",
        "Flatten",
        "GRU",
        "LSTM",
        "MaxPooling1D",
        "MaxPooling2D",
        "SimpleRNN",
    )
    pattern = rf"\b({'|'.join(layer_names)})\b"
    return frozenset(re.findall(pattern, code))


def _dataset_signature(code):
    known_datasets = ("cifar10", "fashion_mnist", "imdb", "mnist")
    lowered = code.lower()
    return frozenset(name for name in known_datasets if name in lowered)

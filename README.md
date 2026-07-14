# Design Thinking × AI Portfolio

[![CI](https://github.com/Hunter20041004/design-thinking-ai-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Hunter20041004/design-thinking-ai-portfolio/actions/workflows/ci.yml)

## Executive Summary

This repository is public learning evidence from National Chengchi University’s
Design Thinking × AI coursework and a MOOCs deep-learning course. It shows how
I moved from Python visualization to neural networks, transfer learning and
LLM orchestration while making the evaluation method, public-safety defaults
and limitations explicit.

這是課程作業的整理版，不是 production AI 產品。公開的目的，是讓讀者能檢查我的
問題拆解、方法選擇與安全邊界，而不是把課堂結果包裝成已部署或已驗證的服務。

## Coursework Portfolio

| # | Notebook | Learning evidence | Open in Colab |
| --- | --- | --- | --- |
| 01 | [Function & Math Visualization](notebooks/01-function-math-visualization.ipynb) | Python functions, NumPy and Matplotlib | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/01-function-math-visualization.ipynb) |
| 02 | [MNIST Neural Network](notebooks/02-mnist-neural-network-gradio.ipynb) | TensorFlow DNN with training-only validation and one final test evaluation | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/02-mnist-neural-network-gradio.ipynb) |
| 03 | [BTS Transfer Learning Classifier](notebooks/03-bts-transfer-learning-classifier.ipynb) | ResNet50V2 feature extraction and an honest classroom-evaluation boundary | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/03-bts-transfer-learning-classifier.ipynb) |
| 04 | [Multi-LLM Debate Arena](notebooks/04-multi-llm-debate-arena.ipynb) | Multi-role prompting, message history and a synthetic transcript format | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/04-multi-llm-debate-arena.ipynb) |
| 05 | [Reflection AI Agent](notebooks/05-reflection-ai-agent.ipynb) | Writer → Reviewer → Writer orchestration with an offline example | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/05-reflection-ai-agent.ipynb) |
| 06 | [CNN Handwritten Digit Classifier](notebooks/06-cnn-handwritten-digit-classifier.ipynb) | CNN architecture, responsibility-based cells and separated validation/test use | [Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/06-cnn-handwritten-digit-classifier.ipynb) |

The Dense MNIST exercise from the MOOCs course duplicated an existing learning
objective and was not added again. The CNN exercise remains because its model
architecture and learning objective are materially different.

## Methodology

- DNN and CNN model selection use training-only validation; the held-out test
  data is evaluated once after training.
- Training notebooks set a deterministic seed. Public copies contain no saved
  execution output, Colab identity metadata or live provider response.
- Every notebook contains Problem, Method, Results and Limitations sections.
- Gradio launches default to local-only operation. API credentials are read
  from `GROQ_API_KEY` or Colab Secrets and are never embedded in a notebook.
- Validators and deterministic rewrite functions protect these contracts from
  regression without executing paid or key-dependent services in CI.

## Results and evidence policy

The repository preserves only evidence the public files can support. It does
not invent accuracy numbers or imply that a notebook was re-run during this
portfolio cleanup. Notebook 03’s curves are classroom diagnostics, not an
independent test estimate.

**Synthetic example policy:** Notebooks 04 and 05 include clearly labelled,
author-written examples to demonstrate output shape. No API call produced
those examples, and they are not presented as Groq or AISuite model output.

## Limitations

- Notebook 03 augments images before its classroom validation split and has no
  independent test set; its results cannot establish generalization or identity.
- The BTS image dataset’s provenance, participant consent and reuse rights have
  not been verified. The dataset is not included in this repository.
- Model IDs, availability, cost and provider behavior can change. LLM outputs
  may be inconsistent, incorrect or inappropriate.
- Structural tests verify public source and notebook JSON contracts; they do
  not prove GPU compatibility, third-party uptime or live API success.

## Reproducibility

Open a notebook from the table and run it in order. Some notebooks require a
GPU, public model downloads or a credential supplied by the user. Before using
a live provider, review its current terms and data-handling policy.

Repository checks are CPU-only and do not execute notebook cells:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python tools/validate_portfolio_notebooks.py notebooks/*.ipynb
```

`tools/sanitize_notebooks.py` removes outputs, execution metadata and hard-coded
credential assignments. `tools/notebook_dedup.py` compares normalized code,
datasets and model structure to prevent duplicate portfolio entries.

## Rights

This repository does not make a blanket open-source grant over course material,
third-party assets, datasets, model artifacts or notebook contents. See
[RIGHTS.md](RIGHTS.md) before reusing anything from this repository.

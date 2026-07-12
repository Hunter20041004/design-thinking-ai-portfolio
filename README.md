# Design Thinking × AI Portfolio

國立政治大學「設計思考與人工智慧」課程的五個程式作品。這組 Notebook 從基礎視覺化與神經網路，逐步走到 transfer learning、多模型協作與帶有反思流程的 AI agent。

## Projects

| # | Notebook | 核心能力 | Demo |
|---|---|---|---|
| 01 | [Function & Math Visualization](notebooks/01-function-math-visualization.ipynb) | Python 函式、數值計算、Matplotlib 視覺化 | [Open in Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/01-function-math-visualization.ipynb) |
| 02 | [MNIST Neural Network](notebooks/02-mnist-neural-network-gradio.ipynb) | TensorFlow DNN、模型評估、Gradio 互動介面 | [Open in Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/02-mnist-neural-network-gradio.ipynb) |
| 03 | [BTS Transfer Learning Classifier](notebooks/03-bts-transfer-learning-classifier.ipynb) | ResNet50V2、特徵萃取、影像分類 | [Open in Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/03-bts-transfer-learning-classifier.ipynb) |
| 04 | [Multi-LLM Debate Arena](notebooks/04-multi-llm-debate-arena.ipynb) | 多代理角色設計、對話記憶、LLM 評審 | [Open in Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/04-multi-llm-debate-arena.ipynb) |
| 05 | [Reflection AI Agent](notebooks/05-reflection-ai-agent.ipynb) | agent workflow、reflection、工具整合、Gradio | [Open in Colab](https://colab.research.google.com/github/Hunter20041004/design-thinking-ai-portfolio/blob/main/notebooks/05-reflection-ai-agent.ipynb) |

## Learning Progression

```text
Python / visualization
        ↓
Neural network fundamentals
        ↓
Transfer learning for image classification
        ↓
Multi-LLM role orchestration
        ↓
Reflection-based AI agent workflow
```

## How to Run

最簡單的方式是點表格中的 **Open in Colab**，依序執行每一格。部分 Notebook 需要 GPU、下載公開模型／資料，或由使用者自行設定第三方 API key。

API key 請使用 Colab Secrets 或環境變數，切勿直接寫入 Notebook：

```python
import os

api_key = os.environ.get("GROQ_API_KEY")
```

## Portfolio Notes

- Notebook 是課程作業的重新整理版本；專案名稱與說明改成適合作品集閱讀的格式。
- 所有執行輸出與 Colab 個人 metadata 都已清除，避免公開個人 ID、暫時分享網址或執行紀錄。
- Repository 不包含 API key、登入憑證、模型權重或課程平台資料。
- BTS 影像分類器僅供機器學習練習；人物影像與相關商標權利屬原權利人。

## Repository Safety Check

`tools/sanitize_notebooks.py` 會在發布前移除 Notebook outputs、執行 metadata 與硬編碼的 credential assignment；對應行為由測試保護：

```bash
python -m pytest tests/test_sanitize_notebooks.py
```

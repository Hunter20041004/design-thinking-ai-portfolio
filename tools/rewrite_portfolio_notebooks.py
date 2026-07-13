"""Apply deterministic, notebook-specific portfolio corrections."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DNN_PATH = ROOT / "notebooks" / "02-mnist-neural-network-gradio.ipynb"
PRIVATE_METADATA_KEYS = ("colab", "executionInfo", "outputId")

PROBLEM = """# Problem

建立一個以 MNIST 為訓練資料的全連接神經網路，並提供本機 Gradio 手寫數字展示介面。評估流程必須保留獨立測試資料，避免把測試結果當成訓練期間的調整依據。
"""

METHOD = """# Method

- 將 MNIST 影像拉平成 784 維向量並正規化至 0–1。
- 使用含 Batch Normalization 與 Dropout 的全連接神經網路。
- 固定隨機種子為 2026，並以訓練資料的 10% 作為 validation split。
- 訓練完成後才使用保留的 `x_test`、`y_test` 做一次最終評估。
- Gradio 輸入先以邊界框裁切、縮放並置中；這是展示用的啟發式前處理。
"""

RESULTS = """# Results

執行 Notebook 時，`history` 會保存訓練與 validation 指標，模型也會對保留的 test split 做一次最終評估。公開版本不保存執行輸出，因此不宣稱特定準確率；結果應在實際執行環境中重新產生。
"""

LIMITATIONS = """# Limitations

- DNN 不具備卷積模型的平移不變性，手寫位置與筆畫粗細仍可能影響預測。
- 邊界框置中是啟發式方法，不能保證涵蓋所有輸入樣式。
- `validation_split=0.1` 取自訓練資料；test split 僅用於訓練後評估。
- Gradio 僅以 `share=False`、`debug=False` 在本機啟動，公開版本不建立外部分享連結。
"""


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(text),
    }


def is_dnn_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return (
        "from tensorflow.keras.datasets import mnist" in code
        and "def resize_image_dnn_advanced" in code
    )


def rewrite_dnn(notebook: dict) -> dict:
    """Return the corrected DNN notebook without mutating the input."""

    if not is_dnn_notebook(notebook):
        return copy.deepcopy(notebook)

    rewritten = copy.deepcopy(notebook)
    rewritten.setdefault("metadata", {}).pop("colab", None)

    cells = rewritten.get("cells", [])
    has_problem = any(
        cell.get("cell_type") == "markdown"
        and cell_text(cell).lstrip().startswith("# Problem\n")
        for cell in cells
    )
    has_results = any(
        cell.get("cell_type") == "markdown"
        and cell_text(cell).lstrip().startswith("# Results\n")
        for cell in cells
    )

    corrected_cells = []
    if not has_problem:
        corrected_cells.append(markdown_cell(PROBLEM))

    for cell in cells:
        metadata = cell.setdefault("metadata", {})
        for key in PRIVATE_METADATA_KEYS:
            metadata.pop(key, None)

        text = cell_text(cell)
        if cell.get("cell_type") == "markdown":
            if "終極優化：DNN" in text:
                cell["source"] = source_lines(METHOD)
            elif "外部套件 Gradio 說明與展示" in text:
                cell["source"] = source_lines(LIMITATIONS)
            corrected_cells.append(cell)
            continue

        cell["execution_count"] = None
        cell["outputs"] = []
        text = text.replace(
            "把縮放後的字跡完美地貼到",
            "把縮放後的字跡貼到",
            1,
        )

        if (
            "from tensorflow.keras.datasets import mnist" in text
            and "tf.keras.utils.set_random_seed(2026)" not in text
        ):
            import_anchor = "import gradio as gr\n"
            text = text.replace(
                import_anchor,
                import_anchor
                + "\n# 讓模型初始化與資料切分可重現\n"
                + "tf.keras.utils.set_random_seed(2026)\n",
                1,
            )

        if "history = model.fit(" in text:
            text = text.replace(
                "history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_test, y_test))\n",
                "history = model.fit(\n"
                "    x_train,\n"
                "    y_train,\n"
                "    batch_size=128,\n"
                "    epochs=15,\n"
                "    validation_split=0.1,\n"
                ")\n",
                1,
            )
            text = text.replace(
                'print("模型訓練大功告成！")\n',
                'print("模型訓練完成。")\n',
                1,
            )
            text = text.replace(
                "test_loss, test_accuracy = model.evaluate(\n"
                "    x_test, y_test, verbose=0\n"
                ")\n",
                "test_loss, test_accuracy = model.evaluate("
                "x_test, y_test, verbose=0)\n",
                1,
            )
            compact_text = "".join(text.split())
            if "model.evaluate(x_test,y_test" not in compact_text:
                text += (
                    "\n# 訓練與模型選擇完成後，只評估保留的 test split 一次\n"
                    "test_loss, test_accuracy = model.evaluate("
                    "x_test, y_test, verbose=0)\n"
                )

        if "iface.launch(" in text:
            text = text.replace(
                "iface.launch(share=True, debug=True)",
                "iface.launch(share=False, debug=False)",
                1,
            )
            text = text.replace(
                "我透過『邊界框自動對齊 (Bounding Box)』技術消除誤差。",
                "我使用『邊界框自動對齊 (Bounding Box)』協助減少位置偏差。",
                1,
            )
            text = text.replace(
                "我使用『邊界框自動對齊 (Bounding Box)』協助減少位置偏差。"
                "現在就算特別把字畫在角落，它也能幫我抓回中間給模型測！"
                "(但還是請盡量稍微畫粗一點喔！)",
                "請在畫板輸入一個數字；前處理會裁切非空白區域、"
                "縮放並置中，結果仍會受筆畫與位置影響。",
                1,
            )

        cell["source"] = source_lines(text)
        corrected_cells.append(cell)
        if "history = model.fit(" in text and not has_results:
            corrected_cells.append(markdown_cell(RESULTS))
            has_results = True

    rewritten["cells"] = corrected_cells
    return rewritten


def rewrite_file(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    rewritten = rewrite_dnn(notebook)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    rewrite_file(DNN_PATH)

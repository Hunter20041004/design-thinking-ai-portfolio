"""Apply deterministic, notebook-specific portfolio corrections."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DNN_PATH = ROOT / "notebooks" / "02-mnist-neural-network-gradio.ipynb"
CNN_PATH = ROOT / "notebooks" / "06-cnn-handwritten-digit-classifier.ipynb"
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

CNN_PROBLEM = """# Problem

使用 MNIST 建立卷積神經網路進行手寫數字分類，同時讓 validation 只取自訓練資料，保留 test split 到訓練完成後才評估。
"""

CNN_METHOD = """# Method

- 影像維持 28×28 灰階格式並正規化至 0–1。
- 模型保留原本的兩個 Conv2D、Batch Normalization、Max Pooling、Dropout 與 Dense 分類層。
- 固定隨機種子為 2026，並以訓練資料的 10% 作為 validation split。
- 模型訓練與選擇完成後，才使用保留的 test split 做一次最終評估。
"""

CNN_RESULTS = """# Results

執行 Notebook 時會產生訓練與 validation 指標，並在訓練完成後評估保留的 test split。公開版本不保存執行輸出，因此不宣稱特定準確率；結果需在實際執行環境中重新產生。
"""

CNN_LIMITATIONS = """# Limitations

- MNIST 與真實手寫輸入的分布可能不同，測試結果不能直接代表部署情境。
- `validation_split=0.1` 取自訓練資料；test split 不參與訓練期間的模型調整。
- 範例預測固定顯示 test split 的第 100 張影像，僅用於說明推論流程。
"""

CNN_DATA_STAGE = """## Data preparation

載入 MNIST，保留原本的 reshape、正規化與 one-hot encoding 步驟。
"""

CNN_MODEL_STAGE = """## Model definition

保留原始 CNN 架構與 Adam 設定，只將程式拆成較容易閱讀的責任區塊。
"""

CNN_TRAINING_STAGE = """## Training

以 training-only validation split 訓練；本階段不讀取 test split 作為 validation data。
"""

CNN_DEMO_STAGE = """## Prediction demo

使用已完成訓練的模型預測 test split，並顯示固定索引的影像作為流程示例。
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


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def is_dnn_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def is_cnn_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return "model.add(Conv2D(32" in code and "def my_predict(n):" in code
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


def rewrite_cnn(notebook: dict) -> dict:
    """Return the corrected CNN notebook without mutating the input."""

    if not is_cnn_notebook(notebook):
        return copy.deepcopy(notebook)

    rewritten = copy.deepcopy(notebook)
    rewritten.setdefault("metadata", {}).pop("colab", None)
    code = "\n".join(
        cell_text(cell).strip()
        for cell in rewritten.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    code += "\n"

    replacements = (
        ("# [ignoring loop detection]\n", ""),
        ("# 2. 建立我的創意神經網路", "# 2. 建立 CNN 模型"),
        (
            "# --- 創意點 1：BatchNormalization (批次標準化) ---",
            "# --- BatchNormalization (批次標準化) ---",
        ),
        (
            "# 就像給神經網路裝上穩定器，讓每一層的數據分佈更均勻，大幅提升訓練速度。",
            "# 協助穩定每一層的特徵分布。",
        ),
        (
            "# --- 創意點 2：Dropout (隨機失活) ---",
            "# --- Dropout (隨機失活) ---",
        ),
        (
            "# 隨機讓部分神經元休息，防止模型「死背」題目，這能顯著提升對新資料的辨識準確度。",
            "# 隨機停用部分神經元，以降低過擬合風險。",
        ),
        (
            "# 全連接層 (增加神經元數量提升理解能力)",
            "# 全連接分類層",
        ),
        (
            "# 3. 組裝模型 (創意點 3：使用 Adam 優化器與分類專用損失函數)",
            "# 3. 組裝模型 (Adam 優化器與分類損失函數)",
        ),
        (
            "# 4. 開始訓練 (設定較大的 batch_size=256 以利用 GPU 進行超速訓練)",
            "# 4. 開始訓練 (batch_size=256)",
        ),
        (
            "          validation_data=(x_test, y_test))",
            "          validation_split=0.1)",
        ),
        ("測試正確率達標：", "測試正確率："),
    )
    for old, new in replacements:
        code = code.replace(old, new, 1)

    data_anchor = "# 1. 數據準備 (MNIST 數字辨識)"
    model_anchor = "# 2. 建立 CNN 模型"
    training_anchor = "# 4. 開始訓練 (batch_size=256)"
    evaluation_anchor = "# 5. 評估與預測結果"
    demo_anchor = "# 預測介面"

    if "tf.keras.utils.set_random_seed(2026)" not in code:
        code = code.replace(
            data_anchor,
            "# 讓模型初始化與資料切分可重現\n"
            "tf.keras.utils.set_random_seed(2026)\n\n"
            + data_anchor,
            1,
        )

    data_index = code.index(data_anchor)
    model_index = code.index(model_anchor)
    training_index = code.index(training_anchor)
    evaluation_index = code.index(evaluation_anchor)
    demo_index = code.index(demo_anchor)

    imports = code[:data_index].strip() + "\n"
    data = code[data_index:model_index].strip() + "\n"
    model = code[model_index:training_index].strip() + "\n"
    training = code[training_index:evaluation_index].strip() + "\n"
    evaluation = code[evaluation_index:demo_index].strip() + "\n"
    demo = code[demo_index:].strip() + "\n"

    rewritten["cells"] = [
        markdown_cell(CNN_PROBLEM),
        markdown_cell(CNN_METHOD),
        code_cell(imports),
        markdown_cell(CNN_DATA_STAGE),
        code_cell(data),
        markdown_cell(CNN_MODEL_STAGE),
        code_cell(model),
        markdown_cell(CNN_TRAINING_STAGE),
        code_cell(training),
        markdown_cell(CNN_RESULTS),
        code_cell(evaluation),
        markdown_cell(CNN_DEMO_STAGE),
        code_cell(demo),
        markdown_cell(CNN_LIMITATIONS),
    ]
    return rewritten


def rewrite_file(path: Path, rewrite) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    rewritten = rewrite(notebook)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    rewrite_file(DNN_PATH, rewrite_dnn)
    rewrite_file(CNN_PATH, rewrite_cnn)

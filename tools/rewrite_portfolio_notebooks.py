"""Apply deterministic, notebook-specific portfolio corrections."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DNN_PATH = ROOT / "notebooks" / "02-mnist-neural-network-gradio.ipynb"
CNN_PATH = ROOT / "notebooks" / "06-cnn-handwritten-digit-classifier.ipynb"
MATH_PATH = ROOT / "notebooks" / "01-function-math-visualization.ipynb"
BTS_PATH = ROOT / "notebooks" / "03-bts-transfer-learning-classifier.ipynb"
DEBATE_PATH = ROOT / "notebooks" / "04-multi-llm-debate-arena.ipynb"
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

MATH_PROBLEM = """# Problem

以 NumPy 產生函數與參數曲線的座標，再用 Matplotlib 將正弦函數與心形方程式視覺化。
"""

MATH_METHOD = """# Method

- 使用 `np.linspace` 建立連續取樣點。
- 計算正弦函數與心形參數方程式的座標。
- 以 Matplotlib 繪製座標軸、曲線、填色與手工字樣。
- 本 Notebook 沒有模型訓練流程，因此不需要隨機種子。
"""

MATH_RESULTS = """# Results

執行 Notebook 會產生正弦函數圖與心形參數曲線。公開版本不保存渲染輸出；圖形需在實際 Notebook 環境中重新產生。
"""

MATH_LIMITATIONS = """# Limitations

- 圖形外觀會受 Matplotlib 版本、字型與顯示後端影響。
- 曲線是固定範圍內的離散取樣，不代表符號運算證明。
- 心形中的字樣使用手工座標，僅示範基本繪圖組合。
"""

BTS_METHOD = """# Method

- 使用 ImageNet 預訓練的 ResNet50V2 作為固定特徵萃取器，再訓練 Dense 分類層。
- 現有課堂流程先將原圖與水平翻轉圖合併，再用 `validation_split=0.2` 切分；因此沒有獨立 test split，validation 也可能包含同一來源影像的變形版本。
- 若要做正式評估，應先切分原始照片，再只增強 training split，使用 validation split 做模型選擇，最後只評估一次未參與調整的 test split。
- 固定 TensorFlow 隨機種子為 2026，讓分類層初始化與資料切分較容易重現。
"""

BTS_RESULTS = """# Results

## Training curves

執行 Notebook 時會顯示 training 與 validation 曲線。公開版本不保存輸出，也沒有獨立測試集，因此不宣稱 final test 表現；曲線僅用於觀察課堂流程是否出現明顯過擬合。
"""

BTS_LIMITATIONS = """# Limitations

- 目前資料集規模小，且 augmentation 後才切 validation，不能視為獨立泛化評估。
- 影像著作權、授權與當事人同意尚未驗證；請勿重新散布資料集，並只使用有權使用的照片。
- 模型只在七個預設標籤間分類，不能作為身分驗證工具，也不適合用於高風險決策。
- 雲端檔案 ID 與執行環境可能失效，使用者需自行提供合法且結構相同的資料。
"""

DEBATE_PROBLEM = """# Problem

## AI 辯論擂台 — 用 AISuite 組合多個 LLM 角色

這個課堂練習將正方、反方與評審拆成三個提示角色，觀察多回合訊息如何在角色之間傳遞。實際文字由 Groq 模型產生，需要使用者自行提供 API key，且輸出品質與可用模型會隨供應商改變。
"""

DEBATE_METHOD = """# Method

- `run_debate` 保存正反方訊息歷史，再把完整紀錄交給評審角色。
- API key 只從 `GROQ_API_KEY` 環境變數或 Colab Secrets 取得，不寫入 Notebook。
- 公開範例使用手寫 synthetic transcript 測試顯示格式；沒有呼叫任何 API。
- live Groq 輸出只會在使用者主動操作介面後產生，且可能產生錯誤、不一致或不當內容。
"""

DEBATE_RESULTS = """# Results

下方 transcript 是 **SYNTHETIC EXAMPLE**，由作者手寫來示範正方、反方與評審的輸出格式，沒有呼叫任何 API，也不代表指定模型的真實回答。公開版本不保存 live provider output。
"""

DEBATE_LIMITATIONS = """# Limitations

- 模型可能被供應商更名、淘汰或下架；執行前必須查核 Groq 當下可用的 model ID。
- LLM 回答具有隨機性，評審分數不是客觀事實，也不能取代人類判斷。
- 使用 live API 可能產生費用、速率限制或資料處理風險；請先閱讀供應商政策。
- Synthetic example 只驗證流程呈現，不證明 AISuite 或 Groq 連線成功。
"""

DEBATE_SYNTHETIC_CODE = '''# SYNTHETIC EXAMPLE — 作者手寫，沒有呼叫任何 API
synthetic_example = """SYNTHETIC EXAMPLE
議題：小學生應該要有薪水
正方：零用制度可作為責任教育的討論起點。
反方：報酬可能改變學習與家庭分工的意義。
評審：雙方各提出一項價值衝突；此段僅示範格式，不宣告勝負。
"""
print(synthetic_example)
'''


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
    return (
        "from tensorflow.keras.datasets import mnist" in code
        and "def resize_image_dnn_advanced" in code
    )


def is_cnn_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return "model.add(Conv2D(32" in code and "def my_predict(n):" in code


def is_math_visualization_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return (
        "y = np.sin(x)" in code
        and "Mathematical Heart: I LOVE YOU" in code
    )


def is_bts_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return (
        'category_en = "RM,Jin,SUGA,JHope,Jimin,V,Jungkook"' in code
        and "ResNet50V2(include_top=False" in code
    )


def is_debate_notebook(notebook: dict) -> bool:
    code = "\n".join(
        cell_text(cell)
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return "def run_debate(topic, rounds=2):" in code and "model_judge" in code


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


def rewrite_math_visualization(notebook: dict) -> dict:
    """Return the structured math notebook without changing its code."""

    if not is_math_visualization_notebook(notebook):
        return copy.deepcopy(notebook)

    rewritten = copy.deepcopy(notebook)
    rewritten.setdefault("metadata", {}).pop("colab", None)
    cells = rewritten.get("cells", [])
    for cell in cells:
        metadata = cell.setdefault("metadata", {})
        for key in PRIVATE_METADATA_KEYS:
            metadata.pop(key, None)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    headings = {
        line.strip().casefold()
        for cell in cells
        if cell.get("cell_type") == "markdown"
        for line in cell_text(cell).splitlines()
        if line.lstrip().startswith("#")
    }
    if "# problem" not in headings:
        if cells and cells[0].get("cell_type") == "markdown" and not cell_text(
            cells[0]
        ).strip():
            cells[0]["source"] = source_lines(MATH_PROBLEM)
        else:
            cells.insert(0, markdown_cell(MATH_PROBLEM))
    if "# method" not in headings:
        cells.insert(1, markdown_cell(MATH_METHOD))
    if "# results" not in headings:
        cells.append(markdown_cell(MATH_RESULTS))
    if "# limitations" not in headings:
        cells.append(markdown_cell(MATH_LIMITATIONS))

    rewritten["cells"] = cells
    return rewritten


def rewrite_bts(notebook: dict) -> dict:
    """Return the documented transfer-learning notebook."""

    if not is_bts_notebook(notebook):
        return copy.deepcopy(notebook)

    rewritten = copy.deepcopy(notebook)
    rewritten.setdefault("metadata", {}).pop("colab", None)
    cells = rewritten.get("cells", [])
    headings = {
        line.strip().casefold()
        for cell in cells
        if cell.get("cell_type") == "markdown"
        for line in cell_text(cell).splitlines()
        if line.lstrip().startswith("#")
    }

    for cell in cells:
        metadata = cell.setdefault("metadata", {})
        for key in PRIVATE_METADATA_KEYS:
            metadata.pop(key, None)
        text = cell_text(cell)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            if (
                "import tensorflow as tf" in text
                and "tf.keras.utils.set_random_seed(2026)" not in text
            ):
                text = text.replace(
                    "import tensorflow as tf\n",
                    "import tensorflow as tf\n"
                    "tf.keras.utils.set_random_seed(2026)\n",
                    1,
                )
            replacements = (
                (
                    'description = "上傳一張 BTS 成員的照片，我會告訴你他是誰！支援全部 7 位成員辨識"',
                    'description = "上傳照片後，模型會回傳七個課堂標籤的信心分數；結果不能作為身分驗證。"',
                ),
                (
                    "# 使用標準學習率，讓收斂快一點",
                    "# 使用 Adam 優化器與明確的 learning rate",
                ),
                (
                    "# share=True 會產生一個公開連結，方便分享",
                    "# 公開版本只在本機啟動，不建立分享連結",
                ),
                ("demo.launch(share=True)", "demo.launch(share=False)"),
            )
            for old, new in replacements:
                text = text.replace(old, new, 1)
            cell["source"] = source_lines(text)
            continue

        if not cell_text(cell).strip() and cell is not cells[0]:
            continue
        if text.startswith("# 作業二：遷移式學習做 BTS 成員辨識器"):
            text = text.replace(
                "# 作業二：遷移式學習做 BTS 成員辨識器",
                "# Problem\n\n## 作業二：遷移式學習做 BTS 成員辨識器",
                1,
            )
        replacements = (
            (
                "即使每人只有 ~10 張照片，也能有不錯的辨識效果",
                "少量照片容易造成過擬合，validation 曲線只能作為課堂練習的診斷",
            ),
            (
                "因為只訓練最後的分類層（ResNet 的參數是凍結的），所以訓練非常快！",
                "本 Notebook 只訓練最後的分類層，ResNet 特徵萃取器維持固定。",
            ),
            (
                "上傳任意一張 BTS 成員照片，模型會告訴你他是誰",
                "上傳照片後，模型會回傳七個課堂標籤的信心分數，不能作為身分驗證工具。",
            ),
        )
        for old, new in replacements:
            text = text.replace(old, new, 1)
        if "### 8. 訓練曲線" in text and "# results" not in headings:
            text = BTS_RESULTS
        cell["source"] = source_lines(text)

    if "# method" not in headings:
        cells.insert(1, markdown_cell(BTS_METHOD))
    if "# limitations" not in headings:
        cells.append(markdown_cell(BTS_LIMITATIONS))

    rewritten["cells"] = cells
    return rewritten


def rewrite_debate(notebook: dict) -> dict:
    """Return the safe, explicitly synthetic debate notebook."""

    if not is_debate_notebook(notebook):
        return copy.deepcopy(notebook)

    rewritten = copy.deepcopy(notebook)
    rewritten.setdefault("metadata", {}).pop("colab", None)
    cells = rewritten.get("cells", [])
    headings = {
        line.strip().casefold()
        for cell in cells
        if cell.get("cell_type") == "markdown"
        for line in cell_text(cell).splitlines()
        if line.lstrip().startswith("#")
    }

    for cell in cells:
        metadata = cell.setdefault("metadata", {})
        for key in PRIVATE_METADATA_KEYS:
            metadata.pop(key, None)
        text = cell_text(cell)
        if cell.get("cell_type") != "code":
            if cell is cells[0] and "# problem" not in headings:
                cell["source"] = source_lines(DEBATE_PROBLEM)
            continue

        cell["execution_count"] = None
        cell["outputs"] = []
        key_block = (
            "# 直接寫入你的 Groq API Key（防呆專用）\n"
            "# Set GROQ_API_KEY in the environment before running this cell.\n"
            "groq_api_key = os.environ.get('GROQ_API_KEY')\n"
            "if not groq_api_key:\n"
            "    raise ValueError('GROQ_API_KEY is required')\n"
        )
        safe_key_block = (
            "# 優先讀取環境變數；在 Colab 中可改用 Secrets。\n"
            "groq_api_key = os.environ.get('GROQ_API_KEY')\n"
            "if not groq_api_key:\n"
            "    try:\n"
            "        from google.colab import userdata\n"
            "        groq_api_key = userdata.get('GROQ_API_KEY')\n"
            "    except Exception:\n"
            "        groq_api_key = None\n"
            "if not groq_api_key:\n"
            "    raise ValueError('Set GROQ_API_KEY in the environment or Colab Secrets')\n"
            "os.environ['GROQ_API_KEY'] = groq_api_key\n"
        )
        text = text.replace(key_block, safe_key_block, 1)
        text = text.replace(
            "# 為了保證絕對不報錯，全部換成 Groq 目前最穩定、不會下架的模型",
            "# Model ID 可能變動；執行前請查核 Groq 當下的模型目錄。",
            1,
        )
        if "result = run_debate(" in text:
            text = DEBATE_SYNTHETIC_CODE
        text = text.replace(
            "讓三個頂尖開源 AI 幫你吵架兼做裁判！",
            "讓三個提示角色產生辯論與評審文字。",
            1,
        )
        text = text.replace(
            "demo.launch(share=True, debug=True)",
            "demo.launch(share=False, debug=False)",
            1,
        )
        cell["source"] = source_lines(text)

    if "# method" not in headings:
        cells.insert(1, markdown_cell(DEBATE_METHOD))
    if "# results" not in headings:
        synthetic_index = next(
            index
            for index, cell in enumerate(cells)
            if "synthetic_example" in cell_text(cell)
        )
        cells.insert(synthetic_index, markdown_cell(DEBATE_RESULTS))
    if "# limitations" not in headings:
        cells.append(markdown_cell(DEBATE_LIMITATIONS))

    rewritten["cells"] = cells
    return rewritten


def rewrite_file(path: Path, rewrite) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    rewritten = rewrite(notebook)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    rewrite_file(MATH_PATH, rewrite_math_visualization)
    rewrite_file(DNN_PATH, rewrite_dnn)
    rewrite_file(BTS_PATH, rewrite_bts)
    rewrite_file(DEBATE_PATH, rewrite_debate)
    rewrite_file(CNN_PATH, rewrite_cnn)

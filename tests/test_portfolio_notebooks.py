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


def test_safe_launch_contract_uses_executable_literal_arguments():
    hostile = add_required_sections(
        notebook_with(
            "demo.launch(share=bool(1), debug=bool(1))\n"
            "decoy = 'demo.launch(share=False, debug=False)'\n"
        )
    )

    assert not MODULE.launch_calls_use_literal_keyword(
        hostile, "share", False
    )
    assert not MODULE.launch_calls_use_literal_keyword(
        hostile, "debug", False
    )


def test_credential_contract_uses_executable_calls_not_string_decoys():
    hostile = notebook_with(
        "groq_api_key = input('Paste Groq API key')\n"
        "decoy = \"os.environ.get('GROQ_API_KEY'); "
        "userdata.get('GROQ_API_KEY')\"\n"
    )

    assert MODULE.has_executable_call(hostile, ("input",))
    assert not MODULE.has_executable_call(
        hostile, ("os", "environ", "get")
    )
    assert not MODULE.has_executable_call(hostile, ("userdata", "get"))


def test_methodology_disclosures_must_be_markdown_not_code_decoys():
    hostile = add_required_sections(
        notebook_with(
            "decoy = '沒有獨立 test split；影像著作權、授權與當事人同意尚未驗證'\n"
        )
    )

    markdown = MODULE.markdown_text(hostile)
    assert "沒有獨立 test split" not in markdown
    assert "影像著作權、授權與當事人同意尚未驗證" not in markdown


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

    original = json.loads(json.dumps(notebook))
    setup, method, training, limitations, demo = [
        original["cells"][index] for index in (1, 2, 3, 5, 6)
    ]
    setup_text = "".join(setup["source"]).replace(
        "# 讓模型初始化與資料切分可重現\n"
        "tf.keras.utils.set_random_seed(2026)\n\n",
        "",
        1,
    )
    setup["source"] = setup_text.splitlines(keepends=True)
    method["source"] = ["### 💡 終極優化：DNN\n"]

    training_text = "".join(training["source"])
    training_text = training_text.replace(
        "history = model.fit(\n"
        "    x_train,\n"
        "    y_train,\n"
        "    batch_size=128,\n"
        "    epochs=15,\n"
        "    validation_split=0.1,\n"
        ")\n",
        "history = model.fit(x_train, y_train, batch_size=128, "
        "epochs=15, validation_data=(x_test, y_test))\n",
        1,
    )
    training_text = training_text.replace(
        'print("模型訓練完成。")\n',
        'print("模型訓練大功告成！")\n',
        1,
    )
    training_text = training_text.replace(
        "\n# 訓練與模型選擇完成後，只評估保留的 test split 一次\n"
        "test_loss, test_accuracy = model.evaluate("
        "x_test, y_test, verbose=0)\n",
        "",
        1,
    )
    training["source"] = training_text.splitlines(keepends=True)

    limitations["source"] = ["### 📦 外部套件 Gradio 說明與展示\n"]
    demo_text = "".join(demo["source"])
    demo_text = demo_text.replace(
        "把縮放後的字跡貼到",
        "把縮放後的字跡完美地貼到",
        1,
    )
    demo_text = demo_text.replace(
        "請在畫板輸入一個數字；前處理會裁切非空白區域、"
        "縮放並置中，結果仍會受筆畫與位置影響。",
        "我透過『邊界框自動對齊 (Bounding Box)』技術消除誤差。"
        "現在就算特別把字畫在角落，它也能幫我抓回中間給模型測！"
        "(但還是請盡量稍微畫粗一點喔！)",
        1,
    )
    demo_text = demo_text.replace(
        "iface.launch(share=False, debug=False)",
        "iface.launch(share=True, debug=True)",
        1,
    )
    demo["source"] = demo_text.splitlines(keepends=True)
    original["cells"] = [setup, method, training, limitations, demo]

    assert rewrite_module.is_dnn_notebook(original) is True
    assert rewrite_module.rewrite_dnn(original) == notebook
    assert rewrite_module.is_dnn_notebook(notebook) is True
    assert rewrite_module.rewrite_dnn(notebook) == notebook
    unrelated = notebook_with("print('not the DNN notebook')\n")
    assert rewrite_module.rewrite_dnn(unrelated) == unrelated


def test_cnn_notebook_meets_portfolio_contract():
    notebook = load_notebook("06-cnn-handwritten-digit-classifier.ipynb")

    assert MODULE.validate_notebook(notebook) == []

    code_cells = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert len(code_cells) >= 6
    headings = MODULE.markdown_headings(notebook)
    assert {"problem", "method", "results", "limitations"} <= headings

    code = MODULE.code_text(notebook)
    compact_code = MODULE.compact_text(code)
    assert "validation_split=0.1" in compact_code or "x_val" in code
    assert code.count("tf.keras.utils.set_random_seed(2026)") == 1
    assert code.count("model.evaluate(") == 1
    assert code.count("model.evaluate(x_test, y_test") == 1
    assert code.index("model.fit(") < code.index(
        "model.evaluate(x_test, y_test"
    )

    preserved_code = (
        "x_train = x_train.reshape(60000, 28, 28, 1) / 255",
        "x_test = x_test.reshape(10000, 28, 28, 1) / 255",
        "model.add(Conv2D(32, (3,3), padding='same', "
        "input_shape=(28,28,1), activation='relu'))",
        "model.add(Conv2D(64, (3,3), padding='same', "
        "activation='relu'))",
        "model.add(Dropout(0.25))",
        "model.add(Dense(128, activation='relu'))",
        "model.add(Dropout(0.5))",
        "model.add(Dense(10, activation='softmax'))",
        "optimizer=Adam(learning_rate=0.001)",
    )
    for statement in preserved_code:
        assert code.count(statement) == 1

    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    for phrase in (
        "[ignoring loop detection]",
        "創意神經網路",
        "大幅提升訓練速度",
        "顯著提升",
        "超速訓練",
        "測試正確率達標",
    ):
        assert phrase not in notebook_text

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

    assert rewrite_module.rewrite_cnn(notebook) == notebook
    unrelated = notebook_with("print('not the CNN notebook')\n")
    assert rewrite_module.rewrite_cnn(unrelated) == unrelated


def test_math_visualization_notebook_meets_portfolio_contract():
    notebook = load_notebook("01-function-math-visualization.ipynb")

    assert MODULE.validate_notebook(notebook) == []
    headings = MODULE.markdown_headings(notebook)
    assert {"problem", "method", "results", "limitations"} <= headings

    code = MODULE.code_text(notebook)
    assert "model.fit(" not in code
    assert "set_random_seed(" not in code
    preserved_code = (
        "x = np.linspace(-10, 10, 400)",
        "y = np.sin(x)",
        "x = 16 * np.sin(t)**3",
        "def plot_letter(ax, ay, color='red'):",
        "plt.title('Mathematical Heart: I LOVE YOU'",
    )
    for statement in preserved_code:
        assert code.count(statement) == 1

    private_metadata = {"colab", "executionInfo", "outputId"}
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

    assert rewrite_module.is_math_visualization_notebook(notebook) is True
    assert rewrite_module.rewrite_math_visualization(notebook) == notebook
    unrelated = notebook_with("print('not the math notebook')\n")
    assert rewrite_module.rewrite_math_visualization(unrelated) == unrelated


def test_bts_transfer_learning_notebook_meets_portfolio_contract():
    notebook = load_notebook("03-bts-transfer-learning-classifier.ipynb")

    assert MODULE.validate_notebook(notebook) == []
    headings = MODULE.markdown_headings(notebook)
    assert {"problem", "method", "results", "limitations"} <= headings

    code = MODULE.code_text(notebook)
    assert code.count("tf.keras.utils.set_random_seed(2026)") == 1
    assert "validation_split=0.2" in MODULE.compact_text(code)
    assert "model.evaluate(" not in code
    assert MODULE.launch_calls_use_literal_keyword(notebook, "share", False)
    preserved_code = (
        "ResNet50V2(include_top=False, pooling='avg', weights='imagenet')",
        "features = resnet.predict(data_preprocessed, verbose=1)",
        "layers.Dense(512, input_dim=features.shape[1], activation='relu')",
        "layers.Dense(N, activation='softmax')",
        "epochs=100",
    )
    for statement in preserved_code:
        assert code.count(statement) == 1

    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    markdown = MODULE.markdown_text(notebook)
    required_disclosures = (
        "沒有獨立 test split",
        "先切分原始照片，再只增強 training split",
        "影像著作權、授權與當事人同意尚未驗證",
        "不能作為身分驗證工具",
        "不宣稱 final test 表現",
    )
    for disclosure in required_disclosures:
        assert disclosure in markdown
    for claim in (
        "即使每人只有 ~10 張照片，也能有不錯的辨識效果",
        "所以訓練非常快",
        "模型會告訴你他是誰",
    ):
        assert claim not in notebook_text

    private_metadata = {"colab", "executionInfo", "outputId"}
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

    assert rewrite_module.is_bts_notebook(notebook) is True
    assert rewrite_module.rewrite_bts(notebook) == notebook
    unrelated = notebook_with("print('not the transfer-learning notebook')\n")
    assert rewrite_module.rewrite_bts(unrelated) == unrelated


def test_debate_arena_notebook_meets_portfolio_contract():
    notebook = load_notebook("04-multi-llm-debate-arena.ipynb")

    assert MODULE.validate_notebook(notebook) == []
    headings = MODULE.markdown_headings(notebook)
    assert {"problem", "method", "results", "limitations"} <= headings

    code = MODULE.code_text(notebook)
    assert MODULE.launch_calls_use_literal_keyword(notebook, "share", False)
    assert MODULE.launch_calls_use_literal_keyword(notebook, "debug", False)
    assert MODULE.has_executable_call(notebook, ("os", "environ", "get"))
    assert "from google.colab import userdata" in code
    assert MODULE.has_executable_call(notebook, ("userdata", "get"))
    assert not MODULE.has_executable_call(notebook, ("input",))
    assert "result = run_debate(" not in code
    preserved_code = (
        'model_pro = "groq:llama-3.3-70b-versatile"',
        'model_con = "groq:llama-3.1-8b-instant"',
        "def get_response(model, messages):",
        "def run_debate(topic, rounds=2):",
        "start_btn.click(fn=debate_interface",
    )
    for statement in preserved_code:
        assert code.count(statement) == 1

    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    markdown = MODULE.markdown_text(notebook)
    assert "SYNTHETIC EXAMPLE" in markdown
    assert "沒有呼叫任何 API" in markdown
    assert "模型可能被供應商更名、淘汰或下架" in markdown
    assert "live Groq 輸出只會在使用者主動操作介面後產生" in markdown
    assert "synthetic_example =" in code
    for claim in ("不會下架", "絕對不報錯", "零設定防呆版"):
        assert claim not in notebook_text

    private_metadata = {"colab", "executionInfo", "outputId"}
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

    assert rewrite_module.is_debate_notebook(notebook) is True
    assert rewrite_module.rewrite_debate(notebook) == notebook
    unrelated = notebook_with("print('not the debate notebook')\n")
    assert rewrite_module.rewrite_debate(unrelated) == unrelated


def test_reflection_agent_notebook_meets_portfolio_contract():
    notebook = load_notebook("05-reflection-ai-agent.ipynb")

    assert MODULE.validate_notebook(notebook) == []
    headings = MODULE.markdown_headings(notebook)
    assert {"problem", "method", "results", "limitations"} <= headings

    code = MODULE.code_text(notebook)
    assert MODULE.launch_calls_use_literal_keyword(notebook, "share", False)
    assert MODULE.launch_calls_use_literal_keyword(notebook, "debug", False)
    assert MODULE.has_executable_call(notebook, ("os", "environ", "get"))
    assert "from google.colab import userdata" in code
    assert MODULE.has_executable_call(notebook, ("userdata", "get"))
    assert not MODULE.has_executable_call(notebook, ("input",))
    preserved_code = (
        "def reply(system=",
        "response = client.chat.completions.create(",
        "def trailer_reflect(prompt):",
        "first_version = reply(system_writer,",
        "suggestion = reply(system_reviewer, first_version,",
        "second_version = reply(system_writer, second_prompt,",
        "btn.click(trailer_reflect, inputs=[user_input], outputs=[out1, out2, out3])",
    )
    for statement in preserved_code:
        assert code.count(statement) == 1

    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )
    markdown = MODULE.markdown_text(notebook)
    assert "SYNTHETIC EXAMPLE" in markdown
    assert "沒有呼叫任何 API" in markdown
    assert "只驗證 orchestration 的資料流與呈現格式" in markdown
    assert "live provider output 只會在使用者主動操作介面後產生" in markdown
    assert "模型可能被供應商更名、淘汰或下架" in markdown
    assert "synthetic_reflection =" in code
    for claim in ("頂級", "史詩級", "一定要有一個吊人胃口"):
        assert claim not in notebook_text

    private_metadata = {"colab", "executionInfo", "outputId"}
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

    assert rewrite_module.is_reflection_notebook(notebook) is True
    assert rewrite_module.rewrite_reflection(notebook) == notebook
    unrelated = notebook_with("print('not the reflection notebook')\n")
    assert rewrite_module.rewrite_reflection(unrelated) == unrelated


def test_all_notebooks_meet_portfolio_contract():
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))

    assert notebook_paths
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert MODULE.validate_notebook(notebook) == [], path.name

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "notebook_dedup.py"
SPEC = importlib.util.spec_from_file_location("notebook_dedup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def notebook_with_code(source):
    return {
        "cells": [{"cell_type": "code", "source": source}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def test_same_program_with_comment_and_whitespace_changes_is_duplicate():
    original = notebook_with_code(
        ["import tensorflow as tf\n", "model = tf.keras.Sequential([tf.keras.layers.Dense(10)])\n"]
    )
    candidate = notebook_with_code(
        ["# MNIST model\n", "import  tensorflow  as  tf\n", "\n", "model=tf.keras.Sequential([tf.keras.layers.Dense(10)])\n"]
    )

    result = MODULE.compare_notebooks(candidate, original)

    assert result.is_duplicate is True
    assert result.score >= 0.9


def test_different_neural_network_architectures_are_not_duplicates():
    shared_setup = [
        "import tensorflow as tf\n",
        "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n",
        "x_train = x_train.astype('float32') / 255\n",
    ]
    dnn = notebook_with_code(
        shared_setup
        + ["model = tf.keras.Sequential([tf.keras.layers.Dense(32), tf.keras.layers.Dense(10)])\n"]
    )
    cnn = notebook_with_code(
        shared_setup
        + ["model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3), tf.keras.layers.Dense(10)])\n"]
    )

    result = MODULE.compare_notebooks(cnn, dnn)

    assert result.is_duplicate is False


def test_same_dataset_and_model_family_is_semantic_duplicate():
    keras_version = notebook_with_code(
        [
            "from tensorflow.keras.datasets import mnist\n",
            "from tensorflow.keras.layers import Dense\n",
            "model = Sequential([Dense(128, activation='relu'), Dense(10)])\n",
        ]
    )
    sklearn_version = notebook_with_code(
        [
            "digits = fetch_openml('mnist_784')\n",
            "network.add(tf.keras.layers.Dense(64))\n",
            "network.add(tf.keras.layers.Dense(10, activation='softmax'))\n",
        ]
    )

    result = MODULE.compare_notebooks(sklearn_version, keras_version)

    assert result.is_duplicate is True

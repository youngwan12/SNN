"""
Train.py 기반 학습 스크립트 (분석 코드 설정에 맞춤)
- 구조: 784 x 16 x 10 (MNIST 0~9 전체), 출력층 bias 없음 + NonNeg
- 분석 코드가 읽는 파일명으로 저장:
    model.keras, weights_01_raw.csv, weights_02_train_history.csv, weights_03_quantized.csv
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, constraints
from tensorflow.keras.datasets import mnist

HIDDEN_UNITS = 16
OUTPUT_UNITS = 10
LEVELS = 8


def quantize_weights(weights: np.ndarray) -> np.ndarray:
    """분석 코드와 동일한 균등 양자화"""
    w_min = weights.min()
    w_max = weights.max()
    step = (w_max - w_min) / LEVELS
    quantized = np.round((weights - w_min) / step).clip(0, LEVELS - 1) + 1
    return w_min + (quantized - 1) * step


def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(784,), name="input_784"),
        layers.Dense(HIDDEN_UNITS, activation="relu", name="hidden_layer"),
        layers.Dense(
            OUTPUT_UNITS,
            activation="softmax",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            name="output_layer_nonneg"
        )
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0
    return x_train, y_train, x_test, y_test


def weights_df(w: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(w, index=[f"hidden_{i}" for i in range(w.shape[0])],
                        columns=[f"class_{j}" for j in range(w.shape[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x_train, y_train, x_test, y_test = load_mnist()
    print(f"train {x_train.shape}, test {x_test.shape}")

    model = build_model()
    t0 = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    print(f"학습 시간: {time.time() - t0:.1f}s")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.6f}  Test Accuracy: {test_acc:.6f}")

    model.save(os.path.join(args.out, "model.keras"))
    raw = model.get_layer("output_layer_nonneg").get_weights()[0]  # (16, 10)
    weights_df(raw).to_csv(os.path.join(args.out, "weights_01_raw.csv"))
    pd.DataFrame(history.history).to_csv(os.path.join(args.out, "weights_02_train_history.csv"), index=False)
    weights_df(quantize_weights(raw)).to_csv(os.path.join(args.out, "weights_03_quantized.csv"))
    print(f"저장 완료 → {args.out}")


if __name__ == "__main__":
    main()

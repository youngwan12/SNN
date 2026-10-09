"""
균등 양자화 vs 분포 기반 양자화 — 출력층(16x10) weight, 8단계(3비트)

비교 방식
  float            : 양자화 없음 (기준)
  uniform          : 원본 분석 코드의 quantize_weights (step = 범위/8)
  uniform_pipeline : 원본 코드의 정수 단계 → 복원 경로 (weights_05_restored.csv 와 동일)
  uniform_std      : min~max 를 8레벨로 균등 분할 (양 끝 포함, step = 범위/7)
  quantile         : 분포 기반 ① 등빈도(분위수) 구간, 레벨 = 구간 평균
  kmeans           : 분포 기반 ② 최적 1D k-means (Lloyd-Max, MSE 최소 레벨)
  kmeans_zero      : 분포 기반 ③ 0 을 레벨 1 로 고정 + 나머지(>0) 7레벨 k-means
                     (NonNeg 제약으로 0 이 많이 몰리는 분포용)

사용법 (저장소 루트에서)
  python scripts/quant_compare.py data/seed42 data/seed1 data/seed2 data/seed3 data/seed4
출력
  results/  : 실행별 정확도, 요약, 코드북, 첫 번째 실행의 단계/복원 weight CSV
  figures/  : metrics.txt, summary.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

LEVELS = 8
METHODS = ["float", "uniform", "uniform_pipeline", "uniform_std", "quantile", "kmeans", "kmeans_zero"]


# ── 원본 분석 코드와 동일한 균등 양자화 ─────────────────
def quantize_weights(weights):
    w_min, w_max = weights.min(), weights.max()
    step = (w_max - w_min) / LEVELS
    q = np.round((weights - w_min) / step).clip(0, LEVELS - 1) + 1
    return w_min + (q - 1) * step


def uniform_pipeline(weights):
    """원본 코드 1단계: 양자화값 → get_stage → 복원 (min/max 를 양자화값에서 다시 잡음)"""
    wq = quantize_weights(weights)
    w_min, w_max = wq.min(), wq.max()
    step = (w_max - w_min) / LEVELS
    stage = np.round((wq - w_min) / step).clip(0, LEVELS - 1) + 1
    return w_min + (stage - 1) * step


# ── 코드북 기반 양자화 ────────────────────────────────
def quantile_codebook(weights, k=LEVELS):
    x = np.sort(weights.ravel().astype(np.float64))
    means = [c.mean() for c in np.array_split(x, k)]
    return np.unique(np.round(means, 12))  # 같은 값(예: 0이 몰린 구간)은 합쳐짐


def kmeans_codebook(weights, k=LEVELS):
    """정렬된 1D 데이터에 대한 동적계획법 → 전역 최적 k-means"""
    x = np.sort(weights.ravel().astype(np.float64))
    n = len(x)
    k = min(k, len(np.unique(x)))
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])

    D = np.full((k + 1, n + 1), np.inf)
    B = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for m in range(1, k + 1):
        for j in range(m, n + 1):
            i = np.arange(m - 1, j)
            cnt = j - i
            sse = (s2[j] - s2[i]) - (s1[j] - s1[i]) ** 2 / cnt
            c = D[m - 1, i] + sse
            t = int(np.argmin(c))
            D[m, j], B[m, j] = c[t], i[t]

    centers, j = [], n
    for m in range(k, 0, -1):
        i = B[m, j]
        centers.append(x[i:j].mean())
        j = i
    return np.array(sorted(centers))


def kmeans_zero_codebook(weights, k=LEVELS):
    w = weights.ravel().astype(np.float64)
    pos = w[w > 0]
    if len(pos) == len(w):
        return kmeans_codebook(w, k)
    return np.concatenate([[0.0], kmeans_codebook(pos, k - 1)])


def apply_codebook(weights, codebook):
    stage = np.abs(weights[..., None] - codebook).argmin(axis=-1)  # 0-based
    return codebook[stage], stage + 1                                 # 복원값, 1~8 단계


def quantize(method, w):
    if method == "float":
        return w.copy(), None
    if method == "uniform":
        return quantize_weights(w), None
    if method == "uniform_pipeline":
        return uniform_pipeline(w), None
    if method == "uniform_std":
        return apply_codebook(w, np.linspace(w.min(), w.max(), LEVELS))
    if method == "quantile":
        return apply_codebook(w, quantile_codebook(w))
    if method == "kmeans":
        return apply_codebook(w, kmeans_codebook(w))
    if method == "kmeans_zero":
        return apply_codebook(w, kmeans_zero_codebook(w))
    raise ValueError(method)


def quantize_codebook_levels(method, w):
    if method == "uniform":
        step = (w.max() - w.min()) / LEVELS
        return w.min() + np.arange(LEVELS) * step
    if method == "uniform_std":
        return np.linspace(w.min(), w.max(), LEVELS)
    if method == "quantile":
        return quantile_codebook(w)
    if method == "kmeans_zero":
        return kmeans_zero_codebook(w)
    return kmeans_codebook(w)


# ── 평가 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="학습 결과 폴더들 (첫 번째 폴더에 대해 단계/복원 CSV 저장)")
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    import tensorflow as tf
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

    rows, codebooks = [], {}
    for r_i, run in enumerate(args.runs):
        run_name = os.path.basename(os.path.normpath(run))
        model = tf.keras.models.load_model(os.path.join(run, "model.keras"))
        W1, b1 = model.layers[0].get_weights()
        W2 = pd.read_csv(os.path.join(run, "weights_01_raw.csv"), index_col=0).values.astype(np.float32)
        H = np.maximum(x_test @ W1 + b1, 0)  # 은닉층 출력은 한 번만 계산

        def acc(w_out):
            return float(np.mean((H @ w_out).argmax(axis=1) == y_test))

        # 원본 코드 방식(Keras 모델에 weight 교체 후 evaluate)과 일치하는지 확인
        _, keras_float = model.evaluate(x_test, y_test, verbose=0)
        wq_uni = quantize_weights(W2)
        model.layers[1].set_weights([wq_uni])
        _, keras_uni = model.evaluate(x_test, y_test, verbose=0)
        print(f"[{run_name}] 검증: keras float {keras_float:.4f} / numpy {acc(W2):.4f}, "
              f"keras uniform {keras_uni:.4f} / numpy {acc(wq_uni.astype(np.float32)):.4f}")
        csv03 = pd.read_csv(os.path.join(run, "weights_03_quantized.csv"), index_col=0).values
        assert np.allclose(csv03, wq_uni, atol=1e-6), "weights_03_quantized.csv 와 균등 양자화 결과 불일치"

        for method in METHODS:
            wq, stage = quantize(method, W2)
            rows.append({
                "run": run_name,
                "method": method,
                "accuracy": acc(wq.astype(np.float32)),
                "weight_mse": float(np.mean((wq - W2) ** 2)),
                "levels_used": int(len(np.unique(np.round(wq, 10)))),
            })
            if r_i == 0 and stage is not None and method in ("quantile", "kmeans", "kmeans_zero"):
                idx, cols = [f"hidden_{i}" for i in range(W2.shape[0])], [f"class_{j}" for j in range(W2.shape[1])]
                pd.DataFrame(stage, index=idx, columns=cols).to_csv(
                    os.path.join(args.results, f"weights_{method}_stage.csv"))
                pd.DataFrame(wq, index=idx, columns=cols).to_csv(
                    os.path.join(args.results, f"weights_{method}_restored.csv"))

        _, km_stage = apply_codebook(W2, kmeans_codebook(W2))
        uni_stage = (np.round((W2 - W2.min()) / ((W2.max() - W2.min()) / LEVELS)).clip(0, LEVELS - 1) + 1).astype(int)
        codebooks[run_name] = {
            "weight_min": float(W2.min()), "weight_max": float(W2.max()),
            **{m: [float(v) for v in quantize_codebook_levels(m, W2)]
               for m in ("uniform", "uniform_std", "quantile", "kmeans", "kmeans_zero")},
            "stage_counts": {"uniform": [int((uni_stage == s).sum()) for s in range(1, LEVELS + 1)],
                             "kmeans": [int((km_stage == s).sum()) for s in range(1, LEVELS + 1)]},
        }
        if r_i == 0:
            first = codebooks[run_name]
            cb_df = pd.DataFrame({m: pd.Series(first[m]) for m in ("uniform", "uniform_std", "quantile", "kmeans", "kmeans_zero")})
            cb_df.index = [f"stage{i + 1}" for i in range(len(cb_df))]
            cb_df.to_csv(os.path.join(args.results, f"codebooks_{run_name}.csv"))

    df = pd.DataFrame(rows)
    base = df[df.method == "uniform"].set_index("run").accuracy
    df["delta_vs_uniform_pp"] = (df.accuracy - df.run.map(base)) * 100
    df.to_csv(os.path.join(args.results, "results_per_run.csv"), index=False)

    summary = df.groupby("method", sort=False).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
        delta_vs_uniform_pp_mean=("delta_vs_uniform_pp", "mean"),
        weight_mse_mean=("weight_mse", "mean"), levels_used_mean=("levels_used", "mean"),
    ).reindex(METHODS)
    summary.to_csv(os.path.join(args.results, "summary.csv"))

    km = pd.DataFrame({r: cb["kmeans"] for r, cb in codebooks.items()}).T
    km.columns = [f"stage{i + 1}" for i in range(km.shape[1])]
    km.index.name = "run"
    km.to_csv(os.path.join(args.results, "kmeans_levels_all_seeds.csv"))

    write_metrics(df, summary, codebooks, args.figures)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n== 실행별 결과 ==")
    print(df.pivot(index="run", columns="method", values="accuracy")[METHODS].round(4))
    print("\n== 요약 (runs = %d) ==" % len(args.runs))
    print(summary.round(5))


def write_metrics(df, summary, codebooks, out_dir):
    runs = list(codebooks)
    acc = df.pivot(index="run", columns="method", values="accuracy").loc[runs]
    kmeans_wins = int((acc["kmeans"] > acc["uniform"]).sum())

    lines = [f"[Setup]  MNIST 784x16x10, output layer 16x10 = 160 weights, {LEVELS} levels (3 bit), "
             f"test set 10,000 images, runs = {', '.join(runs)}", "",
             "[Accuracy, mean ± std over runs]  (delta = vs uniform, percentage points)"]
    for m in METHODS:
        s = summary.loc[m]
        lines.append(f"  {m:17s} {s.acc_mean * 100:6.2f}% ± {s.acc_std * 100:4.2f}   "
                     f"delta {s.delta_vs_uniform_pp_mean:+5.2f} pp   weight MSE {s.weight_mse_mean:.5f}   "
                     f"levels used {s.levels_used_mean:.1f}")
    lines += ["", f"[k-means vs uniform]  k-means higher in {kmeans_wins}/{len(runs)} runs; per run delta (pp): "
              + ", ".join(f"{r} {(acc.loc[r, 'kmeans'] - acc.loc[r, 'uniform']) * 100:+.2f}" for r in runs), ""]
    lines.append("[Per-run accuracy %]")
    lines.append("  " + "run".ljust(8) + "".join(m.rjust(18) for m in METHODS))
    for r in runs:
        lines.append("  " + r.ljust(8) + "".join(f"{acc.loc[r, m] * 100:18.2f}" for m in METHODS))
    lines += ["", "[Level values, stage 1 -> 8]"]
    for r, cb in codebooks.items():
        lines.append(f"  {r}  (weight range {cb['weight_min']:.4f} ~ {cb['weight_max']:.4f})")
        for m in ("uniform", "kmeans", "kmeans_zero"):
            lines.append(f"    {m:12s}: " + ", ".join(f"{v:.4f}" for v in cb[m]))
        for m in ("uniform", "kmeans"):
            lines.append(f"    {m + ' count':12s}: " + ", ".join(str(c) for c in cb["stage_counts"][m]))

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    out = {
        "setup": {"architecture": "784x16x10", "quantized_layer": "output (16x10, no bias, NonNeg)",
                  "levels": LEVELS, "test_images": 10000, "runs": runs},
        "summary": {m: {k: (None if pd.isna(v) else round(float(v), 6)) for k, v in summary.loc[m].items()}
                    for m in METHODS},
        "per_run_accuracy": {r: {m: round(float(acc.loc[r, m]), 6) for m in METHODS} for r in runs},
        "codebooks": codebooks,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

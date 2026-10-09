# SNN – 3-bit output-layer weight quantization (MNIST 784×16×10)

Uniform vs. distribution-based 8-level (3-bit, stages 1–8) quantization of the 16×10 output-layer
weights of a 784×16×10 MNIST network, for mapping the weights onto 3-bit synapses.

`python scripts/quant_compare.py data/seed42 data/seed1 data/seed2 data/seed3 data/seed4` loads the
trained Keras models in `data/`, quantizes the output-layer weights with each method and measures
accuracy on the 10,000 MNIST test images. Per-run tables, codebooks and quantized weights go to
`results/`; extracted numbers go to `figures/metrics.txt` and `figures/summary.json`.
`python scripts/plot_report.py` and `python scripts/heatmap_report.py` then write the figures to
`figures/` as PDF and PNG.

Requires `tensorflow`, `numpy`, `pandas` and `matplotlib`. Figure labels are Korean (Malgun Gothic).

## Training

`python scripts/train_16x10.py --seed 42 --out data/seed42`: 784 → 16 (ReLU) → 10 (softmax), output
layer without bias and with a non-negative weight constraint, Adam, 200 epochs, batch 128. Five seeds
(42, 1, 2, 3, 4). Each `data/seedN/` holds `model.keras`, `weights_01_raw.csv` (16×10 output
weights), `weights_02_train_history.csv` and `weights_03_quantized.csv` (uniform quantization).

## Results (mean of 5 seeds)

| Method | Test accuracy | vs. uniform |
|---|---|---|
| Float (no quantization) | 95.05% | +3.09 pp |
| Uniform (`quantize_weights`, step = range/8) | 91.96% | – |
| **Distribution: k-means** | **93.22%** | **+1.26 pp** |
| Distribution: 0 fixed + k-means | 92.42% | +0.46 pp |
| Distribution: quantile | 91.40% | −0.56 pp |
| Uniform, both ends included (step = range/7) | 90.06% | −1.90 pp |
| Uniform, integer stage → restore path | 90.04% | −1.92 pp |

k-means is higher than uniform in all 5 seeds (+0.33 to +2.61 pp). Seed 4, the median run:
ANN 94.55% → k-means 93.01% (uniform 92.24%).

| Method | How the 8 levels are chosen |
|---|---|
| uniform | `w_min + k·(w_max − w_min)/8`, k = 0…7 (original analysis code; the top level stays below `w_max`) |
| uniform_pipeline | original stage → restore path; it re-derives min/max from already-quantized values, so the step shrinks to 7/8 and stages 7 and 8 merge |
| kmeans | globally optimal 1-D k-means (dynamic programming); each level is the mean of its cluster, minimizing weight MSE |
| kmeans_zero | level 1 fixed at 0, the other 7 levels by k-means on the positive weights |
| quantile | equal-count bins, each level is the bin mean |

## Figures

| Figure | Source | Content |
|---|---|---|
| fig1_quant_comparison | data/seed42, results/results_per_run.csv | weight histogram with each method's 8 levels; test accuracy per method over 5 seeds |
| fig2_heatmaps_seed42 | data/seed42 | float / uniform / k-means weights (cell = stage 1–8) and quantization error per weight |
| slide_kmeans_seed4 | data/seed4 (`scripts/slide_heatmap.py`) | float vs. k-means 3-bit weights for the presentation |

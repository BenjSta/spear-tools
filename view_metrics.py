import numpy as np
import pandas as pd

CSV_FILE = "metrics_fullsubnet.csv"
CSV_FILE_COMPARE = "metrics_fullsubnet.csv"
COMPARE = False

full_metrics_compare = pd.read_csv(CSV_FILE_COMPARE)
full_metrics = pd.read_csv(CSV_FILE)
assert np.all(
    full_metrics_compare.global_index.to_numpy() == full_metrics.global_index.to_numpy()
)

pesq_lr_d2 = []
pesq_lr_d3 = []
pesq_lr_d4 = []

sisdr_lr_d2 = []
sisdr_lr_d3 = []
sisdr_lr_d4 = []

datasets = np.array([int(f[1]) for f in full_metrics["file_name"].tolist()])
for col in range(4, full_metrics.shape[1]):
    metric_name = full_metrics.columns[col]
    metric_compare = COMPARE * np.nan_to_num(
        full_metrics_compare[metric_name].to_numpy()
    )
    metric = full_metrics[metric_name].to_numpy()

    if "PESQ" in metric_name and not "NB" in metric_name:
        pesq_lr_d2.append(metric[datasets == 2] - metric_compare[datasets == 2])
        pesq_lr_d3.append(metric[datasets == 3] - metric_compare[datasets == 3])
        pesq_lr_d4.append(metric[datasets == 4] - metric_compare[datasets == 4])

    if "SI-SDR" in metric_name:
        sisdr_lr_d2.append(metric[datasets == 2] - metric_compare[datasets == 2])
        sisdr_lr_d3.append(metric[datasets == 3] - metric_compare[datasets == 3])
        sisdr_lr_d4.append(metric[datasets == 4] - metric_compare[datasets == 4])

pesq_lr_d2 = np.mean(np.concatenate([pesq_lr_d2], 0), 0)
pesq_lr_d3 = np.mean(np.concatenate([pesq_lr_d3], 0), 0)
pesq_lr_d4 = np.mean(np.concatenate([pesq_lr_d4], 0), 0)

sisdr_lr_d2 = np.mean(np.concatenate([sisdr_lr_d2], 0), 0)
sisdr_lr_d3 = np.mean(np.concatenate([sisdr_lr_d3], 0), 0)
sisdr_lr_d4 = np.mean(np.concatenate([sisdr_lr_d4], 0), 0)

print(
    "(delta) PESQ_D2 | mean: %.2f, median: %.2f, first quart.: %.2f, third quart.: %.2f"
    % (
        np.nanmean(pesq_lr_d2),
        np.nanmedian(pesq_lr_d2),
        np.nanquantile(pesq_lr_d2, 0.25),
        np.nanquantile(pesq_lr_d2, 0.75),
    )
)
print(
    "(delta) PESQ_D3 | mean: %.2f, median: %.2f, first quart.: %.2f, third quart.: %.2f"
    % (
        np.nanmean(pesq_lr_d3),
        np.nanmedian(pesq_lr_d3),
        np.nanquantile(pesq_lr_d3, 0.25),
        np.nanquantile(pesq_lr_d3, 0.75),
    )
)
print(
    "(delta) PESQ_D4 | mean: %.2f, median: %.2f, first quart.: %.2f, third quart.: %.2f"
    % (
        np.nanmean(pesq_lr_d4),
        np.nanmedian(pesq_lr_d4),
        np.nanquantile(pesq_lr_d4, 0.25),
        np.nanquantile(pesq_lr_d4, 0.75),
    )
)
print(
    "Mean (delta) PESQ: %.2f"
    % (
        (
            np.nanmean(([pesq_lr_d2]))
            + np.nanmean(([pesq_lr_d3]))
            + np.nanmean(([pesq_lr_d4]))
        )
        / 3
    )
)

if COMPARE:
    print(
        "Compare percentage D2 PESQ:",
        np.mean(pesq_lr_d2[np.logical_not(np.isnan(pesq_lr_d2))] > 0),
    )
    print(
        "Compare percentage D3 PESQ:",
        np.mean(pesq_lr_d3[np.logical_not(np.isnan(pesq_lr_d3))] > 0),
    )
    print(
        "Compare percentage D4 PESQ:",
        np.mean(pesq_lr_d4[np.logical_not(np.isnan(pesq_lr_d4))] > 0),
    )

print("(delta) SISDR_D2: %.1f" % np.nanmean(np.array([sisdr_lr_d2])))
print("(delta) SISDR_D3: %.1f" % np.nanmean(np.array([sisdr_lr_d3])))
print("(delta) SISDR_D4: %.1f" % np.nanmean(np.array([sisdr_lr_d4])))
print(
    "Mean (delta) SISDR: %.1f"
    % (
        (
            np.nanmean(np.array([sisdr_lr_d2]))
            + np.nanmean(np.array([sisdr_lr_d3]))
            + np.nanmean(np.array([sisdr_lr_d4]))
        )
        / 3
    )
)

if COMPARE:
    print(
        "Compare percentage D2 SISDR:",
        np.mean(sisdr_lr_d2[np.logical_not(np.isnan(sisdr_lr_d2))] > 0),
    )
    print(
        "Compare percentage D3 SISDR:",
        np.mean(sisdr_lr_d3[np.logical_not(np.isnan(sisdr_lr_d3))] > 0),
    )
    print(
        "Compare percentage D4 SISDR:",
        np.mean(sisdr_lr_d4[np.logical_not(np.isnan(sisdr_lr_d4))] > 0),
    )


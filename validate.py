import toml

METRICS = ["SI-SDR", "PESQ"]  # choice of metrics to run
config = toml.load("configs/fullsubnet_pretrained.toml")
CHECKPOINT_PATH = ""
VAL_SEGMENTS_FILE = "spear-tools/analysis/segments_Dev.csv"
WRITE_PROCESSED_AUDIO = False
NUM_WORKERS_METRICS = 20
NUM_WORKERS_LOADER = 20
DEVICE = "cuda:0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('./spear-tools/')

import torch
import numpy as np
import soundfile
import tqdm
import glob
import pandas as pd
import parse
from analysis.spear_evaluate import compute_metrics
import models.nn_processor as nn_processor
from torch.utils.data import DataLoader
import soundfile
from dataset import SpearDataset
import scipy.signal
from joblib import Parallel, delayed
from ptflops import get_model_complexity_info

if WRITE_PROCESSED_AUDIO:
    processed_audio_dir = (
        "processed_val_audio_" + os.path.split(CHECKPOINT_PATH)[-1][:-3]
    )
    os.mkdir(processed_audio_dir)

# Setting up columns for metric matrix
isMBSTOI = "MBSTOI" in METRICS
if isMBSTOI:
    METRICS.remove("MBSTOI")
side_str = ["L", "R"]
# 'cols' are the name of columns in metric matrix
cols = [
    "%s (%s)" % (x, y) for x in METRICS for y in side_str
]  # creating 2x (Left & Right) mono-based metric
if isMBSTOI:
    cols.insert(0, "MBSTOI")  # stereo-based metric
cols_csv = ["global_index", "file_name", "chunk_index"] + cols


architecture = config["architecture"]
if architecture != "MaxDirAndFullsubnet":
    net_config = config["net_config"]
else:
    net_config = None
processing_winlen = config["processing_winlen"]
processing_hopsize = config["processing_hopsize"]
fs = config["fs"]

pathlist_vds2 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Dev/Dataset_2/Reference_Audio/*/*/ref_*.wav"
)
pathlist_vds3 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Dev/Dataset_3/Reference_Audio/*/*/ref_*.wav"
)
pathlist_vds4 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Dev/Dataset_4/Reference_Audio/*/*/ref_*.wav"
)
vflist = sorted(
    [os.path.split(p)[-1][4:-4] for p in pathlist_vds2 + pathlist_vds3 + pathlist_vds4]
)
print("val set size: %d minutes" % len(vflist))


vds_full = SpearDataset(
    "spear-tools/analysis/spear_data/Main/Dev", vflist, 60, processing_hopsize / fs, 0.0, 0.0, 48000
)
vdl_full = DataLoader(
    vds_full, 1, shuffle=False, num_workers=1, drop_last=False, prefetch_factor=1
)

try:
    sigma = np.load("gaussian_sigma_fs%g_blk%d.npy" % (fs, processing_winlen))
except:
    raise RuntimeError("Sigma not computed yet, run script first")


model = getattr(nn_processor, architecture)(
    vds_full.get_ATFs(), net_config, fs, processing_winlen, processing_hopsize, sigma
)
model = model.to(DEVICE)

num_doa = int(np.ceil((8 * fs / processing_hopsize)))


def constr(input_res):
    return {
        "noisy": torch.ones(tuple([1]) + input_res[0]).to(DEVICE),
        "doa": torch.ones(tuple([1]) + input_res[1]).to(DEVICE),
    }


macs, params = get_model_complexity_info(
    model,
    ((1 * fs, 6), (num_doa, 3)),
    input_constructor=constr,
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True,
)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))

best_mean_pesq = -np.Inf
if config['architecture'] != 'MaxDirAndFullsubnet':
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_mean_pesq = checkpoint["best_mean_pesq"]
    epoch_offset = checkpoint["epoch"]

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

model.eval()
segments = pd.read_csv(VAL_SEGMENTS_FILE)


def enhance_generator():
    for val_dataset_index, val_sample in enumerate(vdl_full):
        (x_noisy, x_ref, doa, rms) = val_sample
        x_noisy = x_noisy[0, ...].detach().cpu().numpy()
        x_ref = x_ref[0, ...].detach().cpu().numpy()
        doa = doa[0, ...].detach().cpu().numpy()
        rms = rms[0, ...].detach().cpu().numpy()
        filename = vflist[val_dataset_index]
        if fs != vds_full.fs:
            noisy_resampled = scipy.signal.resample_poly(
                x_noisy, fs, vds_full.fs, axis=0
            )
        else:
            noisy_resampled = x_noisy.copy()

        doa = torch.from_numpy(doa).to(DEVICE)
        rms = torch.from_numpy(np.array([rms])).to(DEVICE)
        enh = np.zeros_like(noisy_resampled[:, -2:])
        noisy_resampled = torch.from_numpy(noisy_resampled).to(DEVICE)

        # process in segments, otherwise would be too expensive
        doa_fs = 1 / vds_full.doa_hopsize
        SEGHOP_DOA = int(8.5 * doa_fs) / doa_fs
        SEGLEN_DOA = int(10 * doa_fs) / doa_fs

        if fs != vds_full.fs:
            doa_hopsize_samples = fs / doa_fs
            assert (
                np.abs(doa_hopsize_samples - np.round(doa_hopsize_samples)) < 1e-9
            )  # ignore rounding error
            doa_hopsize_samples = int(np.round(doa_hopsize_samples))
        else:
            doa_hopsize_samples = processing_hopsize

        num_segments = int(
            np.maximum(np.ceil((doa.shape[0] / doa_fs - SEGLEN_DOA) / SEGHOP_DOA + 1), 1)
        )

        for seg_ind in np.arange(num_segments):
            start_ind_doa = seg_ind * int(SEGHOP_DOA * doa_fs)
            end_ind_doa = start_ind_doa + int(SEGLEN_DOA * doa_fs)
            start_ind = start_ind_doa * doa_hopsize_samples
            end_ind = np.minimum(
                end_ind_doa * doa_hopsize_samples, noisy_resampled.shape[0]
            )
            end_ind_doa = np.minimum(end_ind_doa + 2, doa.shape[0])
            if end_ind <= start_ind:
                break
            enh_seg = (
                model(
                    noisy_resampled[None, start_ind:end_ind, :],
                    doa[None, start_ind_doa:end_ind_doa, :],
                )[0, :, :]
                .detach()
                .cpu()
                .numpy()
            )
            if seg_ind == 0:
                enh[start_ind:end_ind, :] = enh_seg
            else:
                enh[last_end_ind:end_ind, :] = enh_seg[
                    last_end_ind - start_ind : end_ind, :
                ]
            last_end_ind = end_ind


        if fs != vds_full.fs:
            x_proc = scipy.signal.resample_poly(enh, vds_full.fs, fs, axis=0)
        else:
            x_proc = enh.copy()

        if WRITE_PROCESSED_AUDIO:
            soundfile.write(
                os.path.join(processed_audio_dir, filename + ".wav"),
                x_proc,
                vds_full.fs,
            )

        yield filename, x_proc, x_ref


def val_metrics(filename, x_proc, x_ref):
    info = parse.parse("D{}_S{}_M{}_ID{}", filename)
    dataset = int(info[0])
    session = int(info[1])
    minute = int(info[2])
    target_id = int(info[3])

    segments_for_file = segments[segments["dataset"] == ("D%d" % dataset)]
    segments_for_file = segments_for_file[segments_for_file["session"] == session]
    segments_for_file = segments_for_file[segments_for_file["minute"] == minute]
    segments_for_file = segments_for_file[segments_for_file["target_ID"] == target_id]

    cols_csv = ["global_index", "file_name", "chunk_index"] + cols

    # Loop through chunks
    metric_vals_df_list = []
    nSeg = len(segments_for_file)

    for n in range(nSeg):
        seg = segments_for_file.iloc[n]
        dataset = int(seg["dataset"][1])  # intseg['dataset'][1]) # integer
        session = seg["session"]  # integer
        minute = seg["minute"]  # integer
        file_name = seg[
            "file_name"
        ]  # was original EasyCom name e.g. 01-00-288, now vad_, no nothing
        sample_start = int((seg["sample_start"] - 1) * vds_full.fs / 48000)
        sample_stop = int((seg["sample_stop"] - 1) * vds_full.fs / 48000)

        # get chunk info
        chunk_info = [seg["global_index"], file_name, seg["chunk_index"]]

        x_proc_seg = x_proc[sample_start : sample_stop + 1, :]
        x_ref_seg = x_ref[sample_start : sample_stop + 1, :]

        scores = compute_metrics(x_proc_seg, x_ref_seg, vds_full.fs, cols)
        metric_vals_df_list.append(
            pd.DataFrame([chunk_info + scores], columns=cols_csv)
        )

    if len(metric_vals_df_list) > 0:
        metric_vals_df = pd.concat(metric_vals_df_list)
    else:
        metric_vals_df = pd.DataFrame([], columns=cols_csv)
    return metric_vals_df


eg = enhance_generator()
metric_dataframes_costs_filenames = Parallel(n_jobs=NUM_WORKERS_METRICS)(
    delayed(val_metrics)(*tup) for tup in tqdm.tqdm(eg, total=vds_full.__len__())
)


full_metrics = pd.concat(metric_dataframes_costs_filenames)

if architecture != 'MaxDirAndFullsubnet':
    full_metrics.to_csv("metrics_" + os.path.split(CHECKPOINT_PATH)[-1][:-3] + ".csv")
else:
    full_metrics.to_csv("metrics_fullsubnet.csv")

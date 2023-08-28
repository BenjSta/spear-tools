import toml

config = toml.load("configs/subband_compon.toml")
VAL_SEGMENTS_FILE = "spear-tools/analysis/segments_Dev.csv"

import torch
import os
import numpy as np
import soundfile
import tqdm
import glob
import pandas as pd
import parse
import time

import sys
sys.path.append('./spear-tools/')

from analysis.spear_evaluate import compute_metrics
import models.nn_processor as nn_processor
from torch.utils.data import DataLoader
import tensorflow as tf
import soundfile
from dataset import SpearDataset
from models.torch_sigproc import MultichannelSTFTLayer
import scipy.signal
from joblib import Parallel, delayed
from ptflops import get_model_complexity_info

os.environ["CUDA_VISIBLE_DEVICES"] = config["visible_cuda_devices"]

architecture = config["architecture"]
net_config = config["net_config"]
processing_winlen = config["processing_winlen"]
processing_hopsize = config["processing_hopsize"]
fs = config["fs"]
duration = config["duration"]
batch_size = config["batch_size"]
resume = config["resume"]
num_epochs = config["num_epochs"]
num_workers_loader = config["num_workers_loader"]
tensorboard_logdir = config["tensorboard_logdir"]
checkpoint_dir = config["checkpoint_dir"]
learning_rate_epochs = config["learning_rate_epochs"]
learning_rates = config["learning_rates"]
log_name = config["log_name"]

cost_function_winlen = config["cost_function_winlen"]
cost_function_hopsize = config["cost_function_hopsize"]

validation_only = config["validation_only"]
validate_every = config["validate_every"]
validation_num_audio_samples = config["validation_num_audio_samples"]
validation_seed = config["validation_seed"]
device = config["device"]

pathlist_ds2 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Train/Dataset_2/Reference_Audio/*/*/ref_*.wav"
)
pathlist_ds3 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Train/Dataset_3/Reference_Audio/*/*/ref_*.wav"
)
pathlist_ds4 = glob.glob(
    "spear-tools/analysis/spear_data/Extra/Train/Dataset_4/Reference_Audio/*/*/ref_*.wav"
)
flist = sorted(
    [os.path.split(p)[-1][4:-4] for p in pathlist_ds2 + pathlist_ds3 + pathlist_ds4]
)
print("train set size: %d minutes" % len(flist))

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

np.random.seed(validation_seed)
validation_audio_samples_full = np.random.choice(
    vflist, size=validation_num_audio_samples, replace=False
)
np.random.seed()

ds = SpearDataset(
    "spear-tools/analysis/spear_data/Main/Train",
    flist,
    duration,
    processing_hopsize / fs,
    0.7,
    0.6,
    fs,
)
dl = DataLoader(
    ds, batch_size, shuffle=True, num_workers=num_workers_loader, drop_last=True
)


vds = SpearDataset(
    "spear-tools/analysis/spear_data/Main/Dev", vflist, 7, processing_hopsize / fs, 0.7, 0.3, fs
)
vds_full = SpearDataset(
    "spear-tools/analysis/spear_data/Main/Dev", vflist, 60, processing_hopsize / fs, 0.0, 0.0, fs
)
vdl_full = DataLoader(
    vds_full,
    1,
    shuffle=False,
    num_workers=num_workers_loader,
    drop_last=False,
    prefetch_factor=num_workers_loader,
)

try:
    sigma = np.load("gaussian_sigma_fs%g_blk%d.npy" % (fs, processing_winlen))
except:
    raise RuntimeError("Sigma not computed yet, run script first")


model = getattr(nn_processor, architecture)(
    ds.get_ATFs(), net_config, fs, processing_winlen, processing_hopsize, sigma
)
model = model.to(device)

num_doa = int(np.ceil((8 * fs / processing_hopsize)))


def constr(input_res):
    return {
        "noisy": torch.ones(tuple([1]) + input_res[0]).to(device),
        "doa": torch.ones(tuple([1]) + input_res[1]).to(device),
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
if not len(list(model.parameters())) == 0:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])

best_mean_pesq = -np.Inf
if resume:
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "model_%s_latest_epoch.pt" % log_name)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_mean_pesq = checkpoint["best_mean_pesq"]
    epoch_offset = checkpoint["epoch"]
else:
    epoch_offset = 0


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

parallel_model = torch.nn.DataParallel(model, output_device=0)
writer = tf.summary.create_file_writer(os.path.join(tensorboard_logdir, log_name))


cost_function_stft = MultichannelSTFTLayer(
    cost_function_winlen, cost_function_hopsize, torch.hann_window
).to(device)
fvec = torch.arange(cost_function_winlen // 2 + 1) / cost_function_winlen * fs


def cost_function(clean, enh, rms):
    enh = enh.to(clean.device)
    clean_spec = cost_function_stft(clean / rms[:, None, None])
    enh_spec = cost_function_stft(enh / rms[:, None, None])

    clean_mag = torch.abs(clean_spec)
    enh_mag = torch.abs(enh_spec)

    clean_mag = torch.clamp(clean_mag, 1e-12, None)
    enh_mag = torch.clamp(enh_mag, 1e-12, None)
    clean_unit_phasor = clean_spec / clean_mag
    enh_unit_phasor = enh_spec / enh_mag
    mag_compressed_loss = (clean_mag**0.3 - enh_mag**0.3) ** 2

    phase_aware_compressed_loss = (
        torch.abs(
            (clean_mag) ** 0.3 * clean_unit_phasor - (enh_mag) ** 0.3 * enh_unit_phasor
        )
        ** 2
    )

    stft_loss = 0.7 * torch.mean(mag_compressed_loss, dim=[1, 2, 3]) + 0.3 * torch.mean(
        phase_aware_compressed_loss, dim=[1, 2, 3]
    )

    return stft_loss


def validate(epoch):
    # defines the time periods for each file where it is valid to compute metrics
    segments = pd.read_csv(VAL_SEGMENTS_FILE)

    with writer.as_default():
        np.random.seed(validation_seed)

        def enhance_generator():
            for val_dataset_index, val_sample in enumerate(vdl_full):
                # torch.cuda.empty_cache()
                (
                    x_noisy,
                    x_ref,
                    doa,
                    rms,
                ) = val_sample  # vds_full.__getitem__(val_dataset_index)
                x_noisy = x_noisy[0, ...].detach().cpu().numpy()
                x_ref = x_ref[0, ...].detach().cpu().numpy()
                doa = doa[0, ...].detach().cpu().numpy()
                rms = rms[0, ...].detach().cpu().numpy()
                filename = vflist[val_dataset_index]
                if fs != vds_full.fs:
                    noisy_resampled = scipy.signal.resample_poly(
                        x_noisy, fs, vds_full.fs, axis=0
                    )
                    clean_resampled = scipy.signal.resample_poly(
                        x_ref, fs, vds_full.fs, axis=0
                    )
                else:
                    noisy_resampled = x_noisy.copy()
                    clean_resampled = x_ref.copy()

                doa = torch.from_numpy(doa).to(device)
                rms = torch.from_numpy(np.array([rms])).to(device)
                enh = np.zeros_like(noisy_resampled[:, -2:])
                noisy_resampled = torch.from_numpy(noisy_resampled).to(device)
                # process in segments, otherwise would be too expensive
                doa_fs = 1 / vds_full.doa_hopsize
                SEGHOP_DOA = int(8.5 * doa_fs) / doa_fs
                SEGLEN_DOA = int(10 * doa_fs) / doa_fs
                if fs != vds_full.fs:
                    doa_hopsize_samples = fs / doa_fs
                    assert (
                        np.abs(doa_hopsize_samples - np.round(doa_hopsize_samples))
                        < 1e-9
                    )  # ignore rounding error
                    doa_hopsize_samples = int(np.round(doa_hopsize_samples))
                else:
                    doa_hopsize_samples = processing_hopsize

                num_segments = int(
                    np.ceil((doa.shape[0] / doa_fs - SEGLEN_DOA) / SEGHOP_DOA + 1)
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

                cost = (
                    cost_function(
                        torch.from_numpy(clean_resampled[None, ...]).to(device),
                        torch.from_numpy(enh[None, ...]).to(device),
                        rms.to(device),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                cost = cost[0]
                if fs != vds_full.fs:
                    x_proc = scipy.signal.resample_poly(enh, vds_full.fs, fs, axis=0)
                else:
                    x_proc = enh.copy()
                yield filename, x_proc, x_ref, x_noisy[:, -2:], cost

        def val_metrics(filename, x_proc, x_ref, x_noisy, cost):
            info = parse.parse("D{}_S{}_M{}_ID{}", filename)
            dataset = int(info[0])
            session = int(info[1])
            minute = int(info[2])
            target_id = int(info[3])

            segments_for_file = segments[segments["dataset"] == ("D%d" % dataset)]
            segments_for_file = segments_for_file[
                segments_for_file["session"] == session
            ]
            segments_for_file = segments_for_file[segments_for_file["minute"] == minute]
            segments_for_file = segments_for_file[
                segments_for_file["target_ID"] == target_id
            ]

            # choice of metrics to run
            metrics = ["PESQ", "SI-SDR"]

            # Setting up columns for metric matrix
            isMBSTOI = "MBSTOI" in metrics
            if isMBSTOI:
                metrics.remove("MBSTOI")
            side_str = ["L", "R"]
            # 'cols' are the name of columns in metric matrix
            cols = [
                "%s (%s)" % (x, y) for x in metrics for y in side_str
            ]  # creating 2x (Left & Right) mono-based metric
            if isMBSTOI:
                cols.insert(0, "MBSTOI")  # stereo-based metric

            cols_csv = ["global_index", "file_name", "chunk_index"] + cols

            # Loop through chunks
            metric_vals_df_list = []
            nSeg = len(segments_for_file)
            if nSeg == 0 and filename in validation_audio_samples_full:
                norm = 0.7 / np.max(
                    np.abs(x_noisy[5 * vds_full.fs : 12 * vds_full.fs, :])
                )
                while True:
                    try:
                        soundfile.write(
                            "tmp_proc_audio/"
                            + filename
                            + "_"
                            + log_name
                            + "_clean.wav",
                            norm * x_ref[5 * vds_full.fs : 12 * vds_full.fs, :],
                            vds_full.fs,
                        )
                        soundfile.write(
                            "tmp_proc_audio/"
                            + filename
                            + "_"
                            + log_name
                            + "_noisy.wav",
                            norm * x_noisy[5 * vds_full.fs : 12 * vds_full.fs, :],
                            vds_full.fs,
                        )
                        soundfile.write(
                            "tmp_proc_audio/" + filename + "_" + log_name + "_enh.wav",
                            norm * x_proc[5 * vds_full.fs : 12 * vds_full.fs, :],
                            vds_full.fs,
                        )
                        break
                    except:
                        print("cannot write to audio file, trying again ...")
                        time.sleep(0.5)

            for n in range(nSeg):
                seg = segments_for_file.iloc[n]
                dataset = int(seg["dataset"][1])  # intseg['dataset'][1]) # integer
                session = seg["session"]  # integer
                minute = seg["minute"]  # integer
                file_name = seg[
                    "file_name"
                ]  # was original EasyCom name e.g. 01-00-288, now vad_, no nothing
                target_ID = seg["target_ID"]  # integer
                sample_start = int((seg["sample_start"] - 1) * vds_full.fs / 48000)
                sample_stop = int((seg["sample_stop"] - 1) * vds_full.fs / 48000)

                # get chunk info
                chunk_info = [seg["global_index"], file_name, seg["chunk_index"]]

                x_proc_seg = x_proc[sample_start : sample_stop + 1, :]
                x_ref_seg = x_ref[sample_start : sample_stop + 1, :]
                x_noisy_seg = x_noisy[sample_start : sample_stop + 1, :]

                if n == 0 and filename in validation_audio_samples_full:
                    norm = 0.7 / np.max(np.abs(x_noisy_seg))
                    while True:
                        try:
                            soundfile.write(
                                "tmp_proc_audio/"
                                + filename
                                + "_"
                                + log_name
                                + "_clean.wav",
                                norm * x_ref_seg,
                                vds_full.fs,
                            )
                            soundfile.write(
                                "tmp_proc_audio/"
                                + filename
                                + "_"
                                + log_name
                                + "_noisy.wav",
                                norm * x_noisy_seg,
                                vds_full.fs,
                            )
                            soundfile.write(
                                "tmp_proc_audio/"
                                + filename
                                + "_"
                                + log_name
                                + "_enh.wav",
                                norm * x_proc_seg,
                                vds_full.fs,
                            )
                            break
                        except:
                            print("cannot write to audio file, trying again ...")
                            time.sleep(0.5)

                scores = compute_metrics(x_proc_seg, x_ref_seg, vds_full.fs, cols)
                metric_vals_df_list.append(
                    pd.DataFrame([chunk_info + scores], columns=cols_csv)
                )

            if len(metric_vals_df_list) > 0:
                metric_vals_df = pd.concat(metric_vals_df_list)
            else:
                metric_vals_df = pd.DataFrame([], columns=cols_csv)
            return metric_vals_df, cost, filename

        eg = enhance_generator()
        metric_dataframes_costs_filenames = Parallel(n_jobs=8)(
            delayed(val_metrics)(*tup)
            for tup in tqdm.tqdm(eg, total=vds_full.__len__())
        )

        for filename in validation_audio_samples_full:
            c, _ = soundfile.read(
                "tmp_proc_audio/" + filename + "_" + log_name + "_clean.wav"
            )
            os.remove("tmp_proc_audio/" + filename + "_" + log_name + "_clean.wav")
            n, _ = soundfile.read(
                "tmp_proc_audio/" + filename + "_" + log_name + "_noisy.wav"
            )
            os.remove("tmp_proc_audio/" + filename + "_" + log_name + "_noisy.wav")
            e, _ = soundfile.read(
                "tmp_proc_audio/" + filename + "_" + log_name + "_enh.wav"
            )
            os.remove("tmp_proc_audio/" + filename + "_" + log_name + "_enh.wav")
            tf.summary.audio(
                filename,
                tf.convert_to_tensor(
                    np.stack(
                        [c.astype("float32"), n.astype("float32"), e.astype("float32")],
                        axis=0,
                    )
                ),
                vds_full.fs,
                step=epoch,
            )

        costs = np.array([mcf[1] for mcf in metric_dataframes_costs_filenames])
        costs_datasets = np.array(
            [int(mcf[2][1]) for mcf in metric_dataframes_costs_filenames]
        )
        tf.summary.scalar(
            "val_loss/D2", np.nanmean(costs[costs_datasets == 2]), step=epoch
        )
        tf.summary.scalar(
            "val_loss/D3", np.nanmean(costs[costs_datasets == 3]), step=epoch
        )
        tf.summary.scalar(
            "val_loss/D4", np.nanmean(costs[costs_datasets == 4]), step=epoch
        )
        tf.summary.scalar("val_loss/mean", np.nanmean(costs), step=epoch)
        tf.summary.scalar(
            "val_loss/reliability", 1 - np.mean(np.isnan(costs)), step=epoch
        )

        full_metrics = pd.concat([mcf[0] for mcf in metric_dataframes_costs_filenames])
        datasets = np.array([int(f[1]) for f in full_metrics["file_name"].tolist()])
        for col in range(3, full_metrics.shape[1]):
            metric_name = full_metrics.columns[col]
            metric = full_metrics[metric_name].to_numpy()
            if metric_name == "PESQ (L)":
                pesql = np.nanmean(metric)
            elif metric_name == "PESQ (R)":
                pesqr = np.nanmean(metric)
            tf.summary.scalar(
                "%s/D2" % metric_name, np.nanmean(metric[datasets == 2]), step=epoch
            )
            tf.summary.scalar(
                "%s/D3" % metric_name, np.nanmean(metric[datasets == 3]), step=epoch
            )
            tf.summary.scalar(
                "%s/D4" % metric_name, np.nanmean(metric[datasets == 4]), step=epoch
            )
            tf.summary.scalar("%s/mean" % metric_name, np.nanmean(metric), step=epoch)
            tf.summary.scalar(
                "%s/reliability" % metric_name,
                1 - np.mean(np.isnan(metric)),
                step=epoch,
            )

        mean_pesq = (pesql + pesqr) / 2
        tf.summary.flush()

    np.random.seed()
    return mean_pesq


def save_best_checkpoint(epoch, best_mean_pesq):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mean_pesq": best_mean_pesq,
        },
        os.path.join(checkpoint_dir, "model_%s_best_epoch%d.pt" % (log_name, epoch)),
    )


def save_checkpoint(epoch, best_mean_pesq):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mean_pesq": best_mean_pesq,
        },
        os.path.join(checkpoint_dir, "model_%s_latest_epoch.pt" % log_name),
    )


if validation_only:
    num_epochs = 1

for epoch in range(1, num_epochs - epoch_offset + 1):
    for lrei in range(len(learning_rate_epochs)):
        if (epoch + epoch_offset) >= learning_rate_epochs[lrei]:
            current_learning_rate = learning_rates[lrei]

    if not validation_only:
        print("epoch %d, lr: %g" % (epoch + epoch_offset, current_learning_rate))
        for g in optimizer.param_groups:
            g["lr"] = current_learning_rate

        model.train()
        for sample in tqdm.tqdm(dl):
            (noisy, clean, doa, rms) = sample
            noisy = noisy.to("cuda")
            clean = clean.to("cuda")
            doa = doa.to("cuda")
            rms = rms.to("cuda")

            optimizer.zero_grad()

            enh = parallel_model(noisy, doa)

            cost = cost_function(clean, enh, rms)
            cost_mean = torch.mean(cost)
            cost_mean.backward()
            print(cost_mean.detach().cpu().numpy())
            optimizer.step()

    model.eval()
    if (epoch + epoch_offset) % validate_every == 0:
        mean_pesq = validate(epoch + epoch_offset)
        if mean_pesq > best_mean_pesq:
            print("Found new best epoch, PESQ: %g -> %g" % (best_mean_pesq, mean_pesq))
            best_mean_pesq = mean_pesq
            if not validation_only:
                save_best_checkpoint(epoch + epoch_offset, best_mean_pesq)

    if not validation_only:
        save_checkpoint(epoch + epoch_offset, best_mean_pesq)

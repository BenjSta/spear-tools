import toml

config = toml.load("configs/subfull.toml")
OUTPUT_DIR = "eval_processed_subfull"
CHKPT_PATH = "trained_model_checkpoints/model_subfull_latest_epoch.pt"


import torch
import os
import numpy as np
import soundfile
import tqdm
import glob
import parse
import sys

sys.path.append("spear-tools")
from models import nn_processor
import soundfile
from dataset import SpearDataset
import scipy.signal
from ptflops import get_model_complexity_info





os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


ds = SpearDataset(
    "spear-tools/analysis/spear_data/Main/Eval", [], 60, processing_hopsize / fs, 0.7, 0.6, fs
)

doa_hopsize = processing_hopsize / fs


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


checkpoint = torch.load(CHKPT_PATH)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
best_mean_pesq = checkpoint["best_mean_pesq"]
epoch_offset = checkpoint["epoch"]


import glob
from analysis.SPEAR import SPEAR_Data
import parse
import scipy.signal
import os
import tqdm
import soundfile
import numpy as np

sp = SPEAR_Data("spear-tools/analysis/spear_data/Main/Eval")

pathlist_eds1 = glob.glob("spear-tools/analysis/spear_data/Main/Eval/Dataset_1/DOA_sources/*/*.csv")
pathlist_eds2 = glob.glob("spear-tools/analysis/spear_data/Main/Eval/Dataset_2/DOA_sources/*/*.csv")
pathlist_eds3 = glob.glob("spear-tools/analysis/spear_data/Main/Eval/Dataset_3/DOA_sources/*/*.csv")
pathlist_eds4 = glob.glob("spear-tools/analysis/spear_data/Main/Eval/Dataset_4/DOA_sources/*/*.csv")
all_lists = pathlist_eds1 + pathlist_eds2 + pathlist_eds3 + pathlist_eds4


fs = 16000
doa_hopsize = 254 / fs
for f in tqdm.tqdm(all_lists):
    filename = os.path.split(f)[-1][4:-4]
    info = parse.parse("D{}_S{}_M{}_ID{}", filename)
    dataset = int(info[0])
    session = int(info[1])
    minute = info[2]
    target_id = int(info[3])
    # dataset_folder = os.path.join(foldername, 'Dataset %d' % dataset)
    # enhanced_folder = os.path.join(dataset_folder, 'Enhanced_Signals')
    session_folder = OUTPUT_DIR  # os.path.join(foldername, 'Session_%d' % session)
    # if not os.path.exists(dataset_folder):
    #    os.mkdir(dataset_folder)
    # if not os.path.exists(enhanced_folder):
    #    os.mkdir(enhanced_folder)
    if not os.path.exists(session_folder):
       os.mkdir(session_folder)

    processed_filename = "Enhanced_D%d_S%s_M%s_ID%d.wav" % (
        dataset,
        info[1],
        minute,
        target_id,
    )
    sp.set_file(dataset, session, filename[:-4])
    array_audio, fs_orig = sp.get_array_audio()
    if fs != fs_orig:
        array_audio = scipy.signal.resample_poly(array_audio, fs, fs_orig, axis=1)
    doa = sp.get_doa(target_id, np.arange(0, 0 + 60 + 2 * doa_hopsize, doa_hopsize))

    enh = np.zeros_like(array_audio.T[:, -2:])
    noisy_resampled = torch.from_numpy(array_audio.astype("float32").T).to(device)
    doa = torch.from_numpy(doa.astype("float32")).to(device)
    # process in segments, otherwise would be too expensive
    doa_fs = 1 / doa_hopsize
    SEGHOP_DOA = int(8.5 * doa_fs) / doa_fs
    SEGLEN_DOA = int(10 * doa_fs) / doa_fs
    if fs != fs_orig:
        doa_hopsize_samples = fs / doa_fs
        assert (
            np.abs(doa_hopsize_samples - np.round(doa_hopsize_samples)) < 1e-9
        )  # ignore rounding error
        doa_hopsize_samples = int(np.round(doa_hopsize_samples))
    else:
        doa_hopsize_samples = processing_hopsize

    num_segments = int(np.ceil((doa.shape[0] / doa_fs - SEGLEN_DOA) / SEGHOP_DOA + 1))

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

    if fs != fs_orig:
        processed_upsampled = scipy.signal.resample_poly(enh, fs_orig, fs, axis=0)
    else:
        processed_upsampled = enh.copy()

    soundfile.write(
        os.path.join(session_folder, processed_filename),
        processed_upsampled,
        fs_orig,
        subtype="FLOAT",
    )

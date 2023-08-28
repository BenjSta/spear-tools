VISIBLE_DEVICES = '0'
import os
import numpy as np
import tqdm
import glob
import toml
VAL_SEGMENTS_FILE = 'analysis/segments_Dev.csv'
from models.nn_processor import STFTFeats
from torch.utils.data import DataLoader
from dataset import SpearDataset

os.environ['CUDA_VISIBLE_DEVICES'] = VISIBLE_DEVICES
config = toml.load('config_subfull_inverse_sigma.toml')
net_config = config['net_config']

processing_winlen = config['processing_winlen']
processing_hopsize = config['processing_hopsize']
fs = config['fs']
duration = config['duration']
batch_size = config['batch_size']
resume = config['resume']
num_epochs = config['num_epochs']
num_workers_loader = config['num_workers_loader']
tensorboard_logdir = config['tensorboard_logdir']
checkpoint_dir = config['checkpoint_dir']
log_name = config['log_name']

cost_function_winlen = config['cost_function_winlen']
cost_function_hopsize = config['cost_function_hopsize']
cost_function_alpha = config['cost_function_alpha']


validate_every = config['validate_every']
validation_num_audio_samples = config['validation_num_audio_samples']
validation_seed = config['validation_seed']
device = config['device']

pathlist_ds2 = glob.glob('analysis/spear_data/Extra/Train/Dataset_2/Reference_Audio/*/*/ref_*.wav')
pathlist_ds3 = glob.glob('analysis/spear_data/Extra/Train/Dataset_3/Reference_Audio/*/*/ref_*.wav')
pathlist_ds4 = glob.glob('analysis/spear_data/Extra/Train/Dataset_4/Reference_Audio/*/*/ref_*.wav')
flist = sorted([os.path.split(p)[-1][4:-4] for p in pathlist_ds2 + pathlist_ds3 + pathlist_ds4])
print('train set size: %d minutes' % len(flist))

pathlist_vds2 = glob.glob('analysis/spear_data/Extra/Dev/Dataset_2/Reference_Audio/*/*/ref_*.wav')
pathlist_vds3 = glob.glob('analysis/spear_data/Extra/Dev/Dataset_3/Reference_Audio/*/*/ref_*.wav')
pathlist_vds4 = glob.glob('analysis/spear_data/Extra/Dev/Dataset_4/Reference_Audio/*/*/ref_*.wav')
vflist = sorted([os.path.split(p)[-1][4:-4] for p in pathlist_vds2 + pathlist_vds3 + pathlist_vds4])
print('val set size: %d minutes' % len(vflist))

np.random.seed(validation_seed)
validation_audio_samples = np.random.randint(len(vflist), size=validation_num_audio_samples)
np.random.seed(validation_seed)
vflist_reduced = vflist#np.random.choice(vflist, size=10, replace=False)
validation_audio_samples_full = np.random.choice(vflist_reduced, size=validation_num_audio_samples, replace=False)
np.random.seed()

ds = SpearDataset('analysis/spear_data/Main/Train', flist, duration, processing_hopsize / fs, 0.7, 0.6, fs)
dl = DataLoader(ds, batch_size, shuffle=True, num_workers = num_workers_loader, drop_last=True)

#%%
model = STFTFeats(ds.get_ATFs(), net_config, fs, processing_winlen, processing_hopsize, np.random.randn(1))
model = model.to('cuda')
#%%
feats = []
for batch in tqdm.tqdm(dl):
    (noisy, clean, doa, rms) = batch
    noisy = noisy.to('cuda')
    clean = clean.to('cuda')
    doa = doa.to('cuda')
    rms = rms.to('cuda')
    feats.append(model(noisy, doa).detach().cpu().numpy())
# %%
compression = 0.3
var_all = []
mean_abs_all = []
for f in tqdm.tqdm(feats):
    var_all.append(np.mean((np.abs(f)**compression)**2, axis=(1, 3)))
    mean_abs_all.append(np.mean((np.abs(f)**compression), axis=(1, 3)))

#%%
var_all_fi = []
mean_abs_all_fi = []
for f in tqdm.tqdm(feats):
    var_all_fi.append(np.mean(np.abs(f)**2, axis=(1, 2, 3)))
    mean_abs_all_fi.append(np.mean(np.abs(f), axis=(1, 2, 3)))


# %%
var_all = np.concatenate(var_all, axis=0)
mean_abs_all = np.concatenate(mean_abs_all, axis=0)
# %%
var_mean = np.mean(var_all, axis=0) / 2
mean_abs = np.mean(mean_abs_all, axis=0)
# %%
std = np.sqrt(var_mean)

# %%
var_all_fi = np.concatenate(var_all_fi, axis=0)
mean_abs_all_fi = np.concatenate(mean_abs_all_fi, axis=0)

#%%
std_fi = np.sqrt(np.mean(var_all_fi) / 2)
mean_abs_fi = np.mean(mean_abs_all)

# %%
np.save('laplace_sigma_fs%g_blk%d_compressed%g.npy'%(fs, processing_winlen, compression), mean_abs)
np.save('gaussian_sigma_fs%g_blk%d_compressed%g.npy'%(fs, processing_winlen, compression), std)
# %%

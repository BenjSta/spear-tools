#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
#torch.set_num_threads(1)
from models.nn_processor import FeatureExtractorMatchedFilterMaxDir
from models.conv_recurrent import GCRN
from models.conv_recurrent import IGCRN
from torch import nn
import torch.nn.functional as F
import time

#%%
class OnlineCausalConvBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()
        # conv = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     groups = groups,
        #     bias=bias
        # )
        self.kernel_size = conv.kernel_size

        if self.kernel_size[0]<2:
            self.kernel2 = torch.nn.parameter.Parameter(conv.weight[:, :, 0, :], requires_grad=True)
        else:
            self.kernel1 = torch.nn.parameter.Parameter(conv.weight[:, :, 0, :], requires_grad=True)
            self.kernel2 = torch.nn.parameter.Parameter(conv.weight[:, :, 1, :], requires_grad=True)

        if conv.bias is None:
            self.bias_flag = False
            self.bias = None
        else:
            self.bias_flag = True
            
            self.bias = torch.nn.parameter.Parameter(conv.bias, requires_grad=True)
        self.groups = conv.groups
        self.stride = conv.stride
        self.padding = (conv.padding[1],)
        self.pad = conv.padding 


    def forward(self, x, x_prev):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F]
        Returns:
            [B, C, F]
        """

        if self.kernel_size[0]<2:
            conv2_out = F.conv1d(x, self.kernel2, stride=self.stride[1], padding=self.padding, groups=self.groups)
            conv1_out = torch.zeros(conv2_out.size())
        else:
            if (x_prev is None):# or (x_prev.shape[0]==0):
                x_prev = torch.zeros(x.size())
            conv1_out = F.conv1d(x_prev, self.kernel1, stride=self.stride[1], padding=self.padding, groups=self.groups)
            conv2_out = F.conv1d(x, self.kernel2, stride=self.stride[1], padding=self.padding, groups=self.groups)
#        conv_out = F.conv2d(torch.concat((x_prev.unsqueeze(dim=2),x.unsqueeze(dim=2)),dim=2), torch.concat((self.kernel1.unsqueeze(dim=2),self.kernel2.unsqueeze(dim=2)), dim=2), bias = self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        if self.bias_flag:
            y = conv1_out + conv2_out + self.bias[None, :, None]
#            y_2d = conv_out + self.bias[None,:,None]
        else:
            y = conv1_out + conv2_out
#            y_2d = conv_out

        x_prev = x

        return y, x_prev#, self.kernel1, self.kernel2


class OnlineCausalTransConvBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()

        # conv = nn.ConvTranspose2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     output_padding=output_padding,
        #     groups=groups,
        #     bias=bias
        # )
        self.kernel_size = conv.kernel_size
        if self.kernel_size[0]<2:
            self.kernel2 = torch.nn.parameter.Parameter(conv.weight[:, :, 0, :], requires_grad=True)
        else:
            self.kernel1 = torch.nn.parameter.Parameter(conv.weight[:, :, 1, :], requires_grad=True)
            self.kernel2 = torch.nn.parameter.Parameter(conv.weight[:, :, 0, :], requires_grad=True)
        
        if conv.bias is None:
            self.bias_flag = False
        else:
            self.bias_flag = True
            self.bias = torch.nn.parameter.Parameter(conv.bias, requires_grad=True)

        self.groups = conv.groups
        self.stride = conv.stride
        self.padding = (conv.padding[1],)
        self.output_padding = (conv.output_padding[1],)


    def forward(self, x, x_prev):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F]
        Returns:
            [B, C, F]
        """



        if self.kernel_size[0]<2:
            conv2_out = F.conv_transpose1d(x, self.kernel2, stride=self.stride[1], padding=self.padding, groups=self.groups, output_padding=self.output_padding)
            conv1_out = torch.zeros(conv2_out.size())
        else:
            if (x_prev is None):# or (x_prev.shape[0]==0):
                x_prev = torch.zeros(x.size())
            conv1_out = F.conv_transpose1d(x_prev, self.kernel1, stride=self.stride[1], padding=self.padding, groups=self.groups, output_padding=self.output_padding)
            conv2_out = F.conv_transpose1d(x, self.kernel2, stride=self.stride[1], padding=self.padding, groups=self.groups, output_padding=self.output_padding)

        if self.bias_flag:
            y = conv1_out + conv2_out + self.bias[None, :, None]
        else:
            y = conv1_out + conv2_out

        x_prev = x

        return y, x_prev


class OnlineCausalGatedConvBlock(nn.Module):
    def __init__(self, cg_block):
        super().__init__()
        self.conv = OnlineCausalConvBlock(cg_block.conv)
        
        self.gate = cg_block.gate
        if cg_block.gate:
            self.conv_gate = OnlineCausalConvBlock(cg_block.conv_gate)

        #if cg_block.bn:
        #    raise NotImplementedError('batch norm not implemented yet')

        self.elu = cg_block.elu


    def forward(self, x, x_prev):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F]
        Returns:
            [B, C, F]
        """
        if self.gate:
            g, _ = self.conv_gate(x, x_prev)
            xg = torch.sigmoid(g)
            x, x_prev = self.conv(x, x_prev)
            
            
            return self.elu(xg * x), x_prev

        else:
            x, x_prev = self.conv(x, x_prev)
            return self.elu(x), x_prev

class OnlineCausalTransGatedConvBlock(nn.Module):
    def __init__(self, cg_block):

        super().__init__()
        
        self.conv = OnlineCausalTransConvBlock(cg_block.conv)
        self.gate = cg_block.gate
        if cg_block.gate:
            self.conv_gate = OnlineCausalTransConvBlock(cg_block.conv_gate)
        self.elu = cg_block.elu
        #if cg_block.bn:
        #    raise NotImplementedError('batch norm not implemented yet')



    def forward(self, x, x_prev):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        if self.gate:
            g, _ = self.conv_gate(x, x_prev)
            xg = torch.sigmoid(g)
            x, x_prev = self.conv(x, x_prev)
            return self.elu(xg * x), x_prev
        else:
            x, x_prev = self.conv(x, x_prev)
            return self.elu(x), x_prev



class OnlineGCRN(nn.Module):
    def __init__(self, gcrn):
        
        super().__init__()
        
        #self.gcrn = gcrn
        self.encoder_blocks = nn.ModuleList()
        self.encoder_block_nonlins = nn.ModuleList()
        for e in gcrn.encoder_blocks:
            for ei, ec in enumerate(e.children()):
                if ei == 0:
                    self.encoder_blocks.append(OnlineCausalGatedConvBlock(ec))
                if ei == 1:
                    self.encoder_block_nonlins.append(ec)

        self.decoders = nn.ModuleList()
        self.decoders_nonlins = nn.ModuleList()
        for d in gcrn.decoders:
            decoder_blocks = nn.ModuleList()
            decoder_blocks_nonlins = nn.ModuleList()
            for db in d:
                for di, dc in enumerate(db.children()):
                    if di == 0:
                        decoder_blocks.append(OnlineCausalTransGatedConvBlock(dc))
                    if di == 1:
                        decoder_blocks_nonlins.append(dc)

            self.decoders.append(decoder_blocks)
            self.decoders_nonlins.append(decoder_blocks_nonlins)

        self.group_gru = gcrn.group_gru



    def forward(self, x, x_prev_enc, x_prev_dec, h_prev_gru):
        encoder_outputs = []
        x_prev_enc_out = []
        for i, (l, ln) in enumerate(zip(self.encoder_blocks, self.encoder_block_nonlins)):
            
            x, x_prev = l(x, x_prev_enc[i])
            x_prev_enc_out.append(x_prev)
            x = ln(x)
            encoder_outputs.append(x)

        batch_size, n_channels, n_bins = x.shape

        #print(x[0, 0, 100:110, 20])
        # x = x.permute(0, 1, 2)
        x = x.reshape(batch_size, n_channels * n_bins)
        x, h_prev_gru  = self.group_gru(x[:, None, :], h_prev_gru)
        x = x[:, 0, :]

        r = x.reshape(batch_size, n_channels, n_bins)

        decoder_outputs = []
        x_prev_dec_out = []
        
        decoder_outputs = []
        for i, (d, dn) in enumerate(zip(self.decoders, self.decoders_nonlins)):
            x = r
            x_prev_decb_out = []
            for j, (db, dbn) in enumerate(zip(d, dn)):
                x, x_prev = db(torch.cat([x, encoder_outputs[-1-j]], 1), x_prev_dec[i][j])
                x = dbn(x)
                x_prev_decb_out.append(x_prev)
            x_prev_dec_out.append(x_prev_decb_out)
            decoder_outputs.append(x)

        return decoder_outputs, x_prev_enc_out, x_prev_dec_out, h_prev_gru
    
class SubFullRealtime(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self, air, fs, processing_winlen, processing_hopsize, subfull
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )
        self.net1 = subfull.net1
        self.net2 = OnlineGCRN(subfull.net2)
        self.sigma = torch.nn.Parameter(subfull.sigma, requires_grad=False)
        self.remult_sigma = subfull.remult_sigma

    def forward(self, noisy_spec, doa, hr_prev_igcrn, x_prev_enc, x_prev_dec, h_prev_gru):
        #noisy_spec: Bx1xFxCx2
        #noisy_spec = torch.view_as_complex(noisy_spec)
        x, x_bf, binaural_weights = self.apply_beamformer(noisy_spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # Bx1xFxC
        feat_spec = torch.concat(
           [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCx1xF
        feat_spec = feat_spec / self.sigma[None, None, None, :]

        subband_output, hr_prev_igcrn = self.net1(feat_spec, hr_prev_igcrn)
        out, x_prev_enc, x_prev_dec, h_prev_gru = self.net2(torch.concat([feat_spec, subband_output[0]], dim=1)[:, :, 0, :], 
                         x_prev_enc, x_prev_dec, h_prev_gru)
        r = out[0][:, :, None, :]
        i = out[1][:, :, None, :]
        #enhanced_single_chan = torch.stack([r, i], -1)
        enhanced_single_channel_spec = torch.complex(r, i).permute(0, 2, 3, 1)

        if self.remult_sigma:
            enhanced_single_channel_spec = (
                enhanced_single_channel_spec * self.sigma[None, None, :, None]
            )

        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        #enhanced_binaural_spec = torch.view_as_real(enhanced_binaural_spec)
        return enhanced_binaural_spec, hr_prev_igcrn, x_prev_enc, x_prev_dec, h_prev_gru

#%%
import toml

METRICS = ["SI-SDR", "PESQ"]  # choice of metrics to run
config = toml.load("configs/config_subfull.toml")
CHECKPOINT_PATH = (
    "/media/DATA/shared/stahl/spear_tools/trained_models/experiment3/checkpoints/model_subfull_inverse_sigma_best_epoch2300.pt"
)
VAL_SEGMENTS_FILE = "analysis/segments_Dev.csv"
WRITE_PROCESSED_AUDIO = False
NUM_WORKERS_METRICS = 20
NUM_WORKERS_LOADER = 20
DEVICE = "cpu"
import os



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
net_config = config["net_config"]
processing_winlen = config["processing_winlen"]
processing_hopsize = config["processing_hopsize"]
fs = config["fs"]

pathlist_vds2 = glob.glob(
    "analysis/spear_data/Extra/Dev/Dataset_2/Reference_Audio/*/*/ref_*.wav"
)
pathlist_vds3 = glob.glob(
    "analysis/spear_data/Extra/Dev/Dataset_3/Reference_Audio/*/*/ref_*.wav"
)
pathlist_vds4 = glob.glob(
    "analysis/spear_data/Extra/Dev/Dataset_4/Reference_Audio/*/*/ref_*.wav"
)
vflist = sorted(
    [os.path.split(p)[-1][4:-4] for p in pathlist_vds2 + pathlist_vds3 + pathlist_vds4]
)
print("val set size: %d minutes" % len(vflist))


vds_full = SpearDataset(
    "analysis/spear_data/Main/Dev", vflist, 60, processing_hopsize / fs, 0.0, 0.0, fs
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
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    best_mean_pesq = checkpoint["best_mean_pesq"]
    epoch_offset = checkpoint["epoch"]

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

model.eval()
# %%
rtmodel = SubFullRealtime(vds_full.get_ATFs(), fs,
                          processing_winlen, processing_hopsize, model)
# %%
print(rtmodel)



#%%
noisy_spec = torch.randn((1, 1, 255, 6)) + 1j * torch.randn((1, 1, 255, 6))
doa = torch.randn((1, 1, 2))

x_prev1_enc = torch.zeros((1, 16, 255))
x_prev2_enc = torch.zeros((1, 64, 127))
x_prev3_enc = torch.zeros((1, 64, 63))
x_prev4_enc = torch.zeros((1, 64, 31))

x_prev1_dec1 = torch.zeros((1, 128, 15))
x_prev2_dec1 = torch.zeros((1, 128, 31))
x_prev3_dec1 = torch.zeros((1, 128, 63))
x_prev4_dec1 = torch.zeros((1, 128, 127))

x_prev1_dec2 = torch.zeros((1, 128, 15))
x_prev2_dec2 = torch.zeros((1, 128, 31))
x_prev3_dec2 = torch.zeros((1, 128, 63))
x_prev4_dec2 = torch.zeros((1, 128, 127))

x_prev_enc = [x_prev1_enc, x_prev2_enc, x_prev3_enc, x_prev4_enc]
x_prev_dec = [[x_prev1_dec1, x_prev2_dec1, x_prev3_dec1, x_prev4_dec1], 
              [x_prev1_dec2, x_prev2_dec2, x_prev3_dec2, x_prev4_dec2]]

#%%
rtmodel = rtmodel.to('cpu')
hr_igcrn = torch.zeros((2, 255, 75))
h_ggru = torch.zeros((8, 1, 240))

#%%
noisy, _, doa, _ = iter(vdl_full).__next__()
noisy_spec = rtmodel.stft(noisy)


#%%

with torch.no_grad():
    start = time.perf_counter()
    enh_all = []
    for _ in range(63):
        enh, hr_igcrn, x_prev_enc, x_prev_dec, h_ggru = \
            rtmodel.forward(noisy_spec, doa, hr_igcrn, x_prev_enc, x_prev_dec, h_ggru)
        
    print((time.perf_counter() - start) / 10)

# #%%
# with torch.no_grad():
#     start = time.perf_counter()
#     enh_all = []
#     for f in range(noisy_spec.shape[1]):
#         n = noisy_spec[:, f,None, :, :]
#         d = doa[:, f, None, :]

#         enh, hr_igcrn, x_prev_enc, x_prev_dec, h_ggru = \
#             rtmodel.forward(n, d, hr_igcrn, x_prev_enc, x_prev_dec, h_ggru)
#         enh_all.append(enh)

# enh = torch.cat(enh_all, 1)
# #%%
# enh_td = rtmodel.istft(enh, noisy.shape[1])


# #%%
# model = model.to('cpu')
# enh_td_offline = model(noisy, doa)

# #%%
# def constr(sizes):
#     return {'feat_spec': torch.ones(sizes[0]), 
#      'hr_prev_igcrn': torch.ones(sizes[1]), 
#      'x_prev_enc': [torch.ones(sz) for sz in sizes[2]],
#      'x_prev_dec': [[torch.ones(sz) for sz in sizes[3][0]], [torch.ones(sz) for sz in sizes[3][1]]],
#      'h_prev_gru': torch.ones(sizes[4]), 
#     }

# macs, params = get_model_complexity_info(
#     rtmodel,
#     ((1, 14, 1, 255), (2, 255, 75), ((1, 16, 255), (1, 64, 127), (1, 64, 63), (1, 64, 31)),
#                                        (((1, 128, 15), (1, 128, 31), (1, 128, 63), (1, 128, 127)),  
#                                        ((1, 128, 15), (1, 128, 31), (1, 128, 63), (1, 128, 127))), 
#                                        (8, 1, 240)),
#     input_constructor=constr,
#     as_strings=True,
#     print_per_layer_stat=True,
#     verbose=True)
# print("{:<30}  {:<8}".format("Computational complexity: ", macs))
# print("{:<30}  {:<8}".format("Number of parameters: ", params))

        

# #%%
# torch.onnx.export(rtmodel, args=(feat_spec, hr_igcrn, x_prev_enc, x_prev_dec, h_ggru),
#                   f="subfull_spear.onnx", verbose=True, input_names=['feat',
#                   'hr_igcrn', 'x_prev_enc1', 'x_prev_enc2', 'x_prev_enc3', 'x_prev_enc4',
#                   'x_prev_dec11', 'x_prev_dec12', 'x_prev_dec13', 'x_prev_dec14',
#                   'x_prev_dec21', 'x_prev_dec22', 'x_prev_dec23', 'x_prev_dec24',
#                   'hr_ggru'], output_names=['enh',
#                   'o_hr_igcrn', 'o_x_prev_enc1', 'o_x_prev_enc2', 'o_x_prev_enc3', 'o_x_prev_enc4',
#                   'o_x_prev_dec11', 'o_x_prev_dec12', 'o_x_prev_dec13', 'o_x_prev_dec14',
#                   'o_x_prev_dec21', 'o_x_prev_dec22', 'o_x_prev_dec23', 'o_x_prev_dec24',
#                   'o_hr_ggru'], opset_version=16)
# # %%

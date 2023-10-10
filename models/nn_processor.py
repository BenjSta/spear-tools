import numpy as np
import torch
import sys
from models.torch_sigproc import (
    MultichannelISTFTLayer,
    MultichannelSTFTLayer,
    sqrt_hann_win_fn,
)
sys.path.append('./spear-tools/')
from analysis.Processor import SPEAR_Processor
import sys
import FullSubNet.recipes.dns_interspeech_2020.inferencer as fullsub_inf
import toml


class FeatureExtractorMatchedFilterMaxDir(torch.nn.Module, SPEAR_Processor):
    def __init__(self, air, fs, processing_winlen, processing_hopsize):
        torch.nn.Module.__init__(self)
        self.stft_params = {
            "winL": processing_winlen / fs,
            "stepL": processing_hopsize / fs,
        }
        self.winlen = int(self.stft_params["winL"] * fs)
        self.hopsize = int(self.stft_params["stepL"] * fs)

        SPEAR_Processor.__init__(self, air, fs=fs, mics=[], out_chan=[5, 6])

        self.dirs_tensor = torch.nn.parameter.Parameter(
            torch.from_numpy(self.dirs.astype("float32")), requires_grad=False
        )

        self.stft = MultichannelSTFTLayer(self.winlen, self.hopsize, sqrt_hann_win_fn)
        self.istft = MultichannelISTFTLayer(self.winlen, self.hopsize, sqrt_hann_win_fn)

    def apply_beamformer(self, X, target_doas):
        # select correct beamformers
        target_doas = (
            np.pi / 180 * target_doas[:, : X.shape[1], :]
        )  # drop unneeded frame doas
        nBatch, nFrame, nFreq, nChan = X.shape  # batch, time, freq, chan
        nOut = len(self.out_chan)
        azi_diff = (
            target_doas[..., 0][:, :, None] - self.dirs_tensor[..., 0][None, None, :]
        )
        zenith_diff = (
            target_doas[..., 1][:, :, None] - self.dirs_tensor[..., 1][None, None, :]
        )
        a = (
            torch.sin(zenith_diff / 2) ** 2
            + torch.cos(target_doas[..., 1][:, :, None])
            * torch.cos(self.dirs_tensor[..., 1][None, None, :])
            * torch.sin(azi_diff / 2) ** 2
        )
        angle_diff = 2 * torch.arcsin(torch.sqrt(a))
        ind = torch.argmin(angle_diff, dim=-1).detach().cpu().numpy()
        w_conj = torch.from_numpy(self.w_conj_tensor[:, :, ind]).to(
            X.device
        )  # ch, freq, batch, time
        w_conj = w_conj.permute(2, 3, 1, 0)
        w_conj_ds = torch.from_numpy(self.w_conj_ds_tensor[:, :, ind]).to(
            X.device
        )  # ch, freq, batch, time
        w_conj_ds = w_conj_ds.permute(2, 3, 1, 0)  # batch, time, freq, ch

        # apply beamformer / delay compensation
        ds_multichannel_out = w_conj_ds * X

        bf_out = torch.sum(w_conj * X, axis=-1)

        binaural_weights = torch.from_numpy(self.w_binaural_tensor[:, :, ind]).to(
            X.device
        )
        binaural_weights = binaural_weights.permute(2, 3, 1, 0)  # batch, time, freq, ch

        return ds_multichannel_out, bf_out, binaural_weights

    def prepare_iso_weights(self):
        # override superfunction to also save DS weights and convert to non-trainable net parameter
        nFreq = self.nFreq
        nChan = self.nChan
        nOut = self.nOut
        nDir = self.nDir
        outChan_ind = np.array(self.out_chan) - 1
        w_conj = np.zeros((nChan, nFreq, nDir), dtype=np.complex64)
        w_conj_ds = np.zeros((nChan, nFreq, nDir), dtype=np.complex64)
        w_binaural = np.zeros((nOut, nFreq, nDir), dtype=np.complex64)
        print_cycle = 50  # number of interations to wait before updating progress print (keep it high as the loop is fast)

        VIRTUAL_ORIGIN_TIME_OF_FLIGHT = 0.0007
        time_centering_phasor = np.exp(
            -1j
            * 2
            * np.pi
            * np.arange(nFreq)
            / self.winlen
            * self.fs
            * VIRTUAL_ORIGIN_TIME_OF_FLIGHT
        )

        for di in range(nDir):
            if di % print_cycle == 0:
                print("\r", "Preparing weights %%%2.2f " % (100 * di / nDir), end="\r")
            h = np.squeeze(self.ATF[:, di, :])  #  unnormalized RTF [nFreq x nChan]
            for fi in range(nFreq):
                R = np.squeeze(self.R_iso[fi, :, :])  # [nChan x nChan]
                rtf = h[fi, :] * np.conj(time_centering_phasor[fi])

                for oi in range(nOut):
                    w_binaural[oi, fi, di] = rtf[outChan_ind[oi]].reshape(-1, 1)

                w_conj_ds[:, fi, di] = np.conj(self.get_ds_weights(rtf))
                w_conj[:, fi, di] = np.conj(self.get_mvdr_weights(R, rtf))

        self.w_conj_tensor = w_conj.astype("complex64")
        self.w_conj_ds_tensor = w_conj_ds.astype("complex64")
        self.w_binaural_tensor = w_binaural.astype("complex64")

        print("\r", "Preparing weights %%%2.2f - Done!" % (100), end="\n")

    def get_ds_weights(self, d):
        # DESCRIPTION: calculates the MVDR weights
        # *** INPUTS ***
        # R  (ndarray) covariance matrix [nChan x nChan]
        # d  (ndarray) steering vector or Relative Transfer Function (RTF) [nChan x 1]
        # *** OUTPUTS ***
        # w  (ndarray) beamformer conjugate weights in the stft domain  [nChan x 1]
        w = d / np.matmul(np.conj(d).T, d)
        return w


class MaxDirAndFullsubnet(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self, air, net_config, fs, processing_winlen, processing_hopsize, sigma
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )
        self.conf = toml.load(
            "FullSubNet/recipes/dns_interspeech_2020/fullsubnet/inference.toml"
        )
        self.inferencer = fullsub_inf.Inferencer(
            self.conf,
            "FullSubNet/fullsubnet_best_model_58epochs.tar",
            "./dummy_dir",
        )
        # self.demucs = master64()
        # self.df_model, self.df_state, _ = init_df('./dfnet2_pretrained')  # Load default model

    def forward(self, noisy, doa):
        assert noisy.shape[0] == 1  # batch size has to be one
        spec = self.stft(noisy)
        _, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        x_bf_td = self.istft(x_bf[..., None], noisy.shape[1])
        # x_bf_td = resample(x_bf_td[:, :, 0], self.fs, 48000)[:, :, None]
        # dfnet2_out_td = enhance(self.df_model, df_state=self.df_state, audio=x_bf_td[:, :, 0].to('cpu')).to(x_bf_td.device)
        # dfnet2_out_td = resample(dfnet2_out_td, 48000, self.fs)
        fullsubnet_out = getattr(self.inferencer, "full_band_crm_mask")(
            x_bf_td[:, :, 0], {"n_neighbor": 15}
        )
        # dfnet2_out_td = self.demucs(x_bf_td.permute(0, 2, 1)).permute(0, 2, 1)
        enh_stft = self.stft(fullsubnet_out[:, :, None])
        return self.istft(
            enh_stft * binaural_weights[:, : enh_stft.shape[1], :], noisy.shape[1]
        )


class STFTFeats(FeatureExtractorMatchedFilterMaxDir):
    def forward(self, noisy, doa):
        spec = self.stft(noisy)
        x, x_bf = self.apply_beamformer(spec, doa)
        return x


class SubFull(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self,
        air,
        net_config,
        fs,
        processing_winlen,
        processing_hopsize,
        sigma,
        remult_sigma=True,
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )

        self.subband_num_outputs = net_config["subband_config"]["num_channels"]
        self.net1 = IGCRN(**net_config["subband_config"])
        net_config["fullband_config"]["num_channels_encoder"][0] = (
            net_config["subband_config"]["num_output_channels"] + 14
        )
        self.net2 = GCRN(**net_config["fullband_config"])

        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma), requires_grad=False)
        self.remult_sigma = remult_sigma

    def forward(self, noisy, doa, ref_mono=None):
        spec = self.stft(noisy)

        x, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # BxTxFxC
        feat_spec = torch.concat(
            [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCxTxF
        feat_spec = feat_spec / self.sigma[None, None, None, :]

        subband_output = self.net1(feat_spec)
        r, i = self.net2(torch.concat([feat_spec, subband_output[0]], dim=1))
        enhanced_single_channel_spec = torch.complex(r, i).permute(0, 2, 3, 1)

        if self.remult_sigma:
            enhanced_single_channel_spec = (
                enhanced_single_channel_spec * self.sigma[None, None, :, None]
            )

        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        enhanced_binaural = self.istft(enhanced_binaural_spec, noisy.shape[1])

        if ref_mono == None:
            return enhanced_binaural
        else:
            ref_spec = self.stft(ref_mono)
            ref_binaural = self.istft(ref_spec * binaural_weights, noisy.shape[1])
            return enhanced_binaural, ref_binaural


class SubFullMapMaskPhase(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self,
        air,
        net_config,
        fs,
        processing_winlen,
        processing_hopsize,
        sigma,
        remult_sigma=True,
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )

        self.subband_num_outputs = net_config["subband_config"]["num_channels"]
        self.net1 = IGCRN(**net_config["subband_config"])
        net_config["fullband_config"]["num_channels_encoder"][0] = (
            net_config["subband_config"]["num_output_channels"] + 14
        )
        self.net2 = GCRN(**net_config["fullband_config"])

        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma), requires_grad=False)
        self.remult_sigma = remult_sigma

        fsize = self.winlen // 2 + 1
        self.mask_layer = torch.nn.Sequential(
            torch.nn.Linear(fsize, fsize), torch.nn.Sigmoid()
        )
        self.map_layer = torch.nn.Sequential(
            torch.nn.Linear(fsize, fsize), torch.nn.ReLU()
        )
        self.phase_r_layer = torch.nn.Linear(fsize, fsize)
        self.phase_i_layer = torch.nn.Linear(fsize, fsize)

    def forward(self, noisy, doa, ref_mono=None):
        spec = self.stft(noisy)

        x, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # BxTxFxC
        feat_spec = torch.concat(
            [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCxTxF
        feat_spec = feat_spec / self.sigma[None, None, None, :]

        subband_output = self.net1(feat_spec)
        map_mask, phase = self.net2(torch.concat([feat_spec, subband_output[0]], dim=1))
        phasor_unnormed = torch.complex(
            self.phase_r_layer(phase[:, 0, :, :]), self.phase_i_layer(phase[:, 1, :, :])
        )
        phasor = torch.exp(1j * torch.angle(phasor_unnormed))
        if self.remult_sigma:
            mag = self.map_layer(map_mask[:, 0, :, :]) + self.mask_layer(
                map_mask[:, 1, :, :]
            ) * (torch.abs(x_bf) / self.sigma[None, None, :])
        else:
            mag = self.map_layer(map_mask[:, 0, :, :]) + self.mask_layer(
                map_mask[:, 1, :, :]
            ) * torch.abs(x_bf)
        enhanced_single_channel_spec = (mag * phasor)[..., None]
        if self.remult_sigma:
            enhanced_single_channel_spec = (
                enhanced_single_channel_spec * self.sigma[None, None, :, None]
            )

        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        enhanced_binaural = self.istft(enhanced_binaural_spec, noisy.shape[1])

        if ref_mono == None:
            return enhanced_binaural
        else:
            ref_spec = self.stft(ref_mono)
            ref_binaural = self.istft(ref_spec * binaural_weights, noisy.shape[1])
            return enhanced_binaural, ref_binaural


class Fullband(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self,
        air,
        net_config,
        fs,
        processing_winlen,
        processing_hopsize,
        sigma,
        remult_sigma=True,
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )
        self.net2 = GCRN(**net_config["fullband_config"])

        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma), requires_grad=False)
        self.remult_sigma = remult_sigma

    def forward(self, noisy, doa, ref_mono=None):
        spec = self.stft(noisy)

        x, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # BxTxFxC
        feat_spec = torch.concat(
            [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCxTxF
        feat_spec = feat_spec / self.sigma[None, None, None, :]

        r, i = self.net2(feat_spec)
        enhanced_single_channel_spec = torch.complex(r, i).permute(0, 2, 3, 1)
        if self.remult_sigma:
            enhanced_single_channel_spec = (
                enhanced_single_channel_spec * self.sigma[None, None, :, None]
            )

        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        enhanced_binaural = self.istft(enhanced_binaural_spec, noisy.shape[1])

        if ref_mono == None:
            return enhanced_binaural
        else:
            ref_spec = self.stft(ref_mono)
            ref_binaural = self.istft(ref_spec * binaural_weights, noisy.shape[1])
            return enhanced_binaural, ref_binaural


class Subband(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self,
        air,
        net_config,
        fs,
        processing_winlen,
        processing_hopsize,
        sigma,
        remult_sigma=True,
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )

        self.net2 = IGCRN(**net_config["subband_config"])

        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma), requires_grad=False)
        self.remult_sigma = remult_sigma

    def forward(self, noisy, doa, ref_mono=None):
        spec = self.stft(noisy)

        x, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # BxTxFxC
        feat_spec = torch.concat(
            [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCxTxF
        if self.remult_sigma:
            feat_spec = feat_spec / self.sigma[None, None, None, :]

        r, i = self.net2(feat_spec)
        enhanced_single_channel_spec = torch.complex(r, i).permute(0, 2, 3, 1)
        enhanced_single_channel_spec = (
            enhanced_single_channel_spec * self.sigma[None, None, :, None]
        )
        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        enhanced_binaural = self.istft(enhanced_binaural_spec, noisy.shape[1])

        if ref_mono == None:
            return enhanced_binaural
        else:
            ref_spec = self.stft(ref_mono)
            ref_binaural = self.istft(ref_spec * binaural_weights, noisy.shape[1])
            return enhanced_binaural, ref_binaural


class SubbandMapMaskPhase(FeatureExtractorMatchedFilterMaxDir):
    def __init__(
        self,
        air,
        net_config,
        fs,
        processing_winlen,
        processing_hopsize,
        sigma,
        remult_sigma=True,
    ):
        FeatureExtractorMatchedFilterMaxDir.__init__(
            self, air, fs, processing_winlen, processing_hopsize
        )

        self.net2 = IGCRN(**net_config["subband_config"])

        fsize = self.winlen // 2 + 1
        self.mask_layer = torch.nn.Sequential(
            torch.nn.Linear(fsize, fsize), torch.nn.Sigmoid()
        )
        self.map_layer = torch.nn.Sequential(
            torch.nn.Linear(fsize, fsize), torch.nn.ReLU()
        )
        self.phase_r_layer = torch.nn.Linear(fsize, fsize)
        self.phase_i_layer = torch.nn.Linear(fsize, fsize)

        self.sigma = torch.nn.Parameter(torch.from_numpy(sigma), requires_grad=False)
        self.remult_sigma = remult_sigma

    def forward(self, noisy, doa, ref_mono=None):
        spec = self.stft(noisy)

        x, x_bf, binaural_weights = self.apply_beamformer(spec, doa)

        feat_spec = torch.concat([x_bf[..., None], x], -1)  # BxTxFxC
        feat_spec = torch.concat(
            [torch.real(feat_spec), torch.imag(feat_spec)], -1
        )  # BxTxFxC
        feat_spec = feat_spec.permute(0, 3, 1, 2)  # BxCxTxF
        feat_spec = feat_spec / self.sigma[None, None, None, :]

        map_mask, phase = self.net2(feat_spec)

        phasor_unnormed = torch.complex(
            self.phase_r_layer(phase[:, 0, :, :]), self.phase_i_layer(phase[:, 1, :, :])
        )
        phasor = torch.exp(1j * torch.angle(phasor_unnormed))

        if self.remult_sigma:
            mag = self.map_layer(map_mask[:, 0, :, :]) + self.mask_layer(
                map_mask[:, 1, :, :]
            ) * (torch.abs(x_bf) / self.sigma[None, None, :])
        else:
            mag = self.map_layer(map_mask[:, 0, :, :]) + self.mask_layer(
                map_mask[:, 1, :, :]
            ) * torch.abs(x_bf)

        enhanced_single_channel_spec = (mag * phasor)[..., None]
        if self.remult_sigma:
            enhanced_single_channel_spec = (
                enhanced_single_channel_spec * self.sigma[None, None, :, None]
            )

        enhanced_binaural_spec = enhanced_single_channel_spec * binaural_weights
        enhanced_binaural = self.istft(enhanced_binaural_spec, noisy.shape[1])

        if ref_mono == None:
            return enhanced_binaural
        else:
            ref_spec = self.stft(ref_mono)
            ref_binaural = self.istft(ref_spec * binaural_weights, noisy.shape[1])
            return enhanced_binaural, ref_binaural

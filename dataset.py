import numpy as np
import sys
sys.path.append('spear-tools')
from analysis.SPEAR import SPEAR_Data
import parse
import os
import librosa
from torch.utils.data import Dataset
import scipy.interpolate
import scipy.signal


class SpearDataset(Dataset):
    def __init__(
        self,
        spear_root,
        filelist,
        duration,
        doa_hopsize,
        percentage_target_speech,
        minimum_speech_percentage,
        fs=48000,
        return_vads_only=False,
        eval_only=False,
    ):
        self.filelist = filelist
        self.sp = SPEAR_Data(spear_root)
        self.duration = duration
        self.fs = fs
        self.doa_hopsize = doa_hopsize
        self.percentage_target_speech = percentage_target_speech
        self.minimum_speech_percentage = minimum_speech_percentage
        self.return_vads_only = return_vads_only
        self.eval_only = eval_only
        if eval_only:
            assert duration == 60, "no segments are chosen with evaluation data"
        if return_vads_only:
            assert duration == 60, "vads are only returned for full recordings"

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        filename = self.filelist[item]
        dataset = int(filename[1])
        session = filename[4:5]
        info = parse.parse("D{}_S{}_M{}_ID{}", filename)
        dataset = int(info[0])
        session = int(info[1])
        minute = info[2]
        target_id = int(info[3])

        self.sp.set_file(
            dataset, session, filename[:-4]
        )  # remove last 4 characters (_ID%d)

        if self.eval_only:
            array_audio, fs_orig = self.sp.get_array_audio()
            if self.fs != fs_orig:
                array_audio = scipy.signal.resample_poly(
                    array_audio, self.fs, fs_orig, axis=1
                )
            doa = self.sp.get_doa(
                target_id,
                np.arange(
                    start_second,
                    start_second + self.duration + 2 * self.doa_hopsize,
                    self.doa_hopsize,
                ),
            )

            return array_audio.T.astype("float32"), doa.astype("float32")

        vads, t_vad, _ = self.sp.get_VAD()
        if t_vad.shape[0] < vads.shape[1]:  # error in data (happens sometimes)
            t_vad = np.linspace(0, 60, vads.shape[1])

        any_speaker_vad = np.any(vads > 0, axis=0)
        correct_speaker_vad = vads[target_id - 1, :] > 0

        if self.return_vads_only:
            return t_vad, any_speaker_vad, correct_speaker_vad

        if np.random.rand() <= self.percentage_target_speech:
            vad_to_consider = correct_speaker_vad
        else:
            vad_to_consider = any_speaker_vad

        timeout = 100

        for _ in range(timeout):
            start_second = np.random.uniform(0, 60 - self.duration)
            ind = (t_vad >= start_second) & (t_vad < (start_second + self.duration))

            if np.mean(vad_to_consider[ind]) > self.minimum_speech_percentage:
                break

        vad = correct_speaker_vad[ind]

        array_audio, fs_orig = self.sp.get_array_audio(
            None, start_second, duration=self.duration
        )
        if self.fs != fs_orig:
            array_audio = scipy.signal.resample_poly(
                array_audio, self.fs, fs_orig, axis=1
            )

        doa = self.sp.get_doa(
            target_id,
            np.arange(
                start_second,
                start_second + self.duration + 2 * self.doa_hopsize,
                self.doa_hopsize,
            ),
        )

        if dataset == 1:
            ref_filepath = os.path.join(
                self.sp.root_path,
                "..",
                "..",
                "Extra",
                self.sp.root_path.split(os.sep)[-1],
                "Dataset_2",
                "Reference_Audio",
                self.sp.session_folder,
                minute,
                "cedar_D2_%s.wav" % filename[3:],
            )
            ref_audio, fs_orig = librosa.load(
                ref_filepath,
                sr=None,
                mono=False,
                offset=start_second,
                duration=self.duration,
            )
            assert len(ref_audio.shape) == 1
            ref_audio = ref_audio[None, :]
        else:
            ref_filepath = os.path.join(
                self.sp.root_path,
                "..",
                "..",
                "Extra",
                self.sp.root_path.split(os.sep)[-1],
                self.sp.dataset_folder,
                "Reference_Audio",
                self.sp.session_folder,
                minute,
                "ref_%s.wav" % filename,
            )
            ref_audio, fs_orig = librosa.load(
                ref_filepath,
                sr=None,
                mono=False,
                offset=start_second,
                duration=self.duration,
            )

        if self.fs != fs_orig:
            ref_audio = scipy.signal.resample_poly(ref_audio, self.fs, fs_orig, axis=1)
        ref_audio = ref_audio.T
        t = np.arange(ref_audio.shape[0]) / self.fs + start_second
        f = scipy.interpolate.interp1d(
            t_vad[: vad.shape[0]], vad, kind="previous", fill_value="extrapolate"
        )

        vad_resampled = f(t)
        if dataset == 1:  # D1
            ref_audio = ref_audio * vad_resampled[:, None]

        rms = np.sqrt(
            (np.mean(ref_audio**2 * vad_resampled[:, None]) + 1e-14)
            / (np.mean(vad_resampled) + 1e-9)
        )  # denominator is larger than numerator, so for mean(vad_resampled) -> 0 : rms -> 0
        rms = np.clip(rms, 10 ** (-60 / 20), 10 ** (-10 / 20))

        return (
            array_audio.T.astype("float32"),
            ref_audio.astype("float32"),
            doa.astype("float32"),
            rms.astype("float32"),
        )

    def get_ATFs(self):
        return self.sp.get_all_AIRs()

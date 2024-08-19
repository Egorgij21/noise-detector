import os
import torch
import scipy
import shutil
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from denoiser import pretrained
from audio_separator.separator import Separator


DEVICE = os.environ.get('device')


class NoiseReducer:
    def __init__(
        self,
        wavs_path: str,
        src_conclusion_filepath: str,
        do_neural_denoising: bool,
        do_vocals_denoising: bool
    ) -> None:
        assert src_conclusion_filepath.endswith(".csv"), "Specify the path to the recording file in .csv format."

        self.wavs_path = wavs_path
        self.dataframe = pd.read_csv(src_conclusion_filepath)
        self.POSSIBLE_TAGS = list(self.dataframe.columns[1:])
        self.FATAL_TAGS = ["NO_SPEECH", "WRONG_ANNOTATION", "WHISPER_SPEECH"]
        self.DENOISABLE_TAGS = ["LOW_QUALITY", "NISQA_NOISE", "BAD_COLORATION", "DISCONTINUITY", "NISQA_BAD_LOUDNESS"]
        self.CHANGEABLE_TAGS = ["SOFT_SPEECH", "LOUD_SPEECH"]

        self.do_neural_denoising = do_neural_denoising
        if do_neural_denoising:
            self.denoiser_model = pretrained.dns64().to(DEVICE).eval()

        self.do_vocals_denoising = do_vocals_denoising
        if do_vocals_denoising:
            self.vocals_model = Separator(
                    output_single_stem="vocals",
                    sample_rate=16000,
                    mdx_params={
                        "hop_length": 1024,
                        "segment_size": 256,
                        "overlap": 0.25,
                        "batch_size": 1,
                        "enable_denoise": True,
                    },
                )
            self.vocals_model.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")

    def read_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        sr, signal = scipy.io.wavfile.read(audio_path, mmap=False)

        if sr != target_sr:
            number_of_samples = round(len(signal) * float(target_sr) / sr)
            signal = scipy.signal.resample(signal, number_of_samples)

        if len(signal.shape) > 1:
            signal = (signal[:,0] + signal[:,1]) / 2

        if signal.max() <= 1.1:
            signal = (signal * 32767).astype(np.int16)

        return sr, signal
    
    def write_audio(self, signal: np.ndarray, sr: int, save_path: str, target_sr: int = 16000) -> None:
        if sr != target_sr:
            number_of_samples = round(len(signal) * float(target_sr) / sr)
            signal = scipy.signal.resample(signal, number_of_samples)

        if signal.max() <= 1.1:
            signal = (signal * 32767).astype(np.int16)

        scipy.io.wavfile.write(save_path, target_sr, signal)

    def make_softer(self, signal: np.ndarray, coeff: int) -> np.ndarray:
        # Not implemented correctly, yet
        return signal

    def make_louder(self, signal: np.ndarray, coeff: int) -> np.ndarray:
        # Not implemented correctly, yet
        return signal

    def denoise_audios(
        self,
        save_dirrectory: str,
    ) -> None:
        os.makedirs(save_dirrectory, exist_ok=True)
        for audio_id in tqdm(self.dataframe.file_id):
            is_fatal = False
            is_denoisable = False
            is_changeable = False
            audio_tags = []
            audio_tag_flags = self.dataframe[self.dataframe.file_id == audio_id].values[0,1:]
            for (tag_flag,tag) in zip(audio_tag_flags,self.POSSIBLE_TAGS):
                if tag_flag:
                    audio_tags.append(tag)
                if tag_flag and tag in self.FATAL_TAGS:
                    is_fatal = True
                if tag_flag and tag in self.DENOISABLE_TAGS:
                    is_denoisable = True
                if tag_flag and tag in self.CHANGEABLE_TAGS:
                    is_changeable = True

            if is_fatal or not audio_tag_flags.sum():
                continue

            new_audio_path = os.path.join(save_dirrectory, str(audio_id)+".wav")
            audio_path = os.path.join(self.wavs_path, str(audio_id)+".wav")
            sr, signal = self.read_audio(audio_path, target_sr=16000)

            if not is_fatal and not is_denoisable and not is_changeable:
                shutil.copy(audio_path, new_audio_path)

            if is_denoisable and self.do_neural_denoising:
                float_signal = torch.Tensor((signal / 32767).astype(float)).to(DEVICE)
                signal = self.denoiser_model(float_signal[None])[0][0].cpu().detach().numpy()

            if is_denoisable and self.do_vocals_denoising:
                self.write_audio(signal, sr, new_audio_path, target_sr=16000)
                new_audio_path_ = self.vocals_model.separate(new_audio_path)[0]
                shutil.move(new_audio_path_, new_audio_path)
                sr, signal = self.read_audio(new_audio_path, target_sr=16000)

            if is_changeable:
                signal = torch.Tensor((signal / 32767).astype(float))
                if "SOFT_SPEECH" in audio_tags:
                    signal = self.make_louder(signal, 2)
                if "LOUD_SPEECH" in audio_tags:
                    signal = self.make_softer(signal, 2)

            self.write_audio(signal, sr, new_audio_path, target_sr=16000)

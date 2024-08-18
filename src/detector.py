import os
import scipy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Optional

from silero_vad import load_silero_vad
from utils import read_metadata, load_nemo_asr, check_speech, check_annotation


class NoiseDetector:
    def __init__(
        self,
        metadata_path: str,
        wavs_path: str,
        do_check_speech: bool,
        do_check_annotations: bool
    ) -> None:
        self.meta = read_metadata(metadata_path)
        self.wavs_path = wavs_path
        self.do_check_speech = do_check_speech
        self.do_check_annotations = do_check_annotations
        self.return_txt_file = ""

        self.vad = None
        self.asr = None

        if do_check_speech:
            self.vad = load_silero_vad(onnx=True)

        if do_check_annotations:
            self.asr = load_nemo_asr()

    def read_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        sr, signal = scipy.io.wavfile.read(audio_path, mmap=False)

        if sr != target_sr:
            number_of_samples = round(len(signal) * float(target_sr) / sr)
            signal = scipy.signal.resample(signal, number_of_samples)

        if len(signal.shape) > 1:
            signal = (signal[:,0] + signal[:,1]) / 2

        if signal.max() <= 1.001:
            signal = (signal * 32767).astype(np.int16)

        return sr, signal

    def process_audios(self, return_dataframe: bool = True) -> Optional[pd.DataFrame]:
        for audio_path in tqdm(glob(os.path.join(self.wavs_path, "*.wav"))):
            TAGS = []
            id = audio_path.split("/")[-1].split(".")[0]
            annotation = self.meta[id]

            _, signal = self.read_audio(audio_path, target_sr=16000)

            if self.do_check_speech and not check_speech(vad_model=self.vad, signal=signal):
                TAGS.append("NO_SPEECH")
            
            if self.do_check_speech and not check_annotation(asr_model=self.asr, signal=signal, target_text=annotation):
                TAGS.append("WRONG_ANNOTATION")
        return None
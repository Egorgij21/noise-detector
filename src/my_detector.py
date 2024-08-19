import sys
sys.path.append("/workdir/NISQA-s/src/")

import os
import re
import yaml
import scipy
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from jiwer import cer

from langdetect import detect as lang_detector
from core.model_torch import model_init as nisqa_init
from silero_vad import load_silero_vad, get_speech_timestamps
from nisqa_utils.process_utils import process as nisqa_process
from utils import read_metadata, load_nemo_asr, remove_punctuation
from voice_utils import is_soft_speech, is_loud_speech, is_whisper


class NoiseDetector:
    def __init__(
        self,
        metadata_path: str,
        wavs_path: str,
        result_filepath: str,
        do_check_speech: bool,
        do_check_annotations: bool,
        do_lang_check: bool,
        do_nisqa_check: bool,
        do_voice_check: bool,
        lang: str = "en"
    ) -> None:
        assert result_filepath.endswith(".txt"), "Specify the path to the recording file in .txt format."

        self.meta = read_metadata(metadata_path)
        self.wavs_path = wavs_path
        self.result_filepath = result_filepath
        self.do_check_speech = do_check_speech
        self.do_check_annotations = do_check_annotations
        self.do_lang_check = do_lang_check
        self.do_nisqa_check = do_nisqa_check
        self.do_voice_check = do_voice_check
        self.lang = lang
        self.return_txt_file = ""

        self.vad = None
        self.asr = None

        self.POSSIBLE_TAGS = ["OK"]
        if do_check_speech:
            self.vad = load_silero_vad(onnx=True)
            self.POSSIBLE_TAGS.append("NO_SPEECH")

        if do_check_annotations:
            self.asr = load_nemo_asr(lang=lang)
            self.POSSIBLE_TAGS.append("WRONG_ANNOTATION")

        if do_lang_check:
            self.POSSIBLE_TAGS.append("WRONG_LANGUAGE")

        if do_nisqa_check:
            with open("NISQA-s/config/nisqa_s.yaml", "r") as ymlfile:
                args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
            args = {**args_yaml}
            args["ckp"] = "/workdir/NISQA-s/src/weights/nisqa_s.tar"
            nisqa_model, h0, c0 = nisqa_init(args)
            self.nisqa_args = args
            self.nisqa_model = nisqa_model
            self.h0 = h0
            self.c0 = c0
            self.POSSIBLE_TAGS.extend(["LOW_QUALITY",
                                       "NISQA_NOISE",
                                       "BAD_COLORATION",
                                       "DISCONTINUITY",
                                       "NISQA_BAD_LOUDNESS"])

        if do_voice_check:
            self.POSSIBLE_TAGS.extend(["SOFT_SPEECH",
                                       "LOUD_SPEECH",
                                       "WHISPER_SPEECH"])

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
    
    def calculate_nisqa(self, signal: np.ndarray, sr: int) -> np.ndarray[float]:
        framesize = sr * 2
        signal = torch.as_tensor(signal)
        h0, c0 = self.h0.clone(), self.c0.clone()

        if signal.shape[0] % framesize != 0:
            signal = torch.cat((signal, torch.zeros(framesize - signal.shape[0] % framesize)))
        signal_spl = torch.split(signal, framesize, dim=0)
        out_all = []
        np.set_printoptions(precision=3)
        for signal in signal_spl:
            out, h0, c0 = nisqa_process(signal, sr, self.nisqa_model, h0, c0, self.nisqa_args)
            out_all.append(out[0].numpy())

        avg_out = np.mean(out_all, axis=0)

        return avg_out

    def check_annotation(self, path_to_audio: str, target_text: str, cer_th: float = 0.1) -> bool:
        pred_text = self.asr.transcribe(paths2audio_files=[path_to_audio], verbose=False)[0][0]
        target_text = remove_punctuation(target_text)

        pred_cer = None
        if pred_text != "":
            pred_cer = cer(pred_text.lower(), target_text.lower())

        if (pred_cer is None and target_text != "") or pred_cer > cer_th:
            return False
        return True

    def check_speech(self, signal: np.ndarray) -> bool:
        timestamps = get_speech_timestamps(signal, self.vad)
        if len(timestamps):
            return True
        return False

    def check_language(self, text: str) -> bool:
        detected_lang = lang_detector(text)
        return detected_lang == self.lang
    
    def save_dataframe(self, save_path: str) -> None:
        file_ids = [path.split("/")[-1].split(".")[0] for path in glob(os.path.join(self.wavs_path, "*.wav"))]
        num_files = len(file_ids)
        dataframe = {"file_id": file_ids}

        for column_name in self.POSSIBLE_TAGS:
            dataframe[column_name] = [False for i in range(num_files)]

        for line in self.return_txt_file.split("\n")[:-1]:
            line_tags = line.split("|")[-1].split(" ")
            line_id = line.split("|")[0]
            id_index = dataframe["file_id"].index(line_id)

            if line_tags[0] == "OK":
                dataframe["OK"][id_index] = True
            else:
                for tag in line_tags:
                    dataframe[tag][id_index] = True

        dataframe = pd.DataFrame(dataframe)
        dataframe.to_csv(save_path, index=False)

    def process_audios(
        self,
        make_dataframe: bool = True,
        quality_th: float = 1.2,
        noiseness_th: float = 2.,
        coloration_th: float = 2.,
        discontinuity_th: float = 1.5,
        loundness_th: float = 1.
    ) -> None:

        for audio_path in tqdm(glob(os.path.join(self.wavs_path, "*.wav"))):
            TAGS = []
            id = audio_path.split("/")[-1].split(".")[0]
            annotation = self.meta[id]

            sr, signal = self.read_audio(audio_path, target_sr=16000)

            if self.do_check_speech and not self.check_speech(signal=signal):
                TAGS.append("NO_SPEECH")

            if self.do_check_annotations and not self.check_annotation(path_to_audio=audio_path, target_text=annotation):
                TAGS.append("WRONG_ANNOTATION")

            if self.do_lang_check and not self.check_language(text=annotation):
                TAGS.append("WRONG_LANGUAGE")

            if self.do_nisqa_check:
                (quality,noiseness,coloration,discontinuity,loundness) = self.calculate_nisqa(signal, sr)
                if quality < quality_th:
                    TAGS.append("LOW_QUALITY")
                if noiseness < noiseness_th:
                    TAGS.append("NISQA_NOISE")
                if coloration < coloration_th:
                    TAGS.append("BAD_COLORATION")
                if discontinuity < discontinuity_th:
                    TAGS.append("DISCONTINUITY")
                if loundness < loundness_th:
                    TAGS.append("BAD_LOUDNESS")

            if self.do_voice_check:
                float_signal = (signal / 32767).astype(float)
                if is_soft_speech(float_signal, sr):
                    TAGS.append("SOFT_SPEECH")
                if is_loud_speech(float_signal, sr):
                    TAGS.append("LOUD_SPEECH")
                if is_whisper(float_signal, sr):
                    TAGS.append("WHISPER_SPEECH")

            if len(TAGS) == 0:
                TAGS.append("OK")

            self.return_txt_file += f"{id}|" + " ".join(tag for tag in TAGS) + "\n"

        with open(self.result_filepath, "w") as f:
            f.write(self.return_txt_file)

        if make_dataframe:
            dataframe_path = re.sub(".txt", ".csv", self.result_filepath)
            self.save_dataframe(save_path=dataframe_path)

        return

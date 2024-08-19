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
    """
    A class to detect noise, speech, language issues, and other quality aspects of audio files.

    Attributes:
        meta (dict): Metadata containing target annotations for audio files.
        wavs_path (str): Path to the directory containing WAV files.
        result_filepath (str): Path to save the results of the noise detection.
        do_check_speech (bool): Flag to indicate if speech presence should be checked.
        do_check_annotations (bool): Flag to indicate if annotations should be checked for correctness.
        do_lang_check (bool): Flag to indicate if language detection should be performed.
        do_nisqa_check (bool): Flag to indicate if NISQA (speech quality assessment) should be performed.
        do_voice_check (bool): Flag to indicate if voice characteristics (e.g., soft speech, loud speech) should be checked.
        lang (str): Expected language of the speech in the audio files.
        return_txt_file (str): String to store the results before saving to file.
        vad (silero_vad): VAD (Voice Activity Detector) model for checking speech presence.
        asr (ASRModel): ASR (Automatic Speech Recognition) model for annotation checking.
        POSSIBLE_TAGS (list): List of possible tags that can be assigned to audio files during processing.
        nisqa_args (dict): Configuration parameters for the NISQA model.
        nisqa_model (torch.nn.Module): The loaded NISQA model.
        h0, c0 (torch.Tensor): Initial hidden and cell states for the NISQA model's RNN.
    """

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
        """
        Initializes the NoiseDetector with specified configurations and loads necessary models.

        Args:
            metadata_path (str): Path to the metadata file containing annotations.
            wavs_path (str): Path to the directory containing WAV files.
            result_filepath (str): Path to save the results of the noise detection.
            do_check_speech (bool): Flag to enable or disable speech presence checking.
            do_check_annotations (bool): Flag to enable or disable annotation correctness checking.
            do_lang_check (bool): Flag to enable or disable language detection.
            do_nisqa_check (bool): Flag to enable or disable NISQA (speech quality assessment).
            do_voice_check (bool): Flag to enable or disable voice characteristics checking.
            lang (str, optional): Expected language of the speech in the audio files. Defaults to "en".
        """

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
        """
        Reads an audio file and resamples it to the target sample rate if necessary.

        Args:
            audio_path (str): Path to the audio file.
            target_sr (int, optional): Target sample rate. Defaults to 16000.

        Returns:
            tuple: Sample rate and the audio signal as a numpy array.
        """
        sr, signal = scipy.io.wavfile.read(audio_path, mmap=False)

        if sr != target_sr:
            number_of_samples = round(len(signal) * float(target_sr) / sr)
            signal = scipy.signal.resample(signal, number_of_samples)

        if len(signal.shape) > 1:
            signal = (signal[:,0] + signal[:,1]) / 2

        if signal.max() <= 1.1:
            signal = (signal * 32767).astype(np.int16)

        return sr, signal.astype(np.int16)
    
    def calculate_nisqa(self, signal: np.ndarray, sr: int) -> np.ndarray[float]:
        """
        Calculates NISQA (speech quality) scores for the given audio signal.

        Args:
            signal (np.ndarray): The audio signal.
            sr (int): Sample rate of the audio signal.

        Returns:
            np.ndarray: Array containing NISQA quality scores.
        """
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
        """
        Checks the correctness of the annotation by comparing the predicted and target transcriptions.

        Args:
            path_to_audio (str): Path to the audio file.
            target_text (str): The target text annotation.
            cer_th (float, optional): CER (Character Error Rate) threshold for correctness. Defaults to 0.1.

        Returns:
            bool: True if the annotation is correct, False otherwise.
        """
        pred_text = self.asr.transcribe(paths2audio_files=[path_to_audio], verbose=False)[0][0]
        target_text = remove_punctuation(target_text)

        pred_cer = None
        if pred_text != "":
            pred_cer = cer(pred_text.lower(), target_text.lower())

        if (pred_cer is None and target_text != "") or pred_cer > cer_th:
            return False
        return True

    def check_speech(self, signal: np.ndarray) -> bool:
        """
        Checks if speech is present in the audio signal using VAD (Voice Activity Detection).

        Args:
            signal (np.ndarray): The audio signal.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        timestamps = get_speech_timestamps(signal, self.vad)
        if len(timestamps):
            return True
        return False

    def check_language(self, text: str) -> bool:
        """
        Detects the language of the given text and checks if it matches the expected language.

        Args:
            text (str): The text to be checked.

        Returns:
            bool: True if the detected language matches the expected language, False otherwise.
        """
        detected_lang = lang_detector(text)
        return detected_lang == self.lang
    
    def save_dataframe(self, save_path: str) -> None:
        """
        Saves the processed results as a CSV file.

        Args:
            save_path (str): Path to save the CSV file.
        """
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
        noiseness_th: float = 2.0,
        coloration_th: float = 2.0,
        discontinuity_th: float = 1.5,
        loundness_th: float = 1.0
    ) -> None:
        """
        Processes all audio files in the specified directory, applying various checks based on the provided configurations.

        Args:
            make_dataframe (bool, optional): Whether to generate a dataframe of results. Defaults to True.
            quality_th (float, optional): Threshold for NISQA quality score. Defaults to 1.2.
            noiseness_th (float, optional): Threshold for NISQA noiseness score. Defaults to 2.0.
            coloration_th (float, optional): Threshold for NISQA coloration score. Defaults to 2.0.
            discontinuity_th (float, optional): Threshold for NISQA discontinuity score. Defaults to 1.5.
            loundness_th (float, optional): Threshold for NISQA loudness score. Defaults to 1.0.
        """

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

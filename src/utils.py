import os
import re
import numpy as np
from jiwer import cer
import matplotlib.pyplot as plt

from silero_vad import get_speech_timestamps

import nemo.collections.asr as nemo_asr


DEVICE = os.environ.get('device')


def read_metadata(path: str) -> dict[str, str]:
    data = {}
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        (id, annotation) = line.split("|")
        annotation = re.sub("\n", "", annotation)
        data[id] = annotation
    return data


def load_nemo_asr():
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
    asr_model.to(DEVICE)
    asr_model.eval()
    return asr_model


def check_annotation(asr_model, path_to_audio: str, target_text: str, cer_th: float = 0.1) -> bool:
    pred_text = asr_model.transcribe(paths2audio_files=[path_to_audio])[0]
    pred_cer = cer(pred_text, target_text)
    if pred_cer > cer_th:
        return False
    return True


def check_speech(vad_model, signal: np.ndarray):
    timestamps = get_speech_timestamps(signal, vad_model)
    if len(timestamps):
        return True
    return False


def calculate_fft(signal, sampling_rate):
    fft_spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1/sampling_rate)
    amplitudes = np.abs(fft_spectrum)
    return frequencies[:len(frequencies) // 2], amplitudes[:len(amplitudes) // 2]


def plot_fft(frequencies, amplitudes, limit=None):
    plt.figure(figsize=(10, 6))
    if limit:
        mask = frequencies >= 0
        mask &= frequencies <= limit
        plt.plot(frequencies[mask], amplitudes[mask])
    else:
        plt.plot(frequencies, amplitudes)

    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

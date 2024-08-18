import os
import re
import string
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

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


def load_nemo_asr(lang: str = "en"):
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=f"stt_{lang}_conformer_transducer_small")
    asr_model.to(DEVICE)
    asr_model.eval()
    return asr_model


def remove_punctuation(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation + '“”‘’\'"')
    return text.translate(translator)


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

import os
import re
import string
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import nemo.collections.asr as nemo_asr


DEVICE = os.environ.get('device')


def read_metadata(path: str) -> dict[str, str]:
    """
    Reads metadata from a file and returns it as a dictionary.

    Args:
        path (str): The path to the metadata file. The file should contain lines in the format "id|annotation".

    Returns:
        dict[str, str]: A dictionary where keys are the audio file IDs and values are the corresponding annotations.
    """
    data = {}
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        (id, annotation) = line.split("|")
        annotation = re.sub("\n", "", annotation)
        data[id] = annotation
    return data


def load_nemo_asr(lang: str = "en"):
    """
    Loads a pre-trained NeMo ASR (Automatic Speech Recognition) model for a specified language.

    Args:
        lang (str, optional): The language code for the ASR model to load. Defaults to "en" for English.

    Returns:
        nemo_asr.models.EncDecRNNTBPEModel: The loaded ASR model ready for inference.
    """
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=f"stt_{lang}_conformer_transducer_small")
    asr_model.to(DEVICE)
    asr_model.eval()
    return asr_model


def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from a given text string.

    Args:
        text (str): The input string from which to remove punctuation.

    Returns:
        str: The text string with all punctuation removed.
    """
    translator = str.maketrans('', '', string.punctuation + '“”‘’\'"')
    return text.translate(translator)


def calculate_fft(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Fast Fourier Transform (FFT) of an audio signal.

    Args:
        signal (np.ndarray): The audio signal array.
        sampling_rate (int): The sampling rate of the audio signal.

    Returns:
        tuple[np.ndarray, np.ndarray]: The frequencies and corresponding amplitudes of the FFT.
    """
    fft_spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1/sampling_rate)
    amplitudes = np.abs(fft_spectrum)
    return frequencies[:len(frequencies) // 2], amplitudes[:len(amplitudes) // 2]


def plot_fft(frequencies: np.ndarray, amplitudes: np.ndarray, limit: int = None) -> None:
    """
    Plots the frequency spectrum of an audio signal using its FFT data.

    Args:
        frequencies (np.ndarray): The frequencies calculated from the FFT.
        amplitudes (np.ndarray): The amplitudes corresponding to the frequencies.
        limit (int, optional): An optional frequency limit to display in the plot. If not provided, plots the full spectrum.

    Returns:
        None
    """
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

import scipy
import librosa
import numpy as np
import parselmouth as pm


FORMANT_RANGES = {
    1: (200, 1000),
    2: (800, 2500),
    3: (1500, 3500),
    4: (2500, 4500),
    5: (3500, 5500),
    6: (5500, 6500),
    7: (6500, 7500),
    8: (7500, 8500),
    9: (8500, 9500),
}


def is_soft_speech(
        signal: np.ndarray,
        sr: int,
        intensity_min_db: int = 20,
        intensity_max_db: int = 45,
        f0_min: int = 50,
        f0_max: int = 300,
        formant_thresholds: int = (300, 3000)
    ):
    """
    Определяет, является ли аудиосигнал тихой речью, используя интенсивность, основной тон (F0) и форманты.

    :param signal: Нормализованный аудиосигнал.
    :param sr: Частота дискретизации.
    :param intensity_min_db: Минимальный порог интенсивности в децибелах для определения тихой речи.
    :param intensity_max_db: Максимальный порог интенсивности в децибелах для определения тихой речи.
    :param f0_min: Минимальная частота тона (Hz).
    :param f0_max: Максимальная частота тона (Hz).
    :param formant_thresholds: Ожидаемые диапазоны частот для первых двух формант (F1, F2).
    :return: True если сигнал является тихой речью, иначе False.
    """
    signal = noise_reduction_spectral_subtraction(signal, sr, normalize=True)

    # Рассчитываем частоту тона в заданном диапазоне F0
    f0, _ = extract_f0_voiced(signal, sr)

    # Если F0 отсутствует или все значения NaN, это не тихая речь
    if f0 is None or np.all(np.isnan(f0)):
        return False

    # Убираем короткие последовательности значений F0
    valid_f0 = remove_short_sequences(f0, min_length=15)
    if len(valid_f0) == 0:
        return False

    f0_mean = np.mean(valid_f0)
    f0_mean_max = np.max(valid_f0)
    if f0_mean < f0_min or f0_mean > f0_max or f0_mean_max > f0_max:
        return False

    # Рассчитываем интенсивность с учетом озвученных фрагментов
    intensity_db = calculate_pitch_intensity(signal, sr, only_speech=True)
    if intensity_db.size == 0:
        return False

    avg_intensity = np.mean(intensity_db)

    # Проверяем, что средняя интенсивность находится в диапазоне для тихой речи
    if not (intensity_min_db <= avg_intensity <= intensity_max_db):
        return False

    # Проверка формант
    formants = extract_formants_from_segment(signal, sr)
    if formants:
        significant_formants = [f for f in formants[:2] if formant_thresholds[0] < f < formant_thresholds[1]]
        if len(significant_formants) < 2:
            return False

    return True


def is_loud_speech(
        signal: np.ndarray,
        sr: int,
        intensity_threshold_db: int = 55,
        spectral_flatness_threshold: float = 0.2,
        f0_min: int = 80,
        f0_max: int = 300,
        formant_thresholds: tuple[int] = (300, 3000)
    ):
    """
    Определяет, является ли аудиосигнал громкой речью, используя интенсивность, спектральную плоскость, основной тон (F0) и форманты.

    :param signal: Нормализованный аудиосигнал.
    :param sr: Частота дискретизации.
    :param intensity_threshold_db: Порог интенсивности в децибелах для определения громкой речи.
    :param spectral_flatness_threshold: Порог для спектральной плоскости.
    :param f0_min: Минимальная частота тона (Hz).
    :param f0_max: Максимальная частота тона (Hz).
    :param formant_thresholds: Ожидаемые диапазоны частот для первых двух формант (F1, F2).
    :return: True если сигнал является громкой речью, иначе False.
    """
    signal = noise_reduction_spectral_subtraction(signal, sr, normalize=True)

    # Рассчитываем интенсивность с учетом озвученных фрагментов
    intensity_db = calculate_pitch_intensity(signal, sr, only_speech=True)
    if intensity_db.size == 0:
        return False

    avg_intensity = np.mean(intensity_db)

    # Проверяем, что средняя интенсивность выше порога
    if avg_intensity < intensity_threshold_db:
        return False

    # Рассчитываем спектральную плоскость
    spectral_flatness = librosa.feature.spectral_flatness(y=signal)
    avg_spectral_flatness = np.mean(spectral_flatness)
    if avg_spectral_flatness > spectral_flatness_threshold:
        return False

    # Рассчитываем частоту тона в заданном диапазоне F0
    f0, _ = extract_f0_voiced(signal, sr)

    # Если F0 отсутствует или все значения NaN, это не громкая речь
    if f0 is None or np.all(np.isnan(f0)):
        return False

    # Убираем короткие последовательности значений F0
    valid_f0 = remove_short_sequences(f0, min_length=15)
    if len(valid_f0) == 0:
        return False

    f0_mean = np.max(valid_f0)
    if f0_mean < f0_min or f0_mean > f0_max:
        return False

    # Проверка формант
    formants = extract_formants_from_segment(signal, sr)
    if formants:
        significant_formants = [f for f in formants[:2] if formant_thresholds[0] < f < formant_thresholds[1]]
        if len(significant_formants) < 2:
            return False

    return True


def is_whisper(
        signal: np.ndarray,
        sr: int,
        f0_min: int = 50,
        f0_avg: int = 100, 
        spectral_flatness_threshold: float = 0.4,
        formant_threshold: int = 500
    ):
    """
    Определяет, является ли аудиосигнал шёпотом, используя отсутствие основного тона (F0),
    высокую спектральную плоскость и отсутствие значимых формантов.

    :param signal: Нормализованный аудиосигнал.
    :param sr: Частота дискретизации.
    :param f0_min: Минимальная частота тона (Hz) для определения наличия F0.
    :param f0_max: Максимальная частота тона (Hz) для определения наличия F0.
    :param spectral_flatness_threshold: Порог для спектральной плоскости, выше которого считается шёпотом.
    :param formant_threshold: Порог для частот формант, ниже которого считается шёпотом.
    :return: True если сигнал является шёпотом, иначе False.
    """
    signal = noise_reduction_spectral_subtraction_fast(signal, sr)

    # Рассчитываем частоту тона в заданном диапазоне F0
    f0, _ = extract_f0_voiced(signal, sr)

    # Если F0 отсутствует или все значения NaN, это может указывать на шёпот
    if f0 is None or np.all(np.isnan(f0)):
        return True

    # Убираем короткие последовательности значений F0
    valid_f0 = remove_short_sequences(f0, min_length=5)

    # Если после удаления коротких последовательностей нет валидных значений F0, это шёпот
    if len(valid_f0) == 0:
        return True

    f0_mean = np.mean(valid_f0)
    if f0_mean < f0_min:
        return True
    if f0_mean > f0_avg:
        return False

    # Рассчитываем спектральную плоскость
    spectral_flatness = librosa.feature.spectral_flatness(y=signal)
    avg_spectral_flatness = np.mean(spectral_flatness)
    if avg_spectral_flatness > spectral_flatness_threshold:
        return True

    # Проверка формант
    formants = extract_formants_from_segment(signal, sr)
    if formants:
        significant_formants = [f for f in formants[:2] if f >= formant_threshold]  # Смотрим только на первые два форманта
        if len(significant_formants) == 0:
            return True

    return False


def is_scream(
        signal: np.ndarray,
        sr: int,
        pitch_threshold: int = 400
    ):
    """
    Определяет, является ли аудиосигнал криком, используя интенсивность, частоту основного тона (F0) и спектральные характеристики.

    :param signal: Нормализованный аудиосигнал.
    :param sr: Частота дискретизации.
    :param pitch_threshold: Порог спектральной плоскости, выше которого может быть крик.
    :return: True если сигнал является криком, иначе False.
    """
    signal = noise_reduction_spectral_subtraction(signal, sr)

    # Рассчитываем частоту тона
    f0, time = extract_f0_voiced(signal, sr)
    if len(f0) == 0:
        return False

    valid_f0 = remove_short_sequences(f0, min_length=5)
    if len(valid_f0) == 0:
        return False

    max_pitch = np.max(valid_f0)
    pitch = np.mean(valid_f0)
    if max_pitch >= pitch_threshold and pitch > 100:
        return True

    return False


def formant_frequencies(
        signal: np.ndarray,
        sr: int,
        speech: bool = True
    ):
    sound = pm.Sound(signal, sampling_frequency=sr)
    formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)

    frequencies = [[] for _ in range(5)]
    bandwidths = [[] for _ in range(5)]
    intensities = []
    times = formants.ts()

    min_duration = 0.064  # Minimum duration required for intensity analysis
    sound_duration = sound.get_total_duration()

    oldvalue = [0, 0, 0, 0, 0]
    if sound_duration >= min_duration:
        intensity = sound.to_intensity()
        if speech:
            # Define the intensity threshold for detecting speech
            intensity_threshold = 50

            for t in times:
                intensity_value = intensity.get_value(t)
                if intensity_value > intensity_threshold:
                    intensities.append(intensity_value)
                    for i in range(1, 6):
                        freq = formants.get_value_at_time(i, t)
                        bandwidth = formants.get_bandwidth_at_time(i, t)
                        if not np.isnan(freq) and FORMANT_RANGES[i][0] <= freq <= FORMANT_RANGES[i][1]:
                            frequencies[i - 1].append(freq)
                            bandwidths[i - 1].append(bandwidth)
                            oldvalue[i - 1] = freq
                        else:
                            frequencies[i - 1].append(np.nan)
                            bandwidths[i - 1].append(np.nan)
                else:
                    intensities.append(np.nan)
                    for i in range(1, 6):
                        frequencies[i - 1].append(np.nan)
                        bandwidths[i - 1].append(np.nan)
        else:
            for t in times:
                intensity_value = intensity.get_value(t)
                intensities.append(intensity_value)
                for i in range(1, 6):
                    freq = formants.get_value_at_time(i, t)
                    bandwidth = formants.get_bandwidth_at_time(i, t)
                    if not np.isnan(freq) and FORMANT_RANGES[i][0] <= freq <= FORMANT_RANGES[i][1]:
                        frequencies[i - 1].append(freq)
                        bandwidths[i - 1].append(bandwidth)
                    else:
                        frequencies[i - 1].append(np.nan)
                        bandwidths[i - 1].append(np.nan)
    else:
        intensities = [np.nan] * len(times)
        for t in times:
            for i in range(1, 6):
                freq = formants.get_value_at_time(i, t)
                bandwidth = formants.get_bandwidth_at_time(i, t)
                frequencies[i - 1].append(freq)
                bandwidths[i - 1].append(bandwidth)

    # Ensure all lists have the same length by padding with NaNs
    max_length = max(len(f) for f in frequencies)
    for i in range(5):
        while len(frequencies[i]) < max_length:
            frequencies[i].append(np.nan)
            bandwidths[i].append(np.nan)

    while len(intensities) < max_length:
        intensities.append(np.nan)

    return frequencies, times, bandwidths, intensities


def extract_formants(
        signal: np.ndarray,
        sr: int,
        speech: bool = True
    ):
    frequencies, _, _, _ = formant_frequencies(signal, sr, speech)
    freqs = []
    for i in range(5):
        if all(np.isnan(freq) for freq in frequencies[i]):
            freqs.append(np.nan)
        else:
            freqs.append(np.nanmean(frequencies[i]))

    return freqs


def extract_formants_from_segment(
        signal_segment: np.ndarray,
        sr: int,
        speech: bool = True
    ):
    """
    Извлекает частоты формант из сегмента сигнала.

    :param signal_segment: Сегмент аудиосигнала.
    :param sr: Частота дискретизации.
    :return: Массив частот формант.
    """
    # Пре-эмфазис для усиления высоких частот
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal_segment[0], signal_segment[1:] - pre_emphasis * signal_segment[:-1])

    frequencies = extract_formants(emphasized_signal, sr, speech)
    return frequencies


def extract_f0_voiced(
        signal: np.ndarray,
        sr: int
    ):
    # Создание объекта Sound из аудиосигнала
    sound = pm.Sound(signal, sr)

    # Извлечение частоты основного тона
    pitch = sound.to_pitch(time_step=0.01)
    pitch_values = pitch.selected_array['frequency']
    times = pitch.xs()

    # Определение сегментов с основным тоном
    voiced_segments = ~np.isnan(pitch_values)

    # Вычисление времени для озвученных сегментов
    voiced_times = times[voiced_segments]
    pitch_values = pitch_values[voiced_segments]

    voiced_times = voiced_times[pitch_values > 0]
    pitch_values = pitch_values[pitch_values > 0]

    return pitch_values, voiced_times


def remove_short_sequences(
        f0: np.ndarray,
        min_length: int = 15
    ):
    """
    Убирает короткие последовательности значений F0.

    :param f0: Массив значений F0 с NaN.
    :param min_length: Минимальная длина последовательности для сохранения.
    :return: Массив значений F0 без коротких последовательностей.
    """
    valid_f0 = []
    current_sequence = []
    old_value = np.nan

    for value in f0:
        if not np.isnan(value) and value > 0 and (
                np.isnan(old_value) or (old_value > 0 and abs(old_value - value < 50))):
            current_sequence.append(value)
            old_value = value
        else:
            if len(current_sequence) >= min_length:
                valid_f0.extend(current_sequence)
            current_sequence = []
        if not np.isnan(value) and value > 0:
            old_value = value

    # Проверяем последнюю последовательность
    if len(current_sequence) >= min_length:
        valid_f0.extend(current_sequence)

    return np.array(valid_f0)


def calculate_pitch_intensity(
        signal: np.ndarray,
        sr: int,
        only_speech: bool = False
    ):
    # Загрузка аудиофайла
    sound = pm.Sound(signal, sr)

    # Получение основного тона (pitch)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_times = pitch.xs()

    # Получение интенсивности звука
    intensity = sound.to_intensity()
    intensity_values = intensity.values.flatten()

    if only_speech:
        intensity_times = intensity.xs()

        # Интерполяция значений pitch до временных меток интенсивности
        interpolated_pitch_values = np.interp(intensity_times, pitch_times, pitch_values)

        # Маска для выбора только тех интенсивностей, где есть pitch
        mask = interpolated_pitch_values > 0
        pitch_intensity_values = intensity_values[mask]

        intensity_threshold = 50
        voiced_frames = pitch_intensity_values > intensity_threshold
        intensity_values = pitch_intensity_values[voiced_frames]

    return intensity_values


def normalize_audio(signal: np.ndarray):
    signal = librosa.util.normalize(signal)
    return signal


def apply_high_pass_filter(
        signal: np.ndarray,
        sr: int,
        cutoff: int = 100
    ):
    sos = scipy.signal.butter(10, cutoff, btype='highpass', fs=sr, output='sos')
    y_filtered = scipy.signal.sosfilt(sos, signal)
    return y_filtered


def apply_low_pass_filter(
        signal: np.ndarray,
        sr: int,
        cutoff: int = 8000
    ):
    sos = scipy.signal.butter(10, cutoff, btype='lowpass', fs=sr, output='sos')
    y_filtered = scipy.signal.sosfilt(sos, signal)
    return y_filtered


def noise_reduction_spectral_subtraction_fast(
        signal: np.ndarray,
        sr: int,
        noise_profile_sec=0.5,
        num_iterations=2
    ):
    # Compute STFT of the input signal
    stft = librosa.stft(signal)
    magnitude, phase = librosa.magphase(stft)

    # Estimate the noise profile from the first `noise_profile_sec` seconds
    num_noise_samples = int(noise_profile_sec * sr)
    noise_profile = np.mean(magnitude[:, :num_noise_samples], axis=1)

    # Apply spectral subtraction iteratively
    for _ in range(num_iterations):
        magnitude = np.maximum(magnitude - noise_profile[:, np.newaxis], 0)

    # Reconstruct the denoised signal using the phase
    stft_denoised = magnitude * phase
    y_denoised = librosa.istft(stft_denoised)

    # Save the denoised audio
    return y_denoised


def noise_reduction_spectral_subtraction(
        signal: np.ndarray,
        sr: int,
        noise_profile_sec=0.5,
        num_iterations=2,
        high_pass=False,
        low_pass=False,
        normalize=False,
    ):
    y_denoised = noise_reduction_spectral_subtraction_fast(signal, sr, noise_profile_sec, num_iterations)

    # Apply high-pass and low-pass filters if enabled
    if high_pass:
        y_denoised = apply_high_pass_filter(y_denoised, sr, cutoff=100)
    if low_pass:
        y_denoised = apply_low_pass_filter(y_denoised, sr, cutoff=8000)

    # Normalize the amplitude if enabled
    if normalize:
        y_denoised = normalize_audio(y_denoised)

    # Save the denoised audio
    return y_denoised

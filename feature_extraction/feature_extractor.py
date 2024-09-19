from scipy.io import wavfile
from signal_processing.signal_processing import butter_bandpass_filter
from feature_extraction.DTHF import DTHF
import numpy as np
import logging

def read_wav_to_signal(wav_path, apply_filter=False, lowcut=25, highcut=400):
    try:
        sample_rate, data = wavfile.read(wav_path)
        logging.info(f"Sample rate: {sample_rate}")
        if data.ndim > 1:
            data = data[:, 0]
        data = data / np.max(np.abs(data))
        if apply_filter:
            data = butter_bandpass_filter(data, lowcut, highcut, sample_rate)
        return data
    except Exception as e:
        logging.error(f"Error reading {wav_path}: {e}")
        return None

def extract_features(file_path, label, wav_suffix):
    signal = read_wav_to_signal(file_path + wav_suffix, apply_filter=True, lowcut=25, highcut=400)
    if signal is None:
        return None
    dthf = DTHF(signal)
    dthf.multiple_iterations(100)
    features = [f"[{item[0]}, {item[1]}]" for item in dthf.minimum]
    return [file_path + wav_suffix, label] + features

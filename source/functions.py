import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa
import librosa.feature as feat

def import_sounds(path_to_sound_files):
    """
    function to load the files

    :param path_to_sound_files: path for the sound files
    :return:
    sound_waves: loaded files
    sound_types: their names
    """

    # 1. get sound file names and 2. names of types of sounds
    sound_file_names=os.listdir(path_to_sound_files)
    sound_types=[name[4:-4] for name in sound_file_names]

    # load sound_files
    sound_waves = [librosa.load(path_to_sound_files+name,None)[0] for name in sound_file_names]


    return sound_waves,sound_types

def extract_features(soundwave,sampling_rate,feature_list=[]):

    if len(feature_list)==0:
        feature_list=["chroma_stft","chroma_cqt","chroma_cens","melspectrogram",
                      "mfcc","rmse","spectral_centroid","spectral_bandwidth",
                      "spectral_contrast","spectral_flatness","spectral_rolloff",
                      "poly_features","tonnetz","zero_crossing_rate"]

    features=[]

    if "chroma_stft" in features:
        features.append(feat.chroma_stft(soundwave, sampling_rate))

    if "chroma_cqt" in features:
        features.append(feat.chroma_cqt(soundwave, sampling_rate))

    if "chroma_cens" in features:
        features.append(feat.chroma_cens(soundwave, sampling_rate))

    if "melspectrogram" in features:
        features.append(feat.melspectrogram(soundwave, sampling_rate))

    if "mfcc" in features:
        features.append(feat.mfcc(soundwave, sampling_rate))

    if "rmse" in features:
        features.append(feat.rmse(soundwave))

    if "spectral_centroid" in features:
        features.append(feat.spectral_centroid(soundwave, sampling_rate))

    if "spectral_bandwidth" in features:
        features.append(feat.spectral_bandwidth(soundwave, sampling_rate))

    if "spectral_contrast" in features:
        features.append(feat.spectral_contrast(soundwave, sampling_rate))

    if "spectral_flatness" in features:
        features.append(feat.spectral_flatness(soundwave))

    if "spectral_rolloff" in features:
        features.append(feat.spectral_rolloff(soundwave, sampling_rate))

    if "poly_features" in features:
        features.append(feat.poly_features(soundwave, sampling_rate))

    if "tonnetz" in features:
        features.append(feat.tonnetz(soundwave, sampling_rate))

    if "zero_crossing_rate" in features:
        features.append(feat.zero_crossing_rate(soundwave))


    return features










if __name__ == '__main__':
    pass
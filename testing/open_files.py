import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature as feat
from source.functions import import_sounds, save_all_features_for_all_files


def test_func(soundwave,sampling_rate,type="test"):

    print(type)

    feature_list=[]
    if len(feature_list)==0:
        feature_list=["chroma_stft","chroma_cqt","chroma_cens","melspectrogram",
                      "mfcc","rmse","spectral_centroid","spectral_bandwidth",
                      "spectral_contrast","spectral_flatness","spectral_rolloff",
                      "poly_features","tonnetz","zero_crossing_rate"]

    features=[]

    if "chroma_stft" in feature_list:
        features.append(feat.chroma_stft(soundwave, sampling_rate))

    if "chroma_cqt" in feature_list:
        features.append(feat.chroma_cqt(soundwave, sampling_rate))

    if "chroma_cens" in feature_list:
        features.append(feat.chroma_cens(soundwave, sampling_rate))

    if "melspectrogram" in feature_list:
        features.append(feat.melspectrogram(soundwave, sampling_rate))

    if "mfcc" in feature_list:
        features.append(feat.mfcc(soundwave, sampling_rate))

    if "rmse" in feature_list:
        features.append(feat.rmse(soundwave))

    if "spectral_centroid" in feature_list:
        features.append(feat.spectral_centroid(soundwave, sampling_rate))

    if "spectral_bandwidth" in feature_list:
        features.append(feat.spectral_bandwidth(soundwave, sampling_rate))

    if "spectral_contrast" in feature_list:
        features.append(feat.spectral_contrast(soundwave, sampling_rate))

    if "spectral_flatness" in feature_list:
        features.append(feat.spectral_flatness(soundwave))

    if "spectral_rolloff" in feature_list:
        features.append(feat.spectral_rolloff(soundwave, sampling_rate))

    if "poly_features" in feature_list:
        features.append(feat.poly_features(soundwave, sampling_rate))

    if "tonnetz" in feature_list:
        features.append(feat.tonnetz(soundwave, sampling_rate))

    if "zero_crossing_rate" in feature_list:
        features.append(feat.zero_crossing_rate(soundwave))



    return np.concatenate(features)








if __name__ == '__main__':

    path_for_sound_files="../sound_files/"
    path_to_save="/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy"

    save_all_features_for_all_files(path_for_sound_files,path_to_save)

    #
    # soundwaves,soundtypes,actual_file_names=import_sounds(path)
    #
    # features_of_all=[[test_func(soundwave,samplingrate,soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate) in enumerate(cl)]
    #                  for idx_cl,cl in enumerate(soundwaves)]
    #
    # np.save("/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy",features_of_all)
import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature as feat


def import_sounds(path_to_sound_files):


    # 1. get sound file names and 2. names of types of sounds
    sound_file_names=os.listdir(path_to_sound_files)
    sound_types=[name[4:-4] for name in sound_file_names]


    #getting test sound
    sci_sr,sci_y=sci_wav.read(path_to_sound_files+sound_file_names[0])
    sound_waves, sampling_rate = librosa.load(path_to_sound_files+sound_file_names[0],None)

    #processing file
    chroma_stft=feat.chroma_stft(sound_waves,sampling_rate)
    chroma_cqt=feat.chroma_cqt(sound_waves,sampling_rate)
    chroma_cens=feat.chroma_cens(sound_waves,sampling_rate)

    melspectrogram=feat.melspectrogram(sound_waves,sampling_rate)
    mfcc=feat.mfcc(sound_waves,sampling_rate)
    rmse=feat.rmse(sound_waves)

    spectral_centroid=feat.spectral_centroid(sound_waves,sampling_rate)
    spectral_bandwidth=feat.spectral_bandwidth(sound_waves,sampling_rate)
    spectral_contrast=feat.spectral_contrast(sound_waves,sampling_rate)
    spectral_flatness=feat.spectral_flatness(sound_waves)
    spectral_rolloff=feat.spectral_rolloff(sound_waves,sampling_rate)

    poly_features=feat.poly_features(sound_waves,sampling_rate)
    tonnetz=feat.tonnetz(sound_waves,sampling_rate)
    zero_crossing_rate=feat.zero_crossing_rate(sound_waves)



    librosa.feature.mfcc()


    return sound_waves,sound_types





if __name__ == '__main__':

    path="../sound_files/"
    sound_file_names=import_sounds(path)

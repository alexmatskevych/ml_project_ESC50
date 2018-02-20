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

    # get sound file names
    sound_file_names=np.array(os.listdir(path_to_sound_files))

    #get classes
    first_letters=np.array([word[0] for word in sound_file_names])
    classes=np.unique(first_letters)

    #sort by classes
    sorted_file_names=np.array([sound_file_names[np.where(first_letters==cl)[0]] for cl in classes ])
    sound_types=np.array([[name[4:-4] for name in cl] for cl in sorted_file_names])


    # load sound_files
    sound_waves = np.array([[librosa.load(path_to_sound_files+name, None) for name in cl] for cl in sorted_file_names])


    return sound_waves,sound_types,sound_file_names





def extract_features(soundwave,sampling_rate,feature_list=[],sound_name="test"):


    print(sound_name)

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



def save_all_features_for_all_files(path_to_files,path_to_save):
    """
    computes all paths for all soundfiles and saves them in a feature_vector to a file with the names

    :param path_to_files:
    :param path_to_save:
    :return:
    """
    soundwaves,soundtypes,actual_file_names=import_sounds(path_to_files)

    features_of_all=[[extract_features(soundwave,samplingrate,sound_name=soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate)
                      in enumerate(cl)] for idx_cl,cl in enumerate(soundwaves)]


    np.save(path_to_save,(features_of_all,soundtypes))



if __name__ == '__main__':
    pass
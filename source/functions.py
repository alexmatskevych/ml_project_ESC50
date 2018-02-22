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
    """
    extracts features with help of librosa
    :param soundwave: extracted soundwave from file
    :param sampling_rate: sampling rate
    :param feature_list: list of features to compute
    :param sound_name: type of sound, i.e. dog
    :return: np.array of all features for the soundwave
    """
    print(sound_name)

    if len(feature_list)==0:
        feature_list=["chroma_stft","chroma_cqt","chroma_cens","melspectrogram",
                      "mfcc","rmse","spectral_centroid","spectral_bandwidth",
                      "spectral_contrast","spectral_flatness","spectral_rolloff",
                      "poly_features","tonnetz","zero_crossing_rate"]

    features=[]
    now_len=0
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



def save_all_features_for_all_files(path_to_files,path_to_save=None):
    """
    computes all paths for all soundfiles and saves them in a feature_vector to a file with the names if wanted

    :param path_to_files:
    :param path_to_save:
    :return: 1. array with all features for all soundwaves and 2. all the according classes
    """
    soundwaves,soundtypes,actual_file_names=import_sounds(path_to_files)

    features_of_all=[[extract_features(soundwave,samplingrate,sound_name=soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate)
                      in enumerate(cl)] for idx_cl,cl in enumerate(soundwaves)]

    if path_to_save!=None:
        np.save(path_to_save,(features_of_all,soundtypes))

    return features_of_all,soundtypes


def same_class_means(features,classes):
    """
    compute mean and variance of the mean distance between same class for each feature


    :param features: array with features for each type of each class
    :param classes: array with class name for each type of each class
    :return: array with mean and std of each feature for each class
    """



    #compute all the mean features
    features_mean=np.array([np.mean(features_class,axis=-1) for features_class in features ])

    #this is the output array with all the feature means and variances for each class
    computed_mean_array=[]

    #iterate over each class
    for idx_cl,cl in enumerate(classes):

        #nothing to compute
        if len(cl)<2:
            continue

        #array containing all the differences between each of the types in one class
        class_vector=[]

        #iterate over each of the types
        for idx_type,type in enumerate(cl):

            #only look at the types which were not visited
            other_type_idx=np.arange(len(cl))[idx_type+1:]


            #compute differences to the other classes
            for idx_others in other_type_idx:

                #substract each type from another only once
                class_vector.append(np.abs(features_mean[idx_cl][idx_type]-features_mean[idx_cl][idx_others]))

        #compute mean and std
        computed_mean_array.append(np.array([np.mean(class_vector,axis=0),np.std(class_vector,axis=0)]).transpose())


    return computed_mean_array


if __name__ == '__main__':
    pass
import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature as feat
from source.functions import import_sounds, save_all_features_for_all_files,same_class_means,extract_features
import matplotlib.pyplot as plt

def visualize_same_class_features(mean_feature_vector,classes):


    for idx,class_features in enumerate(mean_feature_vector):

        plt.plot(class_features[:,0], class_features[:,1], 'ro')
        plt.xlabel('Mean')
        plt.ylabel('Std')
        plt.title("Class {}".format(idx))
        plt.show()


if __name__ == '__main__':

    path_for_sound_files="../sound_files/"
    path_to_save="/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy"
    sound_waves, sound_types, sound_file_names=import_sounds(path_for_sound_files)
    extract_features(sound_waves[0][0][0],sound_waves[0][0][1])

    features,classes=np.load(path_to_save)

    computed_mean_array=same_class_means(features,classes)

    visualize_same_class_features(computed_mean_array,classes)

    print(1+1)
    # save_all_features_for_all_files(path_for_sound_files,path_to_save)

    #
    # soundwaves,soundtypes,actual_file_names=import_sounds(path)
    #
    # features_of_all=[[test_func(soundwave,samplingrate,soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate) in enumerate(cl)]
    #                  for idx_cl,cl in enumerate(soundwaves)]
    #
    # np.save("/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy",features_of_all)
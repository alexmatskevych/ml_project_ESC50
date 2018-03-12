import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature as feat
from source.functions import import_sounds, save_all_features_for_all_files,same_class_means,\
    extract_features,visualize_same_class_features,visualize_distances_between_class_features,class_means_for_each_feature
import matplotlib.pyplot as plt

def visualize_same_class_features_alt(mean_feature_vector,classes):

    length_dict={
    "chroma_stft":12,
    "chroma_cqt":12,
    "chroma_cens":12,
    "malspectrogram":128,
    "mfcc":20,
    "rmse":1,
    "spectral_centroid":1,
    "spectral_bandwidth":1,
    "chroma_contrast":7,
    "spectral_flatness":1,
    "spectral_rolloff":1,
    "poly_features":2,
    "tonnetz":6,
    "zero_crossing_rate":1
    }

    color_dict={
    "chroma_stft":"red",
    "chroma_cqt":"gold",
    "chroma_cens":"palegreen",
    "malspectrogram":"violet",
    "mfcc":"mediumblue",
    "rmse":"orangered",
    "spectral_centroid":"teal",
    "spectral_bandwidth":"lightpink",
    "chroma_contrast":"black",
    "spectral_flatness":"sienna",
    "spectral_rolloff":"darkorange",
    "poly_features":"darkgreen",
    "tonnetz":"yellow",
    "zero_crossing_rate":"skyblue"
    }

    for idx,class_features in enumerate(mean_feature_vector):

        abs_len=0
        for type in length_dict:

            # #to many points for malspectrogram
            # if type=="malspectrogram":
            #     abs_len += length_dict[type]
            #     continue

            plt.plot(class_features[abs_len:abs_len+length_dict[type],0],
                     class_features[abs_len:abs_len+length_dict[type],1],
                     'ro',label=type,color=color_dict[type])

            abs_len+=length_dict[type]


        plt.xlabel('Mean')
        plt.ylabel('Std')
        plt.title("Class {}".format(idx))
        plt.legend()
        plt.show()




def visualize_same_class_features_combined(mean_feature_vector,classes):
    combine_each_single_feature=True


    length_array=[
    ["chroma_stft",12],
    ["chroma_cqt",12],
    ["chroma_cens",12],
    ["malspectrogram",128],
    ["mfcc",20],
    ["rmse",1],
    ["spectral_centroid",1],
    ["spectral_bandwidth",1],
    ["chroma_contrast",7],
    ["spectral_flatness",1],
    ["spectral_rolloff",1],
    ["poly_features",2],
    ["tonnetz",6],
    ["zero_crossing_rate",1]
    ]

    color_dict={
    "chroma_stft":"red",
    "chroma_cqt":"gold",
    "chroma_cens":"palegreen",
    "malspectrogram":"violet",
    "mfcc":"mediumblue",
    "rmse":"orangered",
    "spectral_centroid":"teal",
    "spectral_bandwidth":"lightpink",
    "chroma_contrast":"black",
    "spectral_flatness":"sienna",
    "spectral_rolloff":"darkorange",
    "poly_features":"darkgreen",
    "tonnetz":"yellow",
    "zero_crossing_rate":"skyblue"
    }

    mean_combined=np.mean(mean_feature_vector,axis=0)
    std_of_means=np.std(mean_feature_vector,axis=0)[:,0]

    new_feature_vector_combined=mean_combined
    new_feature_vector_combined[:,1]+=std_of_means



    abs_len=0
    for type,length in length_array:

        if combine_each_single_feature==False:
            plt.plot(new_feature_vector_combined[abs_len:abs_len+length,0],
                     new_feature_vector_combined[abs_len:abs_len+length,1],
                        'ro',label=type,color=color_dict[type])

        else:

            to_combine=new_feature_vector_combined[abs_len:abs_len+length]
            mean_tc=np.mean(to_combine,axis=0)
            std_of_mean_tc=np.std(to_combine,axis=0)[0]
            mean_tc[1]+=std_of_mean_tc

            plt.plot(mean_tc[0],mean_tc[1],'ro',label=type,color=color_dict[type])


        abs_len+=length

    plt.xlabel('Mean')
    plt.ylabel('Std')
    plt.title("All classes combined")
    plt.legend()
    plt.show()



if __name__ == '__main__':

    path_for_sound_files="../sound_files/"
    path_to_save="/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy"
    # sound_waves, sound_types, sound_file_names=import_sounds(path_for_sound_files)
    # extract_features(sound_waves[0][0][0],sound_waves[0][0][1])

    features,classes=np.load(path_to_save)

    classes=classes[:-1]
    features=np.array(features)

    computed_mean_array=class_means_for_each_feature(features,classes)

    visualize_same_class_features(computed_mean_array,True)

    visualize_distances_between_class_features(computed_mean_array,True)

    # save_all_features_for_all_files(path_for_sound_files,path_to_save)

    #
    # soundwaves,soundtypes,actual_file_names=import_sounds(path)
    #
    # features_of_all=[[test_func(soundwave,samplingrate,soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate) in enumerate(cl)]
    #                  for idx_cl,cl in enumerate(soundwaves)]
    #
    # np.save("/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy",features_of_all)
import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.feature as feat
from source.functions import import_sounds, save_all_features_for_all_files,same_class_inner_distance_means_for_each_class,\
    extract_features,visualize_same_class_features,visualize_distances_between_class_features,class_means_for_each_feature,\
    show_each_indice_for_each_feature,split_sound_files,split_wrapper,sort_classes,\
    compute_mean_distances_between_class_features,visualize_mean_features,test_out_rf
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

    # path_for_sound_files="/export/home/amatskev/ml_project_ESC50/sound_files/"

    path_for_sound_files="/mnt/localdata1/amatskev/environmental-sound-classification-50/split_files/"
    path_to_save="/mnt/localdata1/amatskev/environmental-sound-classification-50/features_of_all_scipy.npy"

    features_path="/mnt/localdata1/amatskev/environmental-sound-classification-50/features_of_all_sorted_split.npy"

    # sound_waves_librosa, sound_types, sound_file_names=import_sounds(path_for_sound_files,False)
    # sound_waves_scipy, sound_types, sound_file_names=import_sounds(path_for_sound_files,True)


    # save_all_features_for_all_files(path_for_sound_files,path_to_save)

    # sound_waves,sound_types

    # sound_waves=split_sound_files(sound_waves_scipy,sound_types)




    # features_scipy=extract_features(np.float64(sound_waves_scipy[0][0][1]),sound_waves_scipy[0][0][0])
    # features_librosa=extract_features(sound_waves_librosa[0][0][0],sound_waves_librosa[0][0][1])
    #
    #
    features,classes=np.load(features_path)

    test_out_rf(features,classes)
    # features=np.array([[[np.array(feature) for feature in type] for type in cl] for cl in features])

    # new_sorted=sort_classes(features,classes)
    #
    # np.save("/mnt/localdata1/amatskev/environmental-sound-classification-50/features_of_all_sorted_split.npy",new_sorted)
    # np.save(features_path,(features, classes))
    # features, classes=np.load("/mnt/localdata1/amatskev/environmental-sound-classification-50/features_of_all_sorted_split_test.npy")

    #
    # classes=classes[:-1]
    # features=np.array(features)
    #
    # show_each_indice_for_each_feature(features,classes)

    # computed_mean_array_inner_distances=same_class_inner_distance_means_for_each_class(features,classes)
    # visualize_same_class_features(computed_mean_array_inner_distances,np.concatenate(classes),True,False)
    #
    #
    #
    # computed_mean_array_for_each_class=class_means_for_each_feature(features,classes)
    # mean_distances_between_class_features=compute_mean_distances_between_class_features(computed_mean_array_for_each_class,np.concatenate(classes),True)
    #
    # leave_out_features=["malspectrogram","spectral_rolloff","spectral_centroid","spectral_bandwidth","poly_features"]
    #
    # visualize_mean_features(np.mean(computed_mean_array_inner_distances,axis=0),"Inner distances",leave_out_features)
    # visualize_mean_features(np.mean(mean_distances_between_class_features,axis=0),"Distances between classes",leave_out_features)

    # save_all_features_for_all_files(path_for_sound_files,path_to_save)

    #
    # soundwaves,soundtypes,actual_file_names=import_sounds(path)
    #
    # features_of_all=[[test_func(soundwave,samplingrate,soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(soundwave,samplingrate) in enumerate(cl)]
    #                  for idx_cl,cl in enumerate(soundwaves)]
    #
    # np.save("/mnt/localdata1/amatskev/debugging_ml_project/features_of_all.npy",features_of_all)
import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa
import librosa.feature as feat
import matplotlib.pyplot as plt

def get_neccecary(return_length=True,return_color=True):
    """

    :param return_length: whether we return length array
    :param return_color: whether we return color dict
    :return: the color and length dict, to know where each feature starts and ends and which color it has
    """
    length_array = np.array([
        ["chroma_stft", 12],
        ["chroma_cqt", 12],
        ["chroma_cens", 12],
        ["malspectrogram", 128],
        ["mfcc", 20],
        ["rmse", 1],
        ["spectral_centroid", 1],
        ["spectral_bandwidth", 1],
        ["chroma_contrast", 7],
        ["spectral_flatness", 1],
        ["spectral_rolloff", 1],
        ["poly_features", 2],
        ["tonnetz", 6],
        ["zero_crossing_rate", 1]])

    color_dict = {
        "chroma_stft": "red",
        "chroma_cqt": "gold",
        "chroma_cens": "palegreen",
        "malspectrogram": "violet",
        "mfcc": "mediumblue",
        "rmse": "orangered",
        "spectral_centroid": "teal",
        "spectral_bandwidth": "lightpink",
        "chroma_contrast": "black",
        "spectral_flatness": "sienna",
        "spectral_rolloff": "darkorange",
        "poly_features": "darkgreen",
        "tonnetz": "yellow",
        "zero_crossing_rate": "skyblue"
    }

    if return_length and return_color:
        return length_array,color_dict

    elif return_length and not return_color:
        return length_array

    elif not return_length and return_color:
        return color_dict




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


    #feature_len
    #"chroma_stft":12
    if "chroma_stft" in feature_list:
        features.append(feat.chroma_stft(soundwave, sampling_rate))

    #"chroma_cqt":12
    if "chroma_cqt" in feature_list:
        features.append(feat.chroma_cqt(soundwave, sampling_rate))

    #"chroma_cens":12
    if "chroma_cens" in feature_list:
        features.append(feat.chroma_cens(soundwave, sampling_rate))

    #"malspectrogram":128
    if "melspectrogram" in feature_list:
        features.append(feat.melspectrogram(soundwave, sampling_rate))

    #"mfcc":20
    if "mfcc" in feature_list:
        features.append(feat.mfcc(soundwave, sampling_rate))

    #"rmse":1
    if "rmse" in feature_list:
        features.append(feat.rmse(soundwave))

    #"spectral_centroid":1
    if "spectral_centroid" in feature_list:
        features.append(feat.spectral_centroid(soundwave, sampling_rate))

    #"spectral_bandwidth":1
    if "spectral_bandwidth" in feature_list:
        features.append(feat.spectral_bandwidth(soundwave, sampling_rate))

    #"chroma_contrast":7
    if "spectral_contrast" in feature_list:
        features.append(feat.spectral_contrast(soundwave, sampling_rate))

    #"spectral_flatness":1
    if "spectral_flatness" in feature_list:
        features.append(feat.spectral_flatness(soundwave))

    #"spectral_rolloff":1
    if "spectral_rolloff" in feature_list:
        features.append(feat.spectral_rolloff(soundwave, sampling_rate))

    #"poly_features":2
    if "poly_features" in feature_list:
        features.append(feat.poly_features(soundwave, sampling_rate))

    #"tonnetz":6
    if "tonnetz" in feature_list:
        features.append(feat.tonnetz(soundwave, sampling_rate))

    #"zero_crossing_rate":1
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


    return np.array(computed_mean_array)




def visualize_same_class_features(mean_feature_vector, combine_each_single_feature=True):
    """


    :param mean_feature_vector: all the features of each class stacked
    :param combine_each_single_feature: wehether we show one combined point
            for each class or for each index of each array (nearly each feature has more than one index in the feature array)
    :return: nothing
    """

    #get the color and length dict, to know where each feature starts and ends
    length_array,color_dict=get_neccecary()

    #combine all classes into one
    mean_combined=np.mean(mean_feature_vector,axis=0)

    #compute the std of the mean
    std_of_means=np.std(mean_feature_vector,axis=0)[:,0]

    #create new array and add the mean of the old stds to the std of the new means
    new_feature_vector_combined=mean_combined
    new_feature_vector_combined[:,1]+=std_of_means



    abs_len=0

    #iterate for each feature
    for type,length in length_array:

        #if we show more than one dot per feature or not
        if combine_each_single_feature==False:

            #plot the dots
            plt.plot(new_feature_vector_combined[abs_len:abs_len+length,0],
                     new_feature_vector_combined[abs_len:abs_len+length,1],
                        'ro',label=type,color=color_dict[type])


        #if we show only one dot per feature or not
        else:

            #create new array for current feature
            to_combine=new_feature_vector_combined[abs_len:abs_len+length]

            #compute mean for the array
            mean_tc=np.mean(to_combine,axis=0)

            #again compute the std of new mean
            std_of_mean_tc=np.std(to_combine,axis=0)[0]

            #add the std of the new means to the mean of the old stds
            mean_tc[1]+=std_of_mean_tc

            #plot the dots
            plt.plot(mean_tc[0],mean_tc[1],'ro',label=type,color=color_dict[type])

        #prepare length for next iteration
        abs_len+=length

    #construct labels for graph
    plt.xlabel('Mean')
    plt.ylabel('Std')
    plt.title("All classes combined")
    plt.legend()
    
    #show graph
    plt.show()


if __name__ == '__main__':
    pass
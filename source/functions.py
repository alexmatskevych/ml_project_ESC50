import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa
import librosa.feature as feat
import matplotlib.pyplot as plt
from librosa.effects import split
from sklearn.ensemble import RandomForestClassifier



def read_and_split_file(path_file,file_name,output_path):

    rate,wave=sci_wav.read(path_file+file_name)

    [sci_wav.write(output_path+file_name[:-4]+"_{}.wav".format(int(nr/80000)),16000,wave[nr:nr+80000])
    for nr in range(0,3200000,80000)]



def split_wrapper(path_to_sound_files,output_path):


    sound_file_names=np.array(os.listdir(path_to_sound_files))

    [read_and_split_file(path_to_sound_files,file_name,output_path)
     for file_name in sound_file_names]


def get_neccecary():
    """

    :param return_length: whether we return length array
    :param return_color: whether we return color dict
    :return: the color and length dict, to know where each feature starts and ends and which color it has
    """
    length_array = [
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
        ["zero_crossing_rate", 1]]

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
        "zero_crossing_rate": "skyblue",
        "15": "tan",
        "16": "bisque",
        "17": "crimson",
        "18": "goldenrod",
        "19": "darkslateblue",
        "20": "lavender",

    }

    classes=np.array(["Animals", "Environment", "Human noises", "Tools", "Machines"])


    return length_array,color_dict,classes



def split_sound_files(sound_waves,sound_types):

    test_sound=np.float64(sound_waves[0][0][1])
    splits=split(test_sound, 10,frame_length=10)

    [sci_wav.write("/export/home/amatskev/testlibrosa/second_test{}.wav".format(idx),
                              16000,np.uint8(test_sound[splits[idx][0]:splits[idx][1]])) for idx,intervall in enumerate(splits)]


    print(1+2)
    print(1+2)




def import_sounds(path_to_sound_files,scipy=True):
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
    if scipy==True:
        sound_waves = [[sci_wav.read(path_to_sound_files+name) for name in cl] for cl in sorted_file_names]

    else:
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

    features_of_all=[[extract_features(np.float64(soundwave),samplingrate,sound_name=soundtypes[idx_cl][idx_soundwave]) for idx_soundwave,(samplingrate,soundwave)
                      in enumerate(cl)] for idx_cl,cl in enumerate(soundwaves)]

    if path_to_save!=None:
        np.save(path_to_save,(features_of_all,soundtypes))

    return features_of_all,soundtypes


def same_class_inner_distance_means_for_each_class(features,classes,contract_to_single_features=True):
    """
    compute mean and variance of the mean distance between all the types in the same class for each feature


    :param features: array with features for each type of each class
    :param classes: array with class name for each type of each class
    :return: array with mean and std of each feature for each class
    """

    #sort out the single features
    features_without_single=np.array([feature for feature in features if len(feature) > 1])

    # compute all the mean features and stds for the ~6000 values for each feature
    features_mean = np.mean(features_without_single, axis=-1)
    features_mean_std = np.std(features_without_single, axis=-1)


    #this is the output array with all the feature means and variances for each class
    computed_mean_array=[]




    #iterate over each class
    for idx_cl,cl in enumerate(features_mean):


        #array containing all the differences between each of the types in one class
        class_vector=[]
        class_vector_std=[]
        #iterate over each of the types
        for idx_type,type in enumerate(cl):

            #only look at the types which were not visited
            other_type_idx=np.arange(len(cl))[idx_type+1:]


            #compute differences to the other classes
            for idx_others in other_type_idx:

                #substract each type from another only once
                class_vector.append(np.abs(features_mean[idx_cl][idx_type]-features_mean[idx_cl][idx_others]))
                class_vector_std.append(np.abs(features_mean_std[idx_cl][idx_type]-features_mean_std[idx_cl][idx_others]))

        #compute mean and std
        computed_mean_array.append(np.array([np.mean(class_vector,axis=0),
                                             np.std(class_vector,axis=0)+np.mean(class_vector_std,axis=0)]).transpose())

    #add the stds from the mean of ~6000 values for each feature
    for idx, mean_std in enumerate(features_mean_std):
        if len(mean_std) > 1:
            computed_mean_array[idx][:, 1] += np.mean(features_mean_std[idx], axis=0)


    if contract_to_single_features:

        features_combined=[]

        computed_mean_array=np.array(computed_mean_array)

        length_array,color_dict,_=get_neccecary()
        abs_len=0

        # iterate for each feature
        for type, length in length_array:
            # create new array for current feature for all classes
            to_combine = computed_mean_array[:, abs_len:abs_len + length,0]
            old_std_to_combine = computed_mean_array[:, abs_len:abs_len + length,1]

            #compute mean for the array
            mean_tc = np.mean(to_combine, axis=1)

            #compute the std of new mean
            std_of_mean_tc = np.std(to_combine, axis=1)

            # compute the mean of old std
            std_of_mean_tc += np.mean(old_std_to_combine, axis=1)

            # append in array
            features_combined.append([mean_tc, std_of_mean_tc])

            # prepare length for next iteration
            abs_len += length

        return np.array(features_combined).swapaxes(1,2).swapaxes(0,1)





    return np.array(computed_mean_array)


def class_means_for_each_feature(features,classes):
    """

    :param features: features raw
    :param classes: class names and types
    :return: features for each class with dimensions: (class,feature(already mean, so 1 indice for each feature), type(mean or std))
    """


    length_array,_,_=get_neccecary()

    #sort out the single features
    features_without_single=np.array([feature for feature in features if len(feature) > 1])

    # compute all the mean features and stds for the ~6000 values for each feature
    features_mean = np.mean(features_without_single, axis=-1)
    features_mean_std = np.std(features_without_single, axis=-1)

    #compute means and stds for each class
    class_features_mean=np.mean(features_mean, axis=1)
    class_features_mean_std=np.std(features_mean, axis=1)

    #add old stds to new stds
    mean_of_old_std = np.mean(features_mean_std, axis=1)
    class_features_mean_std+=mean_of_old_std

    abs_len=0

    #for later returning
    features_combined=[]



    # iterate for each feature
    for type, length in length_array:


        # create new array for current feature for all classes
        to_combine = class_features_mean[:,abs_len:abs_len + length]
        old_std_to_combine=class_features_mean_std[:,abs_len:abs_len + length]

        # compute mean for the array
        mean_tc = np.mean(to_combine, axis=1)

        # again compute the std of new mean
        std_of_mean_tc = np.std(to_combine, axis=1)

        #compute the mean of old std
        std_of_mean_tc+=np.mean(old_std_to_combine, axis=1)

        #append in array
        features_combined.append([mean_tc,std_of_mean_tc])

        # prepare length for next iteration
        abs_len += length


    #fiddeling with axes so that we have (class,feature,type={mean,std})
    features_combined=np.array(features_combined)
    features_combined=features_combined.transpose()
    features_combined=np.swapaxes(features_combined,1,2)

    return features_combined

def visualize_same_class_features(mean_feature_vector,classes=[], combine_each_single_feature=True,show_for_each_single_class=True):
    """


    :param mean_feature_vector: all the features of each class stacked
    :param combine_each_single_feature: wehether we show one combined point
            for each class or for each index of each array (nearly each feature has more than one index in the feature array)
    :return: nothing
    """

    #get the color and length dict, to know where each feature starts and ends
    if len(classes)==0:

        length_array,color_dict,classes=get_neccecary()
    else:
        length_array, color_dict, _ = get_neccecary()


    if show_for_each_single_class==False:

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
                if type != "malspectrogram":

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

    else:

        for cl_idx,cl in enumerate(mean_feature_vector):


            abs_len = 0

            # iterate for each feature
            for type, length in length_array:

                # if we show more than one dot per feature or not
                if combine_each_single_feature == False:

                    # plot the dots
                    plt.plot(cl[abs_len:abs_len + length, 0],
                             cl[abs_len:abs_len + length, 1],
                             'ro', label=type, color=color_dict[type])


                # if we show only one dot per feature or not
                else:

                    # create new array for current feature
                    to_combine = cl[abs_len:abs_len + length]

                    # compute mean for the array
                    mean_tc = np.mean(to_combine, axis=0)

                    # again compute the std of new mean
                    std_of_mean_tc = np.std(to_combine, axis=0)[0]

                    # add the std of the new means to the mean of the old stds
                    mean_tc[1] += std_of_mean_tc

                    if type!="malspectrogram":
                        # plot the dots
                        plt.plot(mean_tc[0], mean_tc[1], 'ro', label=type, color=color_dict[type])

                # prepare length for next iteration
                abs_len += length

            # construct labels for graph
            plt.xlabel('Mean')
            plt.ylabel('Std')
            plt.title(classes[cl_idx])
            plt.legend()

            # show graph
            plt.show()


def compute_mean_distances_between_class_features(mean_feature_vector, classes=[], show_single_differences=True):
    """


    :param mean_feature_vector: feature vector with features for each class with
            dimensions: (class,feature(already mean, so 1 indice for each feature), type(mean or std))
    :param show_single_differences: wehether we show each of the differences between each class
    :return: nothing
    """

    # get the color and length dict, to know where each feature starts and ends
    if len(classes) == 0:

        length_array, color_dict, classes = get_neccecary()
    else:
        length_array, color_dict, _ = get_neccecary()

    feature_distances_means = np.concatenate([np.abs(cl - mean_feature_vector[cl_idx + 1:])
                                              for cl_idx, cl in enumerate(mean_feature_vector[:-1])])


    return feature_distances_means

def visualize_mean_features(features,title="test",leave_out_features=[]):

    length_array,color_dict,_=get_neccecary()

    for idx,feature_type in enumerate(length_array):

        if feature_type[0] not in leave_out_features:
            plt.plot(features[idx,0], features[idx,1], 'ro', label=feature_type[0],
                     color=color_dict[feature_type[0]])

    plt.xlabel('Mean')
    plt.ylabel('Std')
    plt.title(title)
    plt.legend()

    # show graph
    plt.show()


def visualize_distances_between_class_features(mean_feature_vector,classes=[], show_single_differences=True):
    """


    :param mean_feature_vector: feature vector with features for each class with
            dimensions: (class,feature(already mean, so 1 indice for each feature), type(mean or std))
    :param show_single_differences: wehether we show each of the differences between each class
    :return: nothing
    """

    #get the color and length dict, to know where each feature starts and ends
    if len(classes)==0:

        length_array,color_dict,classes=get_neccecary()
    else:
        length_array, color_dict, _ = get_neccecary()


    feature_distances_means = np.concatenate([np.abs(cl - mean_feature_vector[cl_idx + 1:])
                               for cl_idx,cl in enumerate(mean_feature_vector[:-1])])


    diff_names=[]
    for current_cl_idx,current_cl in enumerate(classes):
        for cl in classes[current_cl_idx+1:]:
            diff_names.append(current_cl+"<-->"+cl)


    if show_single_differences:

        # iterate for each feature
        for diff_idx,distances_mean in enumerate(feature_distances_means):

            for single_feature_idx,single_feature in enumerate(distances_mean):

                if length_array[single_feature_idx][0]=='malspectrogram':
                    continue

                # plot the dots
                plt.plot(single_feature[0],
                         single_feature[1],
                        'ro', label=length_array[single_feature_idx][0], color=color_dict[length_array[single_feature_idx][0]])

            # construct labels for graph
            plt.xlabel('Mean')
            plt.ylabel('Std')
            plt.title(diff_names[diff_idx])
            plt.legend()

            # show graph
            plt.show()


def plot_difference(features1,features2,std1,std2,name1,name2):

    length_array,color_dict,overlaying_classes=get_neccecary()

    color_dict_keys=list(color_dict.keys())

    difference=np.abs(features1-features2)
    difference_std=np.abs(std1-std2)

    [plt.plot(difference[idx],
             difference_std[idx],
             'ro', label=str(idx), color=color_dict[color_dict_keys[idx]]) for idx,bla in enumerate(difference)]

    # construct labels for graph
    plt.xlabel('Mean')
    plt.ylabel('Std')
    plt.title(name1+"<-->"+name2)
    plt.legend()

    # show graph
    plt.show()


def show_each_indice_for_each_feature(features,single_classes,desired_feature="mfcc"):

    features=np.array([[[np.array(feature) for feature in type] for type in cl] for cl in features])

    length_array,color_dict,overlaying_classes=get_neccecary()

    length_array=np.array(length_array)

    #find desired feature
    idx_feature=np.where(length_array[:, 0] == desired_feature)[0][0]

    #find start index and length of feature
    start_index=np.sum(np.uint(length_array[:idx_feature, 1]))
    length_of_feature= np.uint(length_array[idx_feature, 1])

    #extracting only desired feature for all classes
    feature=features[:,:,start_index:start_index+length_of_feature,:]

    #concatenating all classes into a mix
    feature=np.concatenate(feature)
    single_classes=np.concatenate(single_classes)

    #mean and std of ~6000 features
    feature_mean=np.mean(feature,axis=-1)
    feature_std=np.std(feature,axis=-1)

    for idx_cl,cl in enumerate(feature_mean):

        for idx_other_cl,other_cl in enumerate(feature_mean[idx_cl+1:]):

            plot_difference(cl,other_cl,feature_std[idx_cl],feature_std[idx_other_cl],
                            single_classes[idx_cl],single_classes[idx_cl+1+idx_other_cl])





def sort_classes(features,classes):


    def sort_and_return(overclass_features,overclass):


        mask=np.argsort(overclass)
        features_sorted=overclass_features[mask]
        overclass_sorted=overclass[mask]

        features_overworked=np.array([features_sorted[idx:idx+40] for idx in range(0,400,40)])
        classes_overworked=np.array([overclass_sorted[idx][:-2]  for idx in range(0,400,40) ])

        return features_overworked,classes_overworked

    stacked=[sort_and_return(features[idx],cl) for idx,cl in enumerate(classes)]

    sorted_features=np.array([st[0] for st in stacked])
    sorted_classes=np.array([st[1] for st in stacked])



    return np.concatenate(sorted_features),sorted_classes


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(500,n_jobs=32)
    clf.fit(features, target)
    return clf


#test out rf
def test_out_rf(features,classes):


    features_test   =[np.concatenate(cl) for cl in features[:,39]]
    features_train  =[np.concatenate(cl) for cl in np.concatenate(features[:,:39],axis=0)]

    classes_test=np.concatenate(classes)
    classes_train=np.concatenate([[cl]*39 for cl in np.concatenate(classes)])


    print("TRAIN RF NOW")

    clf=random_forest_classifier(features_train,classes_train)

    print("PREDICT WITH RF NOW")
    rf_predictions=clf.predict(features_test)

    wrong=0
    right=0

    for idx,prediction in enumerate(rf_predictions):
        print(rf_predictions[idx],classes_test[idx])
        if rf_predictions[idx]==classes_test[idx]:
            right+=1
        else:
            wrong+=1

    print("RIGHT:",right)
    print("WRONG:",wrong)



    np.save("/mnt/localdata1/amatskev/environmental-sound-classification-50/rf_predictions.npy",rf_predictions)
    np.save("/mnt/localdata1/amatskev/environmental-sound-classification-50/classes_test.npy",classes_test)




if __name__ == '__main__':
    pass
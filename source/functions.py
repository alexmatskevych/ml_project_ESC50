import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa

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

def extract_features(soundwaves):


    features=[]

    return features










if __name__ == '__main__':
    pass
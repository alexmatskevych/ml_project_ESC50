import sys
sys.path.append('../')
import os
from scipy import signal
import scipy.io.wavfile as sci_wav
import matplotlib.pyplot as plt
import numpy as np
import librosa


def import_sounds(path_to_sound_files):


    # 1. get sound file names and 2. names of types of sounds
    sound_file_names=os.listdir(path_to_sound_files)
    sound_types=[name[4:-4] for name in sound_file_names]


    #getting test sound
    y, sr = librosa.load(path_to_sound_files+sound_file_names[0],None)

    #processing file


    librosa.feature.mfcc()


    return sound_file_names





if __name__ == '__main__':

    path="../sound_files/"
    sound_file_names=import_sounds(path)

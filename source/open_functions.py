import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os

def import_sounds(path_to_sound_files):


    # 1. get sound file names and 2. names of types of sounds
    sound_file_names=os.listdir(path_to_sound_files)
    sound_types=[name[4:-4] for name in sound_file_names]


    return sound_file_names
from pupil_parse.preprocess_utils import config as cf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cf.plot_config()

def plot_gaze(samples, subsample=2000):
    plt.figure()
    plt.plot(samples.gaze_x[:subsample], samples.gaze_y[:subsample])

def gaze_correct(samples, camera_eye_theta, eye_stim_theta):

    # some code here to implement correction from Hayes 2016

from pupil_parse.preprocess_utils import config as cf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cf.plot_config()

def plot_gaze(samples, subsample):
    plt.plot(samples.pupil_x, samples.pupil_y)

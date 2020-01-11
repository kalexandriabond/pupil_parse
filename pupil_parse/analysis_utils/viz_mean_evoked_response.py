import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sys import platform

import os
import numpy as np
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import glob
import scipy.stats as stats
jtplot.style('grade3', context='poster', fscale=1.4, spines=False, gridlines='--')


# # TODO: functionify. implement across subjects. 
# # collapsed across conditions
# bootstrap_start = time.time()
# ax = sns.lineplot(x='trial_sample_shift', y='z_pupil_diameter', hue='condition_test',
#                   data=sub_sample_df, palette='muted')
# plt.vlines(x=np.array([stim_onset, stim_offset]), ymin=-1, ymax=1)
# bootstrap_end = time.time()
#
# time_to_plot = (bootstrap_end - bootstrap_start)
#
# print('time to plot bootstrapped CIs: ', time_to_plot)

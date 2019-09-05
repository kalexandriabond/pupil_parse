from pupil_parse.preprocess_utils import config as cf

from scipy.signal import find_peaks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
import pandas as pd


import time

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

(raw_data_path, intermediate_data_path,
processed_data_path, figure_path) = cf.path_config()

cf.plot_config()


def find_peak(samples, width=100):

    """ Find the peak of the pupillary response within the trial. """
    peak_idx, _ = find_peaks(samples.z_pupil_diameter, width=width)
    samples['peak_samples'] = np.nan

    if peak_idx.any():
        peaks = samples.z_pupil_diameter.iloc[peak_idx]
        max_peak_idx = peaks.idxmax()
        samples['peak_samples'][max_peak_idx] = samples.z_pupil_diameter[max_peak_idx]
        print('peak values ', samples.z_pupil_diameter[max_peak_idx],
        'peak indices ', max_peak_idx)
    else:
        print('No peak found for this trial.')

    print('peak_samples ', samples.peak_samples.unique())

    return samples

def calc_peaks(samples, stim_offset=2000, stim_onset=500, df=None, save=None):

    trial_samples = samples.loc[(samples.trial_sample >= stim_onset) &
    (samples.trial_sample < stim_offset)]

    trial_peaks = trial_samples.groupby('trial_epoch').apply(find_peak).reset_index()
    print(trial_peaks.head())

    trial_df = pd.DataFrame()

    if df:
        trial_df = df

    trial_df['trial_peaks'] = trial_peaks

    if save:
        trial_df.to_csv(os.path.join(processed_data_path, df_name + '.csv'))

    return trial_df

def find_mean(samples, subj_id, session_n, reward_code,
 stim_onset=500, stim_offset=2000, id_str=None, df=None,
 save=None):

    trial_samples = samples.loc[(samples.trial_sample >= stim_onset) &
    (samples.trial_sample < stim_offset)]

    df_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code) + '_trial')

    if id_str:
        fig_name = fig_name + '_' + id_str


    trial_df = pd.DataFrame()

    if df:
        trial_df = df

    trial_means = trial_samples.groupby('trial_epoch').z_pupil_diameter.mean()
    print(trial_means)

    print('means found, storing ...')

    trial_df['trial_mean'] = trial_means

    if save:
        trial_df.to_csv(os.path.join(processed_data_path, df_name + '.csv'))

    return trial_df


def plot_extrema(samples, subj_id, session_n, reward_code, id_str=None):
    """ Check peak detection. """

    fig_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code) + '_trial')

    if id_str:
        fig_name = fig_name + '_' + id_str

    plt.ioff()
    fig=plt.figure()
    plt.title(fig_name)
    plt.scatter('trial_sample', "z_pupil_diameter", data=samples, s=5)
    plt.scatter('trial_sample', "peak_samples", s=500, marker='x',
    color='red', data=samples)
    plt.ylabel('Pupil diameter (a.u.)')
    plt.xlabel('Time (ms)')
    plt.close()

    return fig_name, fig


def save_extrema(fig_name, figures, figure_path=figure_path):
    """ Save images of peaks. """

    pdf = PdfPages(os.path.join(figure_path, fig_name + '.pdf'))
    for fig in figures:
        pdf.savefig(fig, bbox_inches='tight')
    pdf.close()

    return None

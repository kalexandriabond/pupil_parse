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
processed_data_path, figure_path, simulated_data_path) = cf.path_config()

cf.plot_config()


def calc_peaks(samples, width=100):

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

def locate_peaks(samples,  subj_id, session_n, reward_code,
stim_offset=2000, stim_onset=500,
df=None, n_trials=398, save=None, method='linear', processed_data_path=processed_data_path):

    trial_samples_df = samples.loc[(samples.trial_sample >= stim_onset) &
    (samples.trial_sample < stim_offset)]

    peak_df = trial_samples_df.groupby('trial_epoch').apply(calc_peaks).reset_index(drop=True)

    if save:
        df_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
        str(session_n) +  '_cond-' + str(reward_code) + '_trial_' + 'peaks')
        sparse_peak_df = peak_df[['peak_samples', 'trial_epoch']]
        sparse_peak_df = sparse_peak_df.groupby('trial_epoch').peak_samples.unique().reset_index() # get only one val per trial

        peak_samples_clean = []
        for trial in sparse_peak_df.peak_samples:
            peak_samples_clean.append(trial[-1])


        sparse_peak_df['peak_amplitude'] = peak_samples_clean
        sparse_peak_df['peak_amplitude_interp'] = sparse_peak_df.peak_amplitude.interpolate(method=method)
        sparse_peak_df.drop(columns=['peak_samples'], inplace=True)

        assert len(sparse_peak_df) == n_trials, 'check len of sparse_peak_df'

        sparse_peak_df.to_csv(os.path.join(processed_data_path, df_name + '.csv'),
        index=False)

    return peak_df

def find_mean(samples, subj_id, session_n, reward_code,
 stim_onset=500, stim_offset=2000, outcome_onset=1250,
 outcome_offset=2000, id_str=None, df=None,
 save=None):

    outcome_duration = outcome_offset - outcome_onset

    trial_samples = samples.loc[(samples.trial_sample >= stim_onset) &
    (samples.trial_sample < stim_offset)]

    outcome_samples = trial_samples.loc[(trial_samples.trial_sample >= outcome_onset) &
    (trial_samples.trial_sample < outcome_offset)]

    outcome_samples_early = trial_samples.loc[(trial_samples.trial_sample >= outcome_onset) &
    (trial_samples.trial_sample < (outcome_onset + (outcome_duration/2)))]

    outcome_samples_late = trial_samples.loc[(trial_samples.trial_sample >= outcome_onset + (outcome_duration/2)) &
    (trial_samples.trial_sample <  outcome_offset)]


    df_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code) + '_trial_means')

    if id_str:
        fig_name = fig_name + '_' + id_str


    trial_df = pd.DataFrame()

    if df:
        trial_df = df

    trial_means = trial_samples.groupby('trial_epoch').z_pupil_diameter.mean()
    print(trial_means)
    outcome_means = outcome_samples.groupby('trial_epoch').z_pupil_diameter.mean()
    print(outcome_means)

    early_outcome_means = outcome_samples_early.groupby('trial_epoch').z_pupil_diameter.mean()
    late_outcome_means = outcome_samples_late.groupby('trial_epoch').z_pupil_diameter.mean()

    outcome_change = late_outcome_means - early_outcome_means

    print('means found, storing ...')

    trial_df['trial_mean'] = trial_means
    trial_df['early_outcome_means'] = early_outcome_means
    trial_df['late_outcome_means'] = late_outcome_means
    trial_df['outcome_change'] = outcome_change

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

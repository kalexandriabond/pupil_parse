import numpy as np
from jupyterthemes import jtplot
jtplot.style('grade3', context='poster', fscale=1.4, spines=False, gridlines='--')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
import pandas as pd
import os
pd.options.mode.chained_assignment = None  # default='warn'


def preprocess_message_df(message_df, message_filename):
    """Preprocess the message dataframe."""
    preprocessed_message_df = message_df[1:-1]
    print(preprocessed_message_df.shape)

    preprocessed_message_df['trial_response_time'] = preprocessed_message_df.relative_response_time - preprocessed_message_df.relative_stim_onset_time

    subj_id = message_filename[4:7]
    condition = message_filename[-8:-4]

    preprocessed_message_df['subj_id'] = subj_id
    preprocessed_message_df['condition'] = condition
    preprocessed_message_df['trial'] = np.arange(0,398)

    return preprocessed_message_df


def preprocess_sample_df(sample_df, sample_filename, baseline_interval=500,
trial_end=1500, trial_begin=0):
    """Preprocess the samples dataframe, selecting an interval of interest."""
    subj_id = sample_filename[4:7]
    condition = sample_filename[-8:-4]

    sample_df['subj_id'] = subj_id
    sample_df['condition'] = condition
    sample_df['trial_sample_shift'] = sample_df.trial_sample - baseline_interval
    sample_df['condition_hack'] = ["$%s$" % x for x in sample_df["condition"]]

    sample_df['vol'] = sample_df.condition_hack.str[-3:-1]
    sample_df['conf'] = sample_df.condition_hack.str[1:3]

    preprocessed_sample_df = sample_df.loc[(sample_df.trial_sample_shift <= trial_end) &
                      (sample_df.trial_sample_shift >= trial_begin)]

    return preprocessed_sample_df

def calc_mean(preprocessed_sample_df, preprocessed_message_df):
    """Calculate mean pupil diameter within a time interval.
    The default is the trial interval."""

    trial_means = preprocessed_sample_df.groupby(['trial_epoch'])[['z_pupil_diameter', 'pupil_diameter',
                                   'highpass_pupil_diameter', 'lowpass_pupil_diameter']].mean().reset_index()

    trial_means['subj_id'] = preprocessed_message_df.subj_id.values
    trial_means['condition'] = preprocessed_message_df.condition.values

    return trial_means

def find_peaks(preprocessed_sample_df,
          peak_width_min=100, conservative_curve_detection=False,
              summarize_first_peak=True):
    """Find peak pupil diameter within a time interval.
    The default is the trial interval and the default peak width is 100 ms at minimum.

    Returns peak_amplitude, peak_width, beginning and end of peak, and peak index.
    Note that the index is relative to the _onset_ of the trial / stimulus."""

    import scipy.signal

    peak_amplitudes = []
    peak_widths = []
    peak_begins = []
    peak_ends = []
    peak_idxs = []

    for trial_epoch in preprocessed_sample_df.trial_epoch.unique():

        trial_samples = preprocessed_sample_df.loc[preprocessed_sample_df.trial_epoch == trial_epoch]

        peak_idx, peak_properties = scipy.signal.find_peaks(trial_samples.z_pupil_diameter,
                                                            width=peak_width_min)

        peak_amplitude = trial_samples.z_pupil_diameter.iloc[peak_idx]

        if peak_amplitude.size is 0:
            peak_amplitude = np.nan
        else:
            peak_amplitude = peak_amplitude.iloc[0]

        if peak_properties['widths'].size is 0:
            peak_width = np.nan
        else:
            peak_width = peak_properties['widths'][0]

        # find beginning and end of curve
        # note that this is fairly conservative (large window)
        if conservative_curve_detection:
            peak_end = peak_properties['right_bases']
            peak_begin = peak_properties['left_bases']
        else:
            # if a tighter window is required, then can assume a symmetrical curve
            # and use the peak width + peak amplitude to find beginning and end of curve
            peak_end = peak_idx + (peak_width/2)
            peak_begin = peak_idx - (peak_width/2)


        if peak_end.size is 0:
            peak_end = np.nan
            peak_begin = np.nan
            peak_idx = np.nan

        if peak_end is not np.nan:
            if peak_begin.size > 1:
                peak_begin = peak_begin[0]
                peak_end = peak_end[0]
                peak_idx = peak_idx[0]

        peak_amplitudes.append(peak_amplitude)
        peak_widths.append(peak_width)
        peak_begins.append(peak_begin)
        peak_ends.append(peak_end)
        peak_idxs.append(peak_idx)


    return peak_amplitudes, peak_widths, peak_begins, peak_ends, peak_idxs


def latency_peak_onset_offset(peak_begins, peak_ends):
    """Find latency (time in samples / ms) to peak onset (curve initiation).
    Because the peak-finding function above is defined in terms of trial-relative time (0-1500 ms),
    this is simply the index of the peak amplitude.
    Separated from above for functional clarity."""

    peak_begins = [np.int(peak_begin) if ~np.isnan(peak_begin) else peak_begin for peak_begin in peak_begins]
    peak_ends = [np.int(peak_end) if ~np.isnan(peak_end) else peak_end for peak_end in peak_ends]

    return peak_begins, peak_ends

def latency_peak_amplitude(peak_idxs):
    """Find latency (time in samples / ms) to peak amplitude of pupil diameter.
    Because the above is defined in terms of trial-relative time (0-1500 ms),
    this is simply the index of the peak amplitude.
    Separated from above for functional clarity."""

    peak_latencies = [peak_idx[0] if (isinstance(peak_idx, np.ndarray) and peak_idx.size != 0) else peak_idx for peak_idx in peak_idxs]

    return peak_latencies

def find_auc(preprocessed_sample_df, peak_begins, peak_ends):
    """Finds area under the curve of the phasic pupillary response using
    trapezoidal integration."""

    aucs = []

    for trial_epoch, peak_begin, peak_end in zip(preprocessed_sample_df.trial_epoch.unique(), peak_begins, peak_ends):


        trial_samples = preprocessed_sample_df.loc[preprocessed_sample_df.trial_epoch == trial_epoch]


        if ~np.isnan(peak_begin):
            peak_data = trial_samples.iloc[peak_begin:peak_end].z_pupil_diameter
            auc = np.trapz(peak_data)
        else:
            auc = np.nan
        aucs.append(auc)

    return aucs

def plot_trialwise_metrics(preprocessed_sample_df, aucs, peak_begins, peak_ends,
 peak_latencies, peak_widths, peak_amplitudes, trial_means,
 pupil_metric_fig_path=os.path.join(os.path.expanduser('~'), 'Dropbox/loki_0.5/figures/pupil_metric_validation_figs/')):

    for trial_epoch in preprocessed_sample_df.trial_epoch.unique():

        print('current trial ', trial_epoch)

        trial_data = preprocessed_sample_df.loc[preprocessed_sample_df.trial_epoch == trial_epoch]

        auc = aucs[trial_epoch]
        peak_begin = peak_begins[trial_epoch]
        peak_end = peak_ends[trial_epoch]
        peak_latency = peak_latencies[trial_epoch]
        peak_amplitude = peak_amplitudes[trial_epoch]
        peak_width = peak_widths[trial_epoch]
        trial_mean = trial_means.z_pupil_diameter[trial_epoch]

        plt.ioff()
        fig=plt.figure()
        fig_name = (str(trial_data.subj_id.unique()[0]) +
        '_' + str(trial_data.condition.unique()[0]) + '_' + 't' + str(trial_epoch))
        plt.title(fig_name)
        plt.plot(trial_data.trial_sample_shift, trial_data.z_pupil_diameter)
        plt.plot(peak_latency, peak_amplitude, 'ro')

        plt.vlines([peak_begin, peak_end], color='red',
        ymin=trial_data.z_pupil_diameter.min(),
        ymax=trial_data.z_pupil_diameter.max())

        plt.vlines(peak_latency, color='green',
        ymin=trial_data.z_pupil_diameter.min(),
        ymax=trial_data.z_pupil_diameter.max())

        plt.text(0,0, np.round(peak_width,2), fontsize=16)
        plt.text(2000,0, np.round(auc,2), fontsize=16)
        plt.savefig(os.path.join(pupil_metric_fig_path, fig_name + '.png'))
        plt.close()


    return None

def feature_lists_to_df(peak_amplitudes, peak_widths, trial_means,
 peak_begins, peak_ends,peak_latencies, aucs, subj_id, reward_code,
 feature_path=os.path.join(os.path.expanduser('~'), 'Dropbox/loki_0.5/analysis/pupil/processed_data/pupil_features/')):

    feature_filename = (str(subj_id) + '_' + str(reward_code) + '_feature_df.csv')

    feature_dict = {'peak_amp': peak_amplitudes, 'peak_widths': peak_widths,
    'pupil_mean': trial_means.z_pupil_diameter, 'peak_onset': peak_begins, 'peak_offset': peak_ends,
    'peak_amp_latency': peak_latencies, 'auc': aucs}

    feature_df = pd.DataFrame(feature_dict)
    feature_df['subj_id'] = subj_id
    feature_df['reward_code'] = reward_code
    feature_df['trial'] = np.arange(len(feature_df))

    feature_df.to_csv(os.path.join(feature_path, feature_filename), index=False)

    return feature_df

def concat_features(feature_dfs, subj_id, feature_path=os.path.join(os.path.expanduser('~'),
'Dropbox/loki_0.5/analysis/pupil/processed_data/pupil_features/'), n_trials=398, n_sessions=9):

    grand_feature_filename = (str(subj_id) + '_grand_feature_df.csv')

    grand_feature_df = pd.concat(feature_dfs)

    assert len(grand_feature_df) == (n_trials*n_sessions), 'check grand_feature_df len'

    grand_feature_df.to_csv(os.path.join(feature_path, grand_feature_filename), index=False)

    return grand_feature_df

from pupil_parse.preprocess_utils import config as cf

import matplotlib as mpl
mpl.use('macosx')

import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np

(raw_data_path, intermediate_data_path,
processed_data_path, figure_path, simulated_data_path) = cf.path_config()

cf.plot_config()


def visualize(x,y, subj_id, session_n, reward_code,
stimulus_onset=500, trial_end=2000, interval_end=4000,
 id_str=None, estimator=np.mean):
    """ Visualize the trial-averaged task-evoked pupillary response. """

    fig_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code))

    if id_str:
        fig_name = fig_name + '_' + id_str

    fig = plt.figure()
    sns.lineplot(x, y,
    estimator=estimator)
    plt.axvline(x=stimulus_onset, linestyle='dashed', color='k')
    plt.axvline(x=trial_end, linestyle='dashed', color='k')
    plt.xlabel('time from stimulus onset (ms)'); plt.ylabel('pupil diameter (a.u.)')
    plt.title(fig_name)
    plt.xticks(np.arange(0,
     interval_end+stimulus_onset, stimulus_onset), np.arange(-1*stimulus_onset,
    interval_end, stimulus_onset))
    plt.xlim([0, trial_end])

    # plt.ylim([3000, 10000])

    return fig, fig_name


def indicate_blinks(samples, events, subj_id, session_n, reward_code):
    """ Flag the blink intervals within the samples dataframe. """

    blinks = events.loc[events.blink==1]
    blink_start = blinks.relative_start_time
    blink_end = blinks.relative_end_time

    blink_intervals = list(map(np.arange, blink_start, blink_end+1))
    blink_intervals_flat = [item for sublist in blink_intervals for item in sublist]

    #because the sample data are segmented, the number of blink intervals detected
    #will likely be smaller than the original set of blink intervals

    samples['blink'] = samples.relative_time.isin(blink_intervals_flat)


    assert samples.blink.sum() <= len(blink_intervals_flat), 'blinks detected in segmented data > blinks detected in all data. check.'


    return samples

def raster_plot(samples, subj_id, session_n, reward_code, n_trial_samples, id_str=None,
    stimulus_onset=500, trial_end=2000, interval_end=4000):
    fig_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code) + '_random_raster')

    if id_str:
        fig_name = fig_name + '_' + id_str

    n_trials = samples.trial_epoch.nunique()
    sampled_trials = np.random.choice(np.arange(n_trials), n_trial_samples)

    plt.figure(1)
    fig, ax = plt.subplots(n_trial_samples,
    sharex=True, sharey=True)
    fig.suptitle(fig_name)
    [ax[sample_n].plot(samples.trial_sample.unique(),
    samples.loc[samples.trial_epoch == trial, 'pupil_diameter']) for trial, sample_n in zip(sampled_trials, range(len(sampled_trials)))]
    [ax[sample_n].vlines(samples.loc[(samples.trial_epoch == trial) & (samples.blink == 1), 'trial_sample'], ymax=max(samples.pupil_diameter), ymin=min(samples.pupil_diameter)) for trial, sample_n in zip(sampled_trials, range(len(sampled_trials)))]
    [ax[sample_n].axvline(x=stimulus_onset, linestyle='solid', color='gray', alpha=0.6) for trial, sample_n in zip(sampled_trials, range(len(sampled_trials)))]
    [ax[sample_n].axvline(x=trial_end, linestyle='solid', color='gray', alpha=0.6) for trial, sample_n in zip(sampled_trials, range(len(sampled_trials)))]
    plt.xticks(np.arange(0,
     interval_end+stimulus_onset, stimulus_onset), np.arange(-1*stimulus_onset,
    interval_end, stimulus_onset))
    [ax[sample_n].set_yticks([]) for sample_n in range(len(sampled_trials))]
    [ax[sample_n].set_ylabel(str(sample_n), rotation=0) for sample_n in range(len(sampled_trials))]
    plt.xlabel('trial sample', fontsize=16)
    fig.text(0.095, 0.5, 'trial', va='center', rotation='vertical', fontsize=16)

    return fig, fig_name


def save(fig, fig_name, figure_path=figure_path):
    """ Save the trial-averaged task-evoked pupillary response plot. """

    print('figure saving ...')
    plt.savefig(os.path.join(figure_path, fig_name + '.png'), bbox_inches='tight')
    print('figure saved')

    return None

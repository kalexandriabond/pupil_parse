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

def rt_order_df(samples_df, messages_df):

    rt_ordered_msg_df = messages_df.sort_values(by='trial_response_time', ascending=True).reset_index(drop=True)

    rt_ordered_trials = rt_ordered_msg_df.trial.values

    rt_ordered_samples_df_temp = pd.DataFrame()

    rt_ordered_samples_df_temp = [rt_ordered_samples_df_temp.append(samples_df.loc[samples_df.trial_epoch == rt_ordered_trial]) for trial_epoch, rt_ordered_trial in zip(samples_df.trial_epoch.unique(), rt_ordered_trials)]

    rt_ordered_samples_df = pd.concat(rt_ordered_samples_df_temp).reset_index(drop=True)

    return rt_ordered_samples_df, rt_ordered_msg_df

def random_order_df(samples_df, messages_df, random_state=102819):

    n_trials = len(messages_df)

    random_ordered_msg_df = messages_df.sample(n_trials, replace=False,
                                              random_state=random_state).reset_index(drop=True)

    random_ordered_trials = random_ordered_msg_df.trial.values

    random_ordered_samples_df_temp = pd.DataFrame()

    random_ordered_samples_df_temp = [random_ordered_samples_df_temp.append(samples_df.loc[samples_df.trial_epoch == random_ordered_trial]) for trial_epoch, random_ordered_trial in zip(samples_df.trial_epoch.unique(), random_ordered_trials)]

    random_ordered_samples_df = pd.concat(random_ordered_samples_df_temp).reset_index(drop=True)


    assert np.equal(random_ordered_samples_df.trial_epoch.unique(), random_ordered_trials).sum() == len(random_ordered_trials), 'check trial reordering'

    return random_ordered_samples_df,random_ordered_msg_df

def plot_evoked_response_map(ordered_samples_df, ordered_message_df, fig_name,
                             fig_path, trial_end_sample_idx=1500):

    """call signature:
    _, samples_pivot = plot_evoked_response_map(sample_df, message_df, fig_name='trial_ordered_evoked_responses')
    _= plot_evoked_response_map(rt_ordered_samples_df, rt_ordered_msg_df, fig_name='RT_ordered_evoked_responses')
    _= plot_evoked_response_map(random_ordered_samples_df, random_ordered_msg_df, fig_name='random_ordered_evoked_responses')
    """

    n_trials = len(ordered_samples_df.trial_epoch.unique())

    jtplot.style('grade3', context='poster', fscale=1.4, spines=False, gridlines='--')

    ordered_samples_df = ordered_samples_df.loc[ordered_samples_df.trial_sample < trial_end_sample_idx]

    samples_sparse = ordered_samples_df[['trial_sample', 'trial_epoch',
    'z_pupil_diameter']]

    samples_sparse['reset_trial_epoch_idx'] = np.repeat(np.arange(0,n_trials),
                                                    trial_end_sample_idx)
    # hack to get pivot to respect the stated order of the trial epochs
    # otherwise, will sort the index ...

    samples_pivot = samples_sparse.pivot(index='reset_trial_epoch_idx',
    columns='trial_sample', values='z_pupil_diameter')

    plt.ioff()
    plt.figure(1)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(samples_pivot, fmt="g", cmap='viridis',
    cbar_kws={'label': 'pupil diameter'}, robust=True, vmin=0, vmax=2)
    ax.scatter(x=ordered_message_df.trial_response_time,y=range(n_trials), marker='.', color='white', s=30)
    plt.title(fig_name)
    plt.ylabel('trial')
    plt.savefig(os.path.join(fig_path, fig_name + '.png'))
    plt.close()

    return fig_name, samples_pivot

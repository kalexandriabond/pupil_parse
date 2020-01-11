from pupil_parse.preprocess_utils import config as cf

from scipy.signal import find_peaks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cf.plot_config()

def deblink(samples, events):
    """Deblink data."""

    blink_starts = events.loc[(events.blink == 1), 'relative_start_time'].reset_index(drop=True)
    blink_ends = events.loc[(events.blink == 1), 'relative_end_time'].reset_index(drop=True)

    for blink_start, blink_end in zip(blink_starts, blink_ends):
        samples.loc[(samples.relative_time >= blink_start) &
        (samples.relative_time < blink_end), 'pupil_diameter'] = np.nan

    samples.loc[samples.pupil_diameter <= 0, 'pupil_diameter'] = np.nan

    prop_invalid_data = np.round(samples.pupil_diameter.isna().sum() / len(samples), 2)

    print('proportion of data invalid due to blinks: ', prop_invalid_data)

    return samples, prop_invalid_data


def outlier_removal(samples, sd=3, plot=False):
    """Calculate the min and max pupil size as defined by 3 sd criterion.
    Replace outliers with nans."""

    min_pupil = samples.pupil_diameter.mean() - sd*samples.pupil_diameter.std()
    max_pupil = samples.pupil_diameter.mean() + sd*samples.pupil_diameter.std()

    samples.loc[samples.pupil_diameter <= min_pupil, 'pupil_diameter']  = np.nan
    samples.loc[samples.pupil_diameter >= max_pupil, 'pupil_diameter']  = np.nan

    if plot:
        plt.figure()
        sns.scatterplot(x=samples.relative_time, y=samples.pupil_diameter,
         data=samples, estimator=None, s=1)

    prop_invalid_data = np.round(samples.pupil_diameter.isna().sum() / len(samples), 2)

    print('proportion of data rendered outlying: ', prop_invalid_data)


    return samples, prop_invalid_data


def interpolate(samples, method='linear', plot=False):
    """Interpolate the samples."""

    samples['pupil_diameter_interp'] = samples.pupil_diameter.interpolate(method=method)

    if plot:
        f, axes = plt.subplots(1, 2)

        sns.scatterplot(x=samples.relative_time[:10000], y=samples.pupil_diameter[:10000],
         data=samples, estimator=None, ax=axes[0], s=1)
        sns.scatterplot(x=samples.relative_time[:10000], y=samples.pupil_diameter_interp[:10000],
        data=samples, estimator=None, ax=axes[1], s=1)

        plt.tight_layout()

    samples['pupil_diameter'] = samples.pupil_diameter_interp
    samples.drop(columns='pupil_diameter_interp', inplace=True)

    prop_interp_data = 1 - (samples.pupil_diameter.isna().sum() / len(samples))
    n_failed_interp_data = samples.pupil_diameter.isna().sum()

    print('proportion of data successfully interpolated: ', prop_interp_data)
    print('n datapoints left uninterpolated: ', n_failed_interp_data)


    return samples, prop_interp_data, n_failed_interp_data

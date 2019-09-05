from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import bandpass_filter as bp

from scipy.signal import find_peaks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cf.plot_config()

def calc_pupillary_extrema(samples, sampling_rate=1000,
use_lowpass_data=0):
    """Find peaks and valleys of the pupillary response."""
    # rationale: could use the luminance manipulation to detect peak amplitudes
    # but each person may have a varying latency in their pupillary response.
    # opted for an empirical approach instead, finding peak latencies at an interval approximating the switch in luminance peaks
    # that also accomodates variability in pupillary response.

    # 15 second-long intervals between peaks, lenient threshold for distance between amplitudes
    min_peak_dist = sampling_rate * 15
    min_width = sampling_rate

    if use_lowpass_data:
        sample_temp = samples.lowpass_pupil_diameter
    else:
        sample_temp = samples.pupil_diameter

    peaks, _ = find_peaks(sample_temp, distance=min_peak_dist,
     width=min_width)

    valleys, _ = find_peaks(sample_temp*-1, distance=min_peak_dist,
    width=min_width)

    samples['peak_samples'] = np.nan
    samples['peak_samples'].iloc[peaks] = sample_temp.iloc[peaks]

    samples['valley_samples'] = np.nan
    samples['valley_samples'].iloc[valleys] = sample_temp.iloc[valleys]

    print(samples.valley_samples.unique(), samples.peak_samples.unique())

    median_peak = samples.peak_samples.median()
    median_valley = samples.valley_samples.median()

    assert median_peak > median_valley, 'check peak detection'

    return samples, median_peak, median_valley

def plot_extrema(samples):
    """ Check pupillary extrema. """

    plt.figure()
    plt.scatter('relative_time', "pupil_diameter_interp", data=samples, s=.3)
    plt.scatter('relative_time', "valley_samples", s=500, marker='x',
    color='red', data=samples)
    plt.scatter('relative_time', "peak_samples", s=500, marker='x',
    color='red', data=samples)
    plt.ylabel('Pupil diameter (a.u.)')
    plt.xlabel('Time (s)')

    return None

def flag_dynamic_range_boundary(samples,
 median_peak, median_valley):
    """ If the pupil diameter exceeds the median bounds of the pupillary response as defined by
    the luminance response for that session, flag. """

    samples['physio_range'] = np.nan

    samples.loc[samples.pupil_diameter <= median_valley, 'physio_range'] = 0
    samples.loc[samples.pupil_diameter >= median_peak, 'physio_range'] = 1

    prop_out_bound_data = np.round(((samples.physio_range == 0).sum() +
    (samples.physio_range == 1).sum()) / len(samples), 2)

    print('proportion of data out of phys. range: ', prop_out_bound_data)

    return samples

from pupil_parse.preprocess_utils import config as cf

import numpy as np
from scipy import stats

def zscore(samples):
    """ z-score lowpass filtered data within session. """

    samples['z_pupil_diameter'] = stats.zscore(samples.lowpass_pupil_diameter)

    return samples

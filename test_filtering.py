from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md
from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import deblink_delimit as dd

import time

# from pupil_parse.preprocess_utils import bandpass_filter as bp

from scipy.signal import find_peaks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


subj_id, session_n = 789, 3

(raw_data_path, intermediate_data_path,
processed_data_path, figure_path) = cf.path_config()


subj_data_file, subj_data_file_raw, reward_code = ep.find_data_files(subj_id=subj_id,
session_n=session_n, reward_task=1, lum_task=0,
raw_data_path=raw_data_path)


reward_samples = ep.read_hdf5('samples', subj_id, session_n,
intermediate_data_path, reward_code=reward_code)
print('samples')
reward_messages = ep.read_hdf5('messages', subj_id, session_n,
intermediate_data_path, reward_code=reward_code)
print('messages')
reward_events = ep.read_hdf5('events', subj_id, session_n,
intermediate_data_path, reward_code=reward_code)
print('events')


lum_samples = ep.read_hdf5('samples', subj_id, session_n,
intermediate_data_path, reward_code=None)
print('lum samples')
lum_messages = ep.read_hdf5('messages', subj_id, session_n,
intermediate_data_path, reward_code=None)
print('lum messages')
lum_events = ep.read_hdf5('events', subj_id, session_n,
intermediate_data_path, reward_code=None)
print('lum events')


lum_samples=dd.deblink(lum_samples,lum_events)
lum_samples=dd.outlier_removal(lum_samples)
lum_samples=dd.interpolate(lum_samples, method='cubic')


reward_samples=dd.deblink(reward_samples,reward_events)
reward_samples=dd.outlier_removal(reward_samples)
reward_samples=dd.interpolate(reward_samples, method='cubic')

## TODO: bandpass filtering is resulting in nans... figure out parameters that work for everyone
# lum_samples = dd.high_bandpass_filter(lum_samples)
# lum_samples = dd.low_bandpass_filter(lum_samples)

# _ = dd.check_high_low_bandpass_data(lum_samples)

# may need to do the bandpass filtering first bc extrema are spurious
median_peak, median_valley=dd.calc_pupillary_extrema(lum_samples, use_lowpass_data=0)

reward_samples=dd.flag_dynamic_range_boundary(reward_samples, median_peak, median_valley)

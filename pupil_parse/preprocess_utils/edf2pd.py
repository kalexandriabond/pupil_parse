import pandas as pd
from pyedfread import edf
from glob import glob
import re
import os
import numpy as np

def find_data_files(subj_id, raw_data_path, session_n,
reward_task=0, lum_task=0, n_runs=None):
    """Find the data file for the session."""

    if reward_task:
        task_str = '[0-9]'*4
    if lum_task:
        task_str = 'lum'
    elif not reward_task or lum_task:
        raise ValueError('The task must be specified as either\
        the reinforcement learning task or the luminance range task.\
        See the arguments for this function.')

    subj_id = '{:03d}'.format(subj_id)
    session_n = '{:01d}'.format(session_n)

    subj_data_file_list = glob(raw_data_path + 'sub-' +
    str(subj_id) + '/' + 'ses-' + str(session_n) + '/' + 'pupil' + '/' +
    'sub-' + str(subj_id) + '_' + 'ses-' + str(session_n) +
    '_task-' + task_str + '*.EDF')

    print('subj data file list ', subj_data_file_list)
    print('len subj data file list ', len(subj_data_file_list))

    if n_runs:
        subj_data_file_raw = [''.join(subj_file) for subj_file in subj_data_file_list] # convert list to raw str
        subj_data_file = [os.path.basename(subj_file) for subj_file in subj_data_file_raw]

        if len(subj_data_file_list) != n_runs:
            print('number of runs is aberrant. n_runs = ', len(subj_data_file_list))
    else:
        subj_data_file_raw = ''.join(subj_data_file_list)
        subj_data_file = os.path.basename(subj_data_file_raw)

    assert subj_data_file, 'Subject data file not found. Check input.'

    if reward_task:

        reward_code = np.unique([re.search('task-\d{4}', subj_file).group(0)[-4:] for subj_file in subj_data_file_list])

        assert reward_code, 'Reward code not found. Check input.'

    else:
        reward_code = None

    return subj_data_file, subj_data_file_raw, reward_code


def read_edf(subj_data_file_raw):
    """Convert the EDF to pd.DataFrames containing pupil samples, events, and messages."""

    samples, events, messages = edf.pread(subj_data_file_raw)

    assert not samples.empty, 'Error in EDF to pd.DataFrame conversion. Check input.'

    return samples, events, messages


def clean_df(samples, events, messages, lum_task=0, reward_task=0):
    """Clean the pd.DataFrames."""

    samples = samples[['time', 'gx_left', 'gy_left', 'pa_left',
    'gxvel_left', 'gyvel_left']]
    events = events[['trial','blink', 'eye', 'end', 'start']]

    if lum_task:
        messages = messages[['TRIAL_RESULT_time',
        'stim_onset_time', 'stim_offset_time', 'trialid_time']]
    if reward_task:
        messages = messages[['TRIAL_RESULT_time','iti_begin_time',
        'iti_end_time', 'response_time', 'stim_onset_time',
        'stim_offset_time', 'trialid_time']]

    samples=samples.rename(columns={"time": "raw_time", "pa_left": "pupil_diameter",
    "gx_left": "gaze_x", "gy_left": "gaze_y",
     "gxvel_left": "gaze_vel_x", "gyvel_left": "gaze_vel_y"},
    errors='raise')

    print(samples.gaze_x, samples.gaze_y)

    events=events.rename(columns={"end": "raw_end_time", "start": "raw_start_time"},
     errors='raise')

    messages=messages.rename(columns={"TRIAL_RESULT_time": 'trial_result_time'},
    errors='raise')
    messages = messages.add_prefix('raw_')

    return samples, events, messages

def extract_experimental_data(samples, events, messages):
    """Extract the experimental data from the data stream."""

    exp_begin_time, exp_end_time = (messages.raw_stim_onset_time.min(),
    messages.raw_stim_offset_time.max())

    assert exp_begin_time < exp_end_time, 'exp. begin time >= exp. end time'

    trial_begin, trial_n = 1, events.trial.max()

    samples = samples.loc[(samples.raw_time >= exp_begin_time) &
     (samples.raw_time <= exp_end_time)].reset_index(drop=True)

    events = events.loc[(events.trial >= trial_begin) &
      (events.trial <= trial_n)].reset_index(drop=True)

    messages = messages.loc[(messages.raw_stim_onset_time >= exp_begin_time) &
    (messages.raw_stim_offset_time <= exp_end_time)].reset_index(drop=True)

    return samples, events, messages


def define_relative_time(samples, events, messages):
    """Recast raw time as experiment-relative time."""

    abs_start_time = samples.raw_time.min()
    samples['relative_time'] = samples.raw_time - abs_start_time
    events[['relative_end_time', 'relative_start_time']] = (events[['raw_end_time', 'raw_start_time']]
    - abs_start_time)

    messages_relative = messages.copy()
    messages_relative = messages_relative - abs_start_time
    messages_relative.columns = messages.columns.str.replace("raw", "relative")
    messages = pd.concat([messages_relative, messages], axis=1)

    return samples, events, messages
    # note: trialid occurs before the onset of the stimulus


def save_hdf5(samples, events, messages,
subj_id, session_n, intermediate_data_path,
reward_code=None, id_str=None, run_n=None):
    """Save the samples, events, and messages pd.DataFrames within an HDF5 file
    for quick storage and loading."""

    session_n = '{:02d}'.format(session_n)
    reward_code = reward_code[0]
    run_n = '{0}'.format(run_n)

    subj_id = '{:04d}'.format(subj_id)


    if reward_code:
        task_str = reward_code
    else:
        task_str = 'lum'

    if run_n:
        file_id = os.path.join(intermediate_data_path + 'sub-' +
        str(subj_id) + '_ses-' + str(session_n) + '_task-' +
        str(task_str) + '_run-' + str(run_n))
    else:
        file_id = os.path.join(intermediate_data_path + 'sub-' +
        str(subj_id) + '_ses-' + str(session_n) + '_task-' +
        str(task_str))

    if id_str:
        file_id = os.path.join(file_id + '_' + id_str)

    hdf = os.path.join(file_id + '_pupil.h5')

    samples.to_hdf(hdf, key='samples')
    events.to_hdf(hdf, key='events')
    messages.to_hdf(hdf, key='messages')


    return hdf


def read_hdf5(data_key, subj_id, session_n,
intermediate_data_path, reward_code=None, id_str=None, run_n=None):
    """Test hdf5 conversion and read hdf5 files."""


    session_n = '{:01d}'.format(session_n)
    reward_code = reward_code[0]

    subj_id = '{:03d}'.format(subj_id)

    if reward_code:
        task_str = reward_code
    else:
        task_str = 'lum'

    if run_n:
        run_n = '{0}'.format(run_n)
        file_id = os.path.join(intermediate_data_path + 'sub-' +
        str(subj_id) + '_ses-' + str(session_n) + '_task-' +
        str(task_str) + '_run-' + str(run_n))
    else:
        file_id = os.path.join(intermediate_data_path + 'sub-' +
        str(subj_id) + '_ses-' + str(session_n) + '_task-' +
        str(task_str))

    if id_str:
        file_id = os.path.join(file_id + '_' + id_str)

    hdf = os.path.join(file_id + '_pupil.h5')

    data = pd.read_hdf(hdf, data_key)

    print(data.head())

    return data

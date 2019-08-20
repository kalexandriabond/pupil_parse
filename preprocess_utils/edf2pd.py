from ../../pupil_parse import raw_data_path, intermediate_data_path


import pandas as pd
# from pyedfread import edf # note that this package will not work on linux. need macosx.
import glob
import re


def find_data_files(subj_id, session_n,
reward_task=1, lum_task=0):
    """Find the data file for the session."""

    if reward_task:
        task_str = '[0-9]'*4
    if lum_task:
        task_str = 'lum'
    elif not reward_task or lum_task:
        raise ValueError('The task must be specified as either\
        the reinforcement learning task or the luminance range task.\
        See the arguments for this function.')

    subj_data_file_list = glob.glob(raw_data_path + 'sub-' +
    str(subj_id) + '/' + 'ses-' + str(session_n) + '/' + 'pupil' + '/' +
    'sub-' + str(subj_id) + '_' + 'ses-' + str(session_n) +
    '_task-' + task_str + '.EDF')

    subj_data_file = ''.join(subj_data_file_list) # convert list to raw str

    assert subj_data_file, 'Subject data file not found. Check input.'

    if reward_task:

        reward_code = re.search('\d{4}', subj_data_file).group(0)

        assert reward_code, 'Reward code not found. Check input.'

    else:
        reward_code = None

    return subj_data_file, reward_code


def read_edf(subj_data_file):
    """Convert the EDF to pd.DataFrames containing pupil samples, events, and messages."""

    samples, events, messages = edf.pread(subj_data_file)

    assert samples, 'Error in EDF to pd.DataFrame conversion. Check input.'

    return samples, events, messages


def clean_df(samples, events, messages):
    """Clean the pd.DataFrames."""

    samples = samples[['time', 'px_left', 'py_left', 'pa_left']]
    events = events[['trial','blink', 'eye', 'end', 'start']]
    messages = messages[['TRIAL_RESULT_time',
    'instruction_phase_offset_time', 'instruction_phase_onset_time',
    'iti_begin_time', 'iti_end_time', 'response_time', 'stim_onset_time',
    'stim_offset_time', 'trialid_time']]


    samples.rename(columns={"time": "raw_time", "pa_left": "pupil_diameter",
    "px_left": "pupil_x", "py_left": "pupil_y"},
    inplace=True)

    events.rename(columns={"end": "raw_end_time", "start": "raw_start_time"},
    inplace=True)

    messages.rename("TRIAL_RESULT_time", 'trial_result_time')
    messages = messages.add_prefix('raw_')

    return samples, events, messages

def extract_experimental_data(samples, events, messages):
    """Extract the experimental data from the data stream."""

    exp_begin_time, exp_end_time = (messages.stim_onset_time.min(),
    messages.stim_offset_time.max())

    trial_begin, trial_n = 1, events.trial.max()

    samples = samples.loc[(samples.raw_time >= exp_begin_time) &
     (samples.raw_time <= exp_end_time)].reset_index()

    events = events.loc[(events.trial >= trial_begin) &
      (events.trial <= trial_n)].reset_index()

    messages = messages.loc[(messages.stim_onset_time >= exp_begin_time) &
    (messages.stim_offset_time <= exp_end_time)].reset_index()

    return samples, events, messages


def define_relative_time(samples, events, messages):
    """Recast raw time as experiment-relative time."""

    samples['relative_time'] = samples.raw_time - samples.raw_time.min()
    events[['relative_end_time', 'relative_start_time']] = (events[['raw_end_time', 'raw_start_time']]
    - samples.raw_time.min())
    messages = messages - samples.raw_time.min()

    return samples, events, messages ## TODO:  test this


def save_hdf5(samples, events, messages,
subj_id, session_n, reward_code=None, reward_task=1, lum_task=0):
    """Save the samples, events, and messages pd.DataFrames within an HDF5 file for quick storage and loading."""

    if reward_task:
        task_str = reward_code
    if lum_task:
        task_str = 'lum'

    file_id = (intermediate_data_path + 'sub-' +
    str(subj_id) + '_ses-' + str(session_n) + '_task-' +
    task_str)

    hdf = pd.HDFStore(file_id + '_pupil.h5')

    hdf.put(file_id + '_samples', samples, format='table',
    data_columns=True)
    hdf.put(file_id + '_events', events, format='table',
    data_columns=True)
    hdf.put(file_id + '_messages', messages, format='table',
    data_columns=True)

    hdf.close()

    return hdf



# >>> hdf = read_hdf('storage.h5','d1',where=['A>.5'], columns=['A','B']) # how to read hd5 and extract df

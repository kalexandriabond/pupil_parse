import os
import re
from glob import glob
import numpy as np
from collections import Counter

from pupil_parse.preprocess_utils import config as cf

(raw_data_path, intermediate_data_path,
_, _, _) = cf.path_config()

def extract_subjects_sessions(raw_data_path, reward_task=0, lum_task=0,
session_max=9, session_min=1, n_subjects=4):
    """Find the metadata for each session."""

    if reward_task:
        task_str = '[0-9]'*4
    if lum_task:
        task_str = 'lum'
        session_max = session_max + 1
        session_min = session_min - 1
    elif not reward_task or lum_task:
        raise ValueError('The task must be specified as either\
        the reinforcement learning task or the luminance range task.\
        See the arguments for this function.')

    subj_data_file_list = glob(raw_data_path + 'sub-*' +
     '/' + 'ses-*' + '/' + 'pupil' + '/' +
    'sub-*' + '_' + 'ses-*' +
    '_task-' + task_str + '.EDF')

    subj_data_files = [os.path.basename(file) for file in subj_data_file_list]

    assert subj_data_files, 'Subject data file not found. Check input.'

    subjects = [int(re.search('sub-\d{3}', file).group(0)[-3:]) for file in subj_data_files]
    sessions = [int(re.search('ses-\d{1}', file).group(0)[-1]) for file in subj_data_files]

    print('number of sessions per subject: ', Counter(subjects),
    '\nnumber of session instances: ', Counter(sessions))

    unique_subjects = np.unique(subjects)
    unique_sessions = np.unique(sessions)

    assert len(unique_subjects) == n_subjects, 'check number of subjects'
    assert len(unique_sessions) == session_max, 'check number of sessions'
    assert all(session_min <= session <= session_max for session in sessions), 'check session numbers'

    if reward_task:
        reward_codes = [re.search('\d{4}', file).group(0) for file in subj_data_files]
        assert reward_codes, 'Reward code not found. Check input.'
        unique_reward_codes = np.unique(reward_codes)
        print('number of reward condition instances: ', Counter(reward_codes))
    else:
        unique_reward_codes = None

    return unique_subjects, unique_sessions, unique_reward_codes

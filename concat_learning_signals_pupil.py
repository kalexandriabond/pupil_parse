import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import re

from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md


def main():


    (raw_data_path, _, processed_data_path, figure_path, simulated_data_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
    reward_task=1)

    start_time = time.time()

    for subject in unique_subjects:
        for reward_code, session in zip(unique_reward_codes, unique_sessions):

            processed_fn = ('tepr' +  '_sub-' + str(subject) + '_sess-' +
             str(session) +  '_cond-' + str(reward_code) + '_trial.csv')

            learning_signals_fn = 'sub-{}_cond-{}_learning_signals.csv'.format(subject, reward_code)
            pupil_amplitude_base = 'tepr_sub-{}_sess-*_cond-{}_trial_peaks.csv'.format(subject, reward_code) #cant handle wildcard
            pupil_summary_base = 'tepr_sub-{}_sess-*_cond-{}_trial_means.csv'.format(subject, reward_code)

            pupil_amplitude_fn = glob.glob(os.path.join(processed_data_path, pupil_amplitude_base))
            pupil_summary_fn = glob.glob(os.path.join(processed_data_path, pupil_summary_base))

            if not pupil_amplitude_fn:
                print('No data for this session.')
                continue

            learning_signals_df = pd.read_csv(os.path.join(simulated_data_path, learning_signals_fn))
            pupil_amplitude_df = pd.read_csv(pupil_amplitude_fn[0])
            pupil_summary_df = pd.read_csv(pupil_summary_fn[0])

            try:
                trial_df = pd.concat([learning_signals_df, pupil_amplitude_df, pupil_summary_df], axis=1)
                trial_df = trial_df.loc[:,~trial_df.columns.duplicated()]
                trial_df.drop(columns=['trial_epoch'], inplace=True)
                trial_df.to_csv(os.path.join(processed_data_path, processed_fn), index=False)
            except:
                print('error in concat.')
                pass

    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

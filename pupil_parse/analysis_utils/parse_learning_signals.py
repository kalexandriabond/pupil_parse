import time
import numpy as np
import matplotlib.pyplot as plt
import os

from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md


def main():

    (_, _, _, figure_path, simulated_data_path) = cf.path_config()
    learning_signals_fn = 'learning_signals.csv'

    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
    reward_task=1)

    start_time = time.time()

    learning_signals_df = pd.read_csv(os.path.join(simulated_data_path, learning_signals_fn))

    print(learning_signals_df.head())
    print(learning_signals_df.subj_id.unique() + '\n', learning_signals_df.reward_code.unique())



    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

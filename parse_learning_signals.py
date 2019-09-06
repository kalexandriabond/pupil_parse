import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob, re

from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md


def main():


    (raw_data_path, _, _, figure_path, simulated_data_path) = cf.path_config()
    learning_signals_fn = 'learning_signals.csv'
    base_fn = 'sub-*_cond-*_learning_signals.csv'

    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
    reward_task=1)

    start_time = time.time()

    learning_signals_df = pd.read_csv(os.path.join(simulated_data_path, learning_signals_fn))

    n_trials = 400
    n_subjects = learning_signals_df.subj_id.nunique()
    expected_len = n_trials-2

    unique_reward_codes = np.repeat(learning_signals_df.reward_code.sort_values(ascending=True).unique(),
     n_subjects)
    unique_reward_values = np.repeat([6510, 6520, 6530, 7510, 7520, 7530, 8510, 8520, 8530], n_subjects)

    def split(df, group):
         gb = df.groupby(group)
         return [gb.get_group(x) for x in gb.groups]

    def remove_first_last_trials(df_list, expected_len=expected_len):

        df_lens = []

        sliced_dfs = [df.iloc[1:-1] for df in df_list]
        [df_lens.append(len(df)) for df in sliced_dfs] # check lengths are expected

        assert np.unique(df_lens) == expected_len, 'check length of sliced dfs!'

        return sliced_dfs

    def extract_fns(df_list):

        fns = [('sub-' + str(df.subj_id.unique()[0])
        +  '_cond-' + str(df.reward_code.unique()[0]) +
        '_learning_signals.csv') for df in df_list]

        return fns

    def save_split_dfs(df_list, fn_list,
    data_path=simulated_data_path):

        [df.to_csv(os.path.join(data_path, fn), index=False) for df, fn in zip(df_list, fn_list)]

        return print('dfs saved')

    def decode_condition(base_fn=base_fn, data_path=simulated_data_path,
    reward_values=unique_reward_values, reward_codes=unique_reward_codes):

        def get_cond(elem):
            return elem[-22:-21] # get the condition code for each file

        learning_signals_fns = glob.glob(os.path.join(data_path, base_fn))
        learning_signals_fns.sort(key=get_cond)

        [print(reward_code, reward_val, fn)
        for fn, reward_val, reward_code in zip(learning_signals_fns,
        reward_values, unique_reward_codes)]

        learning_signals_fns_decoded = [re.sub('cond-'+str(reward_code), 'cond-'+str(reward_val), fn)
         for fn, reward_val, reward_code in zip(learning_signals_fns,
        reward_values, unique_reward_codes)]

        [os.rename(original_fn, decoded_fn) for original_fn, decoded_fn
        in zip(learning_signals_fns, learning_signals_fns_decoded)]

        os.listdir(simulated_data_path) # check the renaming

        return None


    dfs = split(learning_signals_df, ['subj_id', 'reward_code'])
    sliced_dfs = remove_first_last_trials(dfs)
    parsed_learning_signals_fns = extract_fns(sliced_dfs)

    save_split_dfs(sliced_dfs, parsed_learning_signals_fns)
    decode_condition()

    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

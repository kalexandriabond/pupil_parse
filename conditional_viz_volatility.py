from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md

from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import visualize as vz
from pupil_parse.preprocess_utils import zscore as z


import time
import numpy as np
import matplotlib.pyplot as plt

def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()

    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
    reward_task=1)

    start_time = time.time()

    trial_end = 2000

    reward_codes = []

    for subj_id in unique_subjects:
        for session_n in unique_sessions:


            print('z-scoring baseline corrected & lowpass filtered data for subject {}'.format(subj_id) +
            ' session {}'.format(session_n))


            _, _, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)

            reward_codes.append(reward_code)

            # search for reward codes that begin with 85 eg
            # plot those samples together

            reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='corr')
            reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='corr')
            reward_events = ep.read_hdf5('events', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='corr')


            reward_samples = z.zscore(reward_samples)

            plotting_reward_samples = reward_samples.loc[reward_samples.trial_sample <= trial_end]

            ## TODO: get reasonable y limits for all data within a subject and use that for plotting

            fig,figname = vz.visualize(plotting_reward_samples.trial_sample,
            plotting_reward_samples.z_pupil_diameter,
            subj_id, session_n, reward_code, id_str='zscored')

            vz.save(fig, figname)

            hdf = ep.save_hdf5(reward_samples, reward_events, reward_messages,
            subj_id, session_n, processed_data_path,
            reward_code=reward_code, id_str='zscored')
            print('z-scored data saved')



        end_time = time.time()

        time_elapsed = end_time - start_time
        print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

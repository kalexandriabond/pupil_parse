from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md

from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import visualize as vz

from pupil_parse.analysis_utils import summarize_amplitude as amp


import time
import numpy as np
import matplotlib.pyplot as plt



def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)

    stim_offset = 2000
    stim_onset = 500

    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            start_time = time.time()

            print('extracting amplitudes for subject {}'.format(subj_id) +
            ' session {}'.format(session_n))

            _, _, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)


            reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='zscored')
            reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='zscored')
            reward_events = ep.read_hdf5('events', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='zscored')


            if (np.isnan(reward_samples.z_pupil_diameter).sum() == 0) != 1:
                print('This session has no data.')

            figures = []

            amp.calc_peaks(reward_samples)

            fig_name, fig = amp.plot_extrema(epoch_samples, subj_id,
             session_n, reward_code)

            figures.append(fig)

            amp.save_extrema(fig_name,figures)

    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

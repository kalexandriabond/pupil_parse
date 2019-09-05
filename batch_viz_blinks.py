from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md

from pupil_parse.preprocess_utils import baseline_correct as bc
from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import visualize as vz


import time

def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)

    n_trial_samples = 30

    start_time = time.time()

    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('plotting subject {}'.format(subj_id) +
            ' session {}'.format(session_n))

            _, _, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)


            reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='bandpass')
            reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='bandpass')
            reward_events = ep.read_hdf5('events', subj_id, session_n,
            processed_data_path, reward_code=reward_code, id_str='bandpass')


            # reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            # intermediate_data_path, reward_code=reward_code, id_str='seg')
            # reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            # intermediate_data_path, reward_code=reward_code, id_str='seg')
            # reward_events = ep.read_hdf5('events', subj_id, session_n,
            # intermediate_data_path, reward_code=reward_code, id_str='seg')
            #
            # if reward_samples is None:
            #     print('check the data. a lot is missing.')

            reward_samples = vz.indicate_blinks(reward_samples, reward_events,
            subj_id, session_n, reward_code)
            fig, figname = vz.raster_plot(reward_samples,
            subj_id, session_n,
             reward_code, n_trial_samples, id_str='raw')
            vz.save(fig, figname)


    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

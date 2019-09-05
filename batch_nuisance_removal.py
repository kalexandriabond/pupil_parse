from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md

from pupil_parse.preprocess_utils import deblink_delimit as dd
from pupil_parse.preprocess_utils import edf2pd as ep


import time

def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)

    start_time = time.time()

    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('processing subject {}'.format(subj_id) +
            ' session {}'.format(session_n))

            time.sleep(1)

            _, _, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)


            reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code, id_str='seg')
            reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code, id_str='seg')
            reward_events = ep.read_hdf5('events', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code, id_str='seg')

            deblinked_reward_samples = dd.deblink(reward_samples,
            reward_events)
            print('deblinking complete')
            outlier_removed_reward_samples = dd.outlier_removal(deblinked_reward_samples)
            print('outliers removed')
            interpolated_reward_samples = dd.interpolate(outlier_removed_reward_samples)
            print('data interpolated')

            hdf = ep.save_hdf5(interpolated_reward_samples, reward_events, reward_messages,
            subj_id, session_n, processed_data_path,
            reward_code=reward_code, id_str='clean')
            print('clean data saved')


    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

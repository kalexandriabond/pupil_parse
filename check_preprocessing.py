from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md
from pupil_parse.preprocess_utils import edf2pd as ep

import time

def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)


    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('checking preprocessing for subject {}'.format(subj_id) +
            ' session {}'.format(session_n))

            time.sleep(1)

            subj_data_file, subj_data_file_raw, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)


            reward_samples = ep.read_hdf5('samples', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code)
            print('samples')
            reward_messages = ep.read_hdf5('messages', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code)
            print('messages')
            reward_events = ep.read_hdf5('events', subj_id, session_n,
            intermediate_data_path, reward_code=reward_code)
            print('events')

            time.sleep(1)

            lum_samples = ep.read_hdf5('samples', subj_id, session_n,
            intermediate_data_path, reward_code=None)
            print('lum samples')
            lum_messages = ep.read_hdf5('messages', subj_id, session_n,
            intermediate_data_path, reward_code=None)
            print('lum messages')
            lum_events = ep.read_hdf5('events', subj_id, session_n,
            intermediate_data_path, reward_code=None)
            print('lum events')

            time.sleep(1)




if __name__ == '__main__':

    main()

from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md


def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    unique_subjects, unique_sessions, unique_reward_codes = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)


    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('processing subject {}'.format(subj_id) + ', session {}'.format(session_n))

            subj_data_file, subj_data_file_raw, reward_code = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, reward_task=1, lum_task=0,
            raw_data_path=raw_data_path)

            print(subj_data_file, reward_code)

            samples, events, messages = ep.read_edf(subj_data_file_raw)
            print(samples.head())

            samples, events, messages = ep.clean_df(samples, events, messages, reward_task=1)
            print(samples.head(), events.head(), messages.head())

            samples, events, messages = ep.extract_experimental_data(samples, events,
             messages)

            samples, events, messages = ep.define_relative_time(samples, events, messages)

            hdf = ep.save_hdf5(samples, events, messages, subj_id, session_n,
            intermediate_data_path, reward_code=reward_code)

if __name__ == '__main__':

    main()

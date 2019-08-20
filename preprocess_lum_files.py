from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md


def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    unique_subjects, unique_sessions, _ = md.extract_subjects_sessions(raw_data_path,
     lum_task=1)


    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('processing subject {}'.format(subj_id) + ', session {}'.format(session_n))

            subj_data_file, subj_data_file_raw, _ = ep.find_data_files(subj_id=subj_id,
            session_n=session_n, lum_task=1, raw_data_path=raw_data_path)

            samples, events, messages = ep.read_edf(subj_data_file_raw)
            print(samples.head())

            samples, events, messages = ep.clean_df(samples, events, messages, lum_task=1)
            print(samples.head(), events.head(), messages.head())

            samples, events, messages = ep.extract_experimental_data(samples, events,
             messages)
            print(events.head())

            samples, events, messages = ep.define_relative_time(samples, events, messages)

            hdf = ep.save_hdf5(samples, events, messages, subj_id, session_n,
            intermediate_data_path)

if __name__ == '__main__':

    main()

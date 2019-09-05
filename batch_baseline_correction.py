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

    start_time = time.time()

    for subj_id in unique_subjects:
        for session_n in unique_sessions:

            print('baseline correcting subject {}'.format(subj_id) +
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

            corrected_reward_samples = bc.subtractive_baseline_correction(reward_samples)

            hdf = ep.save_hdf5(corrected_reward_samples, reward_events, reward_messages,
            subj_id, session_n, processed_data_path,
            reward_code=reward_code, id_str='corr')

            lp_fig, lp_figname = vz.visualize(reward_samples.trial_sample,
            reward_samples.bc_lowpass_pupil_diameter, subj_id, session_n,
             reward_code, id_str='lowpass')
            vz.save(lp_fig, lp_figname)
            hp_fig, hp_figname = vz.visualize(reward_samples.trial_sample,
             reward_samples.bc_highpass_pupil_diameter,  subj_id, session_n,
              reward_code, id_str='highpass')
            vz.save(hp_fig, hp_figname)

    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

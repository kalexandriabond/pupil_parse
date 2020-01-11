from pupil_parse.preprocess_utils import config as cf
import numpy as np

def filter_gaze(samples, subj_id, session_n,
reward_code, id_str=None, display_resolution=(1280,1024),
sampling_frequency=1000):


    max_gaze_x, max_gaze_y = display_resolution
    min_gaze = 0

    sample_df_filt = samples.loc[(samples.gaze_x <= max_gaze_x) &
                                   (samples.gaze_y <= max_gaze_y) &
                                  (samples.gaze_x >= min_gaze) &
                                   (samples.gaze_y >= min_gaze)]

    data_lost_samples = ((samples.shape[0] - sample_df_filt.shape[0]))
    data_lost_minutes = data_lost_samples / sampling_frequency / 60

    prop_samples_lost = data_lost_samples / len(samples)

    print('filtering gaze data results in loss of', np.round(data_lost_minutes,2),
          'm or', data_lost_samples, 'samples of data.')

    print('proportion of data lost: ', prop_samples_lost)


    return sample_df_filt

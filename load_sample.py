from pupil_parse.preprocess_utils import config as cf
from pupil_parse.preprocess_utils import extract_session_metadata as md

from pupil_parse.preprocess_utils import edf2pd as ep
from pupil_parse.preprocess_utils import visualize as vz

from pupil_parse.analysis_utils import summarize_amplitude as amp


import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

def main():

    (raw_data_path, intermediate_data_path,
    processed_data_path, figure_path) = cf.path_config()


    (unique_subjects, unique_sessions, unique_reward_codes) = md.extract_subjects_sessions(raw_data_path,
     reward_task=1)


    # n_trial_samples = 10

    start_time = time.time()

    try:

        subj_id = np.random.choice(unique_subjects)
        session_n = np.random.choice(unique_sessions)

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
            raise ValueError('This session has no data.')

    except:

         subj_id = np.random.choice(unique_subjects)
         session_n = np.random.choice(unique_sessions)

         _, _, reward_code = ep.find_data_files(subj_id=subj_id,
         session_n=session_n, reward_task=1, lum_task=0,
         raw_data_path=raw_data_path)


         reward_samples = ep.read_hdf5('samples', subj_id, session_n,
         processed_data_path, reward_code=reward_code, id_str='zscored')
         reward_messages = ep.read_hdf5('messages', subj_id, session_n,
         processed_data_path, reward_code=reward_code, id_str='zscored')
         reward_events = ep.read_hdf5('events', subj_id, session_n,
         processed_data_path, reward_code=reward_code, id_str='zscored')




    stim_offset = 2000
    stim_onset = 500

    figures = []

    for trial_epoch in reward_samples.trial_epoch.unique():

        epoch_samples = reward_samples.loc[(reward_samples.trial_epoch == trial_epoch) &
        (reward_samples.trial_sample <= stim_offset) &
        (reward_samples.trial_sample > stim_onset)].reset_index()

        amp.find_peak(epoch_samples)
        fig_name, fig = amp.plot_extrema(epoch_samples, subj_id,
         session_n, reward_code, id_str=str(trial_epoch))

        figures.append(fig)
        # amp.save_extrema(fig_name,fig)
        # replace this with below as a fn



    fig_name = ('tepr' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code) + '_trials')

    pdf = PdfPages(os.path.join(figure_path, fig_name + '.pdf'))
    print(figures)

    for fig in figures:
        pdf.savefig(fig)

    pdf.close()

    # trial_df = amp.find_mean(reward_samples, subj_id,
    #  session_n, reward_code)
        # fig, fig_name = vz.visualize(epoch_samples.trial_sample.unique(),
        # epoch_samples.z_pupil_diameter,  subj_id, session_n,
        #      reward_code, id_str='trial_' + str(trial_epoch))
        # vz.save(fig, fig_name)


    # reward_samples = vz.indicate_blinks(reward_samples, reward_events,
    # subj_id, session_n, reward_code)
    # fig, figname = vz.raster_plot(reward_samples,
    # subj_id, session_n,
    #  reward_code, n_trial_samples, id_str='corr')
    # vz.save(fig, figname)


    end_time = time.time()

    time_elapsed = end_time - start_time
    print('time elapsed: ', time_elapsed)


if __name__ == '__main__':

    main()

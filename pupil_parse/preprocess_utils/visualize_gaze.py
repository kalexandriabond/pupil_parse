from pupil_parse.preprocess_utils import config as cf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cf.plot_config()


def plot_gaze_data(sample_df_filt, fig_path,
                    subj_id, session_n,
                    reward_code, id_str=None, fig_name='test',
                      display_resolution=(1280,1024)):


    from jupyterthemes import jtplot
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'

    jtplot.style(context='poster', fscale=2, spines=False, theme='grade3')

    sns.set_context('poster', font_scale=2)
    sns.set_color_codes("muted")


    max_gaze_x, max_gaze_y = display_resolution


    fig_name = ('gaze_map' +  '_sub-' + str(subj_id) + '_sess-' +
     str(session_n) +  '_cond-' + str(reward_code))

    if id_str:
        fig_name = fig_name + '_' + id_str

    fig = plt.figure()
    plt.plot(sample_df.gaze_x, sample_df.gaze_y, '.')

    plt.xlim([0, max_gaze_x])
    plt.ylim([max_gaze_y, 0 ]) # ensure that the screen coordinate system is accurate represented (origin at upper left)


    center_x_dist = display_resolution[0]/5 # as specified in experimental code
    center_x, center_y = (max_gaze_x/2, max_gaze_y/2)
    target_y = center_y + 15 # as specified in experimental code

    left_target_x, left_target_y = center_x - center_x_dist, target_y
    right_target_x, right_target_y = center_x + center_x_dist, target_y

    marker_size = 3000

    plt.scatter(center_x, center_y, color='gray', marker='+', s=marker_size)
    plt.scatter(left_target_x, left_target_y, color='purple', marker='d', s=marker_size)
    plt.scatter(right_target_x, right_target_y, color='purple', marker='d', s=marker_size)

    if fig_name:
        plt.savefig(os.path.join(fig_path, fig_name + '.png'))

    return fig, fig_name

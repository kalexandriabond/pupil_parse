import os
from jupyterthemes import jtplot


def plot_config(theme='onedork', context='poster', fscale=1.4,
spines=False, gridlines='--'):
    """Configure plots."""

    jtplot.style(theme=theme, context=context, fscale=fscale,
    spines=spines, gridlines=gridlines)

    return None

def path_config(home=os.path.join(os.path.expanduser('~'), 'Dropbox/loki_0.5/')):
    """Configure paths for figures, and raw, intermediate, and processed data."""

    # pupil_utils_path = os.path.join(home, 'pupil_utils/')

    raw_data_path = os.path.join(home, 'data/BIDS_data/')
    intermediate_data_path = os.path.join(home, 'pupil/intermediate_data/')
    processed_data_path = os.path.join(home, 'pupil/processed_data/')

    figure_path = os.path.join(home, 'figures/')


    return (raw_data_path, intermediate_data_path,
     processed_data_path, figure_path)

import os
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
import seaborn as sns


def plot_config(theme='grade3', context='poster', fscale=1.4,
spines=False, gridlines='--'):
    """Configure plots."""

    jtplot.style(theme=theme, context=context, fscale=fscale,
    spines=spines, gridlines=gridlines)

    return None

def path_config(home=os.path.join(os.path.expanduser('~'), 'Dropbox/loki_0.5/')):
    """Configure paths for figures, and raw, intermediate, and processed data!"""

    raw_data_path = os.path.join(home, 'data/BIDS_data/')
    intermediate_data_path = os.path.join(home, 'analysis/pupil/intermediate_data/')
    processed_data_path = os.path.join(home, 'analysis/pupil/processed_data/')
    simulated_data_path = os.path.join(home, 'data/simulated_data/')

    figure_path = os.path.join(home, 'figures/')

    paths = [raw_data_path, intermediate_data_path,
    processed_data_path, figure_path, simulated_data_path]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    return (raw_data_path, intermediate_data_path,
     processed_data_path, figure_path, simulated_data_path)

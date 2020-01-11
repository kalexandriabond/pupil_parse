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

def path_config(home=os.path.join(os.path.expanduser('~')), root_folder='Dropbox/loki_0.5/',
data_folder='data/BIDS_data/'):
    """Configure paths for figures, and raw, intermediate, and processed data!"""

    project_dir = os.path.join(home, root_folder)

    raw_data_path = os.path.join(home, root_folder, data_folder)
    intermediate_data_path = os.path.join(home, root_folder, 'analysis/pupil/intermediate_data/')
    processed_data_path = os.path.join(home, root_folder, 'analysis/pupil/processed_data/')
    simulated_data_path = os.path.join(home, root_folder, 'data/simulated_data/')

    figure_path = os.path.join(home, root_folder, 'figures/')

    paths = [raw_data_path, intermediate_data_path,
    processed_data_path, figure_path, simulated_data_path]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    return (raw_data_path, intermediate_data_path,
     processed_data_path, figure_path, simulated_data_path)

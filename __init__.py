"""A package to parse pupillary data."""

from pupil_parse.config import plot_config, path_config
# from pupil_parse.pupil_utils import preprocess_utils, summary_utils
from pupil_parse.pupil_utils import * ## TODO: figure out how to import all the utils at once

_ = plot_config()
(raw_data_path, intermediate_data_path,
 processed_data_path, figure_path) = path_config()

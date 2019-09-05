import pandas as pd
import numpy as np

def subtractive_baseline_correction(segmented_samples, baseline_interval=500):
    """ Subtract baseline pupil diameter from an interval of interest. """

    trial_epochs = segmented_samples.trial_epoch.unique()

    bc_pupil_diameter = []

    for trial_epoch in trial_epochs:

        baseline_estimate = segmented_samples.loc[segmented_samples.trial_epoch == trial_epoch,
        'lowpass_pupil_diameter'][:baseline_interval].median()

        bc_epoch_data = segmented_samples.loc[segmented_samples.trial_epoch == trial_epoch,
        'lowpass_pupil_diameter'] - baseline_estimate

        bc_pupil_diameter.append(bc_epoch_data)

    if bc_pupil_diameter:
        bc_pupil_diameter_flattened = [item for sublist in bc_pupil_diameter for item in sublist]
        segmented_samples['bc_lowpass_pupil_diameter'] = bc_pupil_diameter_flattened
    else:
        segmented_samples['bc_lowpass_pupil_diameter'] = np.nan


    bc_pupil_diameter = []


    for trial_epoch in trial_epochs:

        baseline_estimate = segmented_samples.loc[segmented_samples.trial_epoch == trial_epoch,
        'highpass_pupil_diameter'][:baseline_interval].median()

        bc_epoch_data = segmented_samples.loc[segmented_samples.trial_epoch == trial_epoch,
        'highpass_pupil_diameter'] - baseline_estimate

        bc_pupil_diameter.append(bc_epoch_data)

    if bc_pupil_diameter:
        bc_pupil_diameter_flattened = [item for sublist in bc_pupil_diameter for item in sublist]
        segmented_samples['bc_highpass_pupil_diameter'] = bc_pupil_diameter_flattened
    else:
        segmented_samples['bc_highpass_pupil_diameter'] = np.nan

    return segmented_samples

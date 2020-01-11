import pandas as pd
import numpy as np

def segment(samples, messages, baseline_interval=500,
trial_end=1500, latency=2000):
    """Segment pupil data given a task-evoked latency period, trial length,
    and a baseline interval."""

    outcome_interval = trial_end + latency
    sample_window = baseline_interval + outcome_interval

    interval_onsets = messages.relative_stim_onset_time - baseline_interval
    interval_offsets = messages.relative_stim_onset_time + outcome_interval

    # remove first and last trial for samples and messages because there's no baseline interval before trial 0
    # and no post-trial interval on trial n
    interval_onsets, interval_offsets = interval_onsets[1:-1], interval_offsets[1:-1]

    truncated_messages = messages[1:-1]

    assert (interval_onsets < interval_offsets).sum() == len(interval_onsets), 'window onset is not < window offset time. check data.'
    assert (interval_offsets - interval_onsets).unique() == sample_window, 'check sample window'

    segmented_samples = pd.DataFrame()

    for interval_onset, interval_offset in zip(interval_onsets, interval_offsets):
        segmented_samples_temp = samples.loc[(samples.relative_time >= interval_onset) &
        (samples.relative_time < interval_offset)]

        segmented_samples = segmented_samples.append(segmented_samples_temp,
         ignore_index=True, verify_integrity=True)

    assert len(segmented_samples) == (len(interval_onsets) * sample_window), 'check segmented dataframe'

    segmented_samples['trial_sample'] = np.tile(range(sample_window), len(interval_onsets))
    segmented_samples['trial_epoch'] = np.repeat(range(len(interval_onsets)), sample_window)

    return segmented_samples, truncated_messages

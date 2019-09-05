from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order):
    """Takes the low and high frequencies, sampling rate, and order. Normalizes
    critical frequencies by the nyquist frequency."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut=0.01, highcut=4., fs=30., order=3):
    """Get numerator and denominator coefficient vectors from Butterworth filter
    and then apply bandpass filter to signal."""

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y


def butter_lowpass(highcut, fs, order):
    """Takes the high frequencies, sampling rate, and order. Normalizes
    critical frequencies by the nyquist frequency."""
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_lowpass_filter(signal, highcut=4., fs=30., order=3):
    """Get numerator and denominator coefficient vectors from Butterworth filter
    and then apply higpass filter to signal."""
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y

def high_bandpass_filter(samples, highpass_lowcut=10, highpass_highcut=200,
 sampling_rate=1000, order=2):
    """Allow high frequency signal to pass."""

    print('prop. nan: ', samples.pupil_diameter.isna().sum()/len(samples))

    samples['highpass_pupil_diameter'] = butter_bandpass_filter(samples.pupil_diameter,
    lowcut=highpass_lowcut, highcut=highpass_highcut,
    fs=sampling_rate, order=order)

    return samples

def low_bandpass_filter(samples, lowpass_lowcut=.01, lowpass_highcut=5,
sampling_rate=1000, order=2):
    """Allow low frequency signal to pass."""

    print('prop. nan: ', samples.pupil_diameter.isna().sum()/len(samples))


    samples['lowpass_pupil_diameter'] = butter_bandpass_filter(samples.pupil_diameter,
    lowcut=lowpass_lowcut, highcut=lowpass_highcut,
    fs=sampling_rate, order=order)

    return samples


def check_high_low_bandpass_data(samples):
    """Plot the high and low bandpass data."""

    f, axes = plt.subplots(1, 2, sharey=False)

    sns.scatterplot(x=samples.relative_time, y=samples.lowpass_pupil_diameter,
     data=samples, estimator=None, ax=axes[0], s=1)
    sns.scatterplot(x=samples.relative_time, y=samples.highpass_pupil_diameter,
    data=samples, estimator=None, ax=axes[1], s=1)

    # plt.tight_layout()

    return None

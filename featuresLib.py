import numpy as np
from scipy import stats

EPSILON = 0.00001 # For small float values

# Feature extraction functions

'''L2 norm of an array'''
def l2norm(array):
    return np.linalg.norm(array, ord=2)

'''Correlation of an array with itself'''
def autocorr(array):
    return np.correlate(array, array)[0]

'''Root mean squared of an array'''
def rms(array):
    return np.sqrt(np.mean(array ** 2))

'''Zero crossing rate of an array as a fraction of total size of array'''
def zcr(array):
    # Locations where array > 0, put -1 and 1 for rising/falling,
    # divide by total datapoints
    return len(np.nonzero(np.diff(np.sign(array)))[0]) / len(array)


def msf(freqs, psd_amps):
    '''Mean square frequency'''
    num = np.sum(np.multiply(np.resize(np.power(freqs, 2), len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def rmsf(freqs, psd_amps):
    '''Root mean square frequency'''
    return np.sqrt(msf(freqs, psd_amps))


def fc(freqs, psd_amps):
    '''Frequency center'''
    num = np.sum(np.multiply(np.resize(freqs, len(psd_amps)), psd_amps))
    denom = np.sum(psd_amps)

    # In case zero amplitude transform is ecountered
    if denom <= EPSILON:
        return EPSILON

    return np.divide(num, denom)


def vf(freqs, psd_amps):
    '''Variance frequency'''
    return msf(freqs - fc(freqs, psd_amps), psd_amps)


def rvf(freqs, psd_amps):
    '''Root variance frequency'''
    return np.sqrt(msf(freqs, psd_amps))

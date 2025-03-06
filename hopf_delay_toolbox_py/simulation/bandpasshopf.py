"""
Bandpass filtering function using FFT for the Hopf model.
"""

import numpy as np
import matplotlib.pyplot as plt
from .convert_back_to_time import convert_back_to_time

def bandpasshopf(y, freqrange, fres, do_plot=False):
    """
    Apply bandpass filtering using FFT.
    
    Parameters
    ----------
    y : array-like
        Input signal
    freqrange : tuple
        (low_freq, high_freq) frequency range in Hz
    fres : float
        Frequency resolution in Hz
    do_plot : bool, optional
        Whether to plot the results
        
    Returns
    -------
    x : array-like
        Bandpass filtered signal
    """
    Nsamples = len(y)
    Nunique_points = (Nsamples + 1) // 2
    
    fHz = np.arange(Nunique_points) * fres / Nsamples
    
    # Find frequency indices within the specified range
    freq_ind = np.where((fHz >= freqrange[0]) & (fHz <= freqrange[1]))[0]
    
    # Apply FFT and filter
    fy = np.fft.fft(y)
    fyo = fy[freq_ind]
    
    # Convert back to time domain
    x = convert_back_to_time(fyo, Nsamples, freq_ind)
    
    if do_plot:
        # Plot frequency spectrum
        fy2 = np.zeros_like(fy)
        fy2[freq_ind] = fyo
        plt.figure()
        plt.plot(fHz, np.abs(fy2[:len(fHz)]))
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        
        # Plot time series
        plt.figure()
        plt.plot(x, label='Filtered')
        plt.plot(y, 'r--', label='Original')
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    
    return x 
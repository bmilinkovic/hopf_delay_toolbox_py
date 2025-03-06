"""
Convert frequency domain data back to time domain for the Hopf model.
"""

import numpy as np

def convert_back_to_time(fyo, Nsamples, freq_indtest):
    """
    Convert frequency domain data back to time domain.
    
    Parameters
    ----------
    fyo : array-like
        Frequency domain data
    Nsamples : int
        Number of samples in the original time series
    freq_indtest : array-like
        Indices of frequencies to include
        
    Returns
    -------
    y : array-like
        Time domain signal
    """
    NumUniquePts = (Nsamples + 1) // 2
    
    if isinstance(freq_indtest, int) and freq_indtest == -1:
        freq_indtest = np.arange(1, NumUniquePts + 1)
    
    # Initialize full frequency array
    full_fyo = np.zeros(Nsamples, dtype=complex)
    
    # Place frequency components at specified indices
    full_fyo[freq_indtest] = fyo
    
    # Create symmetric frequency array
    tmp = np.zeros(Nsamples, dtype=complex)
    tmp[:len(full_fyo)] = full_fyo
    tmp[len(NumUniquePts):] += np.conj(full_fyo[1:][::-1])
    
    # Convert back to time domain
    y = np.real(np.fft.ifft(tmp))
    
    return y 
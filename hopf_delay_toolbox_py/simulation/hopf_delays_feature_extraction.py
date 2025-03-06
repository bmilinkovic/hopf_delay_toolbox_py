"""
Feature extraction for the Hopf Delay Model simulations.
Extracts spectral and global dynamical features from simulated time series.
"""

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
from pathlib import Path
from .bandpasshopf import bandpasshopf

def hopf_delays_feature_extraction(a, dt_save, MD, expK, C):
    """
    Extract spectral and global dynamical features from simulated time series.
    
    Parameters
    ----------
    a : float
        Bifurcation parameter
    dt_save : float
        Resolution or downsampling
    MD : array-like
        Range of delays in ms
    expK : array-like
        Range of couplings (expK)
    C : str
        Structural connectome identifier (e.g., 'AAL90n32s')
        
    Returns
    -------
    dict
        Dictionary containing extracted features:
        - PeakFGlobal: Global peak frequency
        - PeakFMean: Mean peak frequency across nodes
        - Sync: Global synchronization measure
        - Meta: Metastability measure
    """
    # Convert MD to seconds
    MD = np.array(MD) * 1e-3
    K = 10**expK
    
    # Define frequency bins
    fbins = 1000
    freqZ = np.arange(fbins) / (dt_save * fbins)
    
    # Initialize feature arrays
    Sync = np.zeros((len(K), len(MD)))
    Meta = np.zeros((len(K), len(MD)))
    PeakFMean = np.zeros((len(K), len(MD)))
    PeakFGlobal = np.zeros((len(K), len(MD)))
    
    # Process each combination of K and MD
    for g, k in enumerate(K):
        for d, md in enumerate(MD):
            print(f"Processing K={k}, mean Delay = {md*1e3}ms")
            
            # Format K label for file loading
            K_label = str(np.log10(k))
            K_label = K_label.replace('.', 'p')
            
            # Load simulation data
            data_dir = Path(__file__).parent.parent.parent / "Data"
            simu_file = f"a_Remote_K1E{K_label}_MD_{int(md*1e3)}a{a}.mat"
            data = sio.loadmat(data_dir / simu_file)
            Zsave = data['Zsave']
            
            # Compute Fourier transform
            Fourier_Complex = np.fft.fft(Zsave, fbins, axis=1)
            
            # Compute global peak frequency
            Fourier_Global = np.abs(np.mean(Fourier_Complex, axis=0))**2
            Imax = np.argmax(Fourier_Global)
            PeakFGlobal[g, d] = freqZ[Imax]
            
            # Compute synchronization and metastability
            Zb = np.zeros_like(Zsave)
            for n in range(Zsave.shape[0]):
                # Bandpass filter around global peak frequency
                Zb[n, :] = bandpasshopf(
                    Zsave[n, :],
                    [max(0.1, freqZ[Imax]-1), freqZ[Imax]+1],
                    1/dt_save
                )
                # Compute phase using Hilbert transform
                Zb[n, :] = np.angle(hilbert(Zb[n, :]))
            
            # Compute order parameter
            OP = np.abs(np.mean(np.exp(1j * Zb), axis=0))
            Sync[g, d] = np.mean(OP)  # Global synchronization
            Meta[g, d] = np.std(OP)    # Metastability
            
            # Compute mean peak frequency across nodes
            Fourier_Mean = np.mean(np.abs(Fourier_Complex)**2, axis=0)
            Imax = np.argmax(Fourier_Mean)
            PeakFMean[g, d] = freqZ[Imax]
    
    # Save results
    results = {
        'PeakFGlobal': PeakFGlobal,
        'PeakFMean': PeakFMean,
        'Sync': Sync,
        'Meta': Meta
    }
    
    output_file = data_dir / "Model_Spectral_Features.mat"
    sio.savemat(output_file, results)
    
    return results 
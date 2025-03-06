"""
Generate model disparity plots for MEG PSD fitting.
Visualizes best fitting points for individual subjects and their mean.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

def meg_psd_model_screening(a, MD, expK, C):
    """
    Generate model disparity plots for MEG PSD fitting.
    
    Parameters
    ----------
    a : float
        Bifurcation parameter
    MD : array-like
        Range of delays in ms
    expK : array-like
        Range of couplings (expK)
    C : str
        Structural connectome identifier (e.g., 'AAL90n32s')
    """
    # Load simulation data
    data_dir = Path(__file__).parent.parent.parent / "Data"
    data = sio.loadmat(data_dir / "MEG_sensor_PSD_Fitting.mat")
    
    Error_MEG_PSD = data['Error_MEG_PSD']
    Error_MEG_PSD_Sub = data['Error_MEG_PSD_Sub']
    
    K = 10**expK
    print(f"Running for a={a} 90AAL32")
    
    # Set up the plotting style
    plt.style.use('seaborn-white')
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Find best fit for each subject
    best_fit_subj = np.zeros(89, dtype=int)
    k_s = np.zeros(89, dtype=int)
    D_s = np.zeros(89, dtype=int)
    
    for s in range(89):
        Dist_MEG = Error_MEG_PSD_Sub[:, :, s]
        index_best_fit = np.argmin(Dist_MEG)
        best_fit_subj[s] = index_best_fit
        k_s[s], D_s[s] = np.unravel_index(index_best_fit, Dist_MEG.shape)
        print(f"Best fit to sub{s+1} for delay= {MD[D_s[s]]*1000}ms and k=1E{expK[k_s[s]]} index {index_best_fit}")
    
    # Count unique best fits
    x = np.unique(best_fit_subj)
    count = np.array([np.sum(best_fit_subj == xi) for xi in x])
    print("Unique best fits and their counts:")
    print(np.column_stack((x, count)))
    
    # Plot individual subject fits
    plt.figure(figsize=(10, 8))
    plt.imshow(Error_MEG_PSD,
              extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Squared Euclidean Distance')
    plt.title('Distance MEG PSD', fontsize=14)
    plt.ylabel('Global Coupling K', fontsize=14)
    plt.xlabel('Mean delay (ms)', fontsize=14)
    plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    
    # Plot best fits for each subject
    for s in range(89):
        plt.plot(MD[D_s[s]]*1000, expK[k_s[s]], '*w')
    plt.legend(['Best fit of individual MEG PSD'])
    plt.tight_layout()
    
    # Find best fit for mean across subjects
    index_best_fit_mean = np.argmin(Error_MEG_PSD)
    k_s_mean, D_s_mean = np.unravel_index(index_best_fit_mean, Error_MEG_PSD.shape)
    print(f"Best fit for delay= {MD[D_s_mean]*1000}ms and k=1E{expK[k_s_mean]} index {index_best_fit_mean}")
    
    # Plot mean fit
    plt.figure(figsize=(10, 8))
    plt.imshow(Error_MEG_PSD,
              extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Squared Euclidean Distance')
    plt.title('Distance MEG PSD', fontsize=14)
    plt.ylabel('Global Coupling K', fontsize=14)
    plt.xlabel('Mean delay (ms)', fontsize=14)
    plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    
    # Plot best fit for mean
    plt.plot(MD[D_s_mean]*1000, expK[k_s_mean], '*w')
    plt.legend(['Best fit of mean MEG PSD'])
    plt.tight_layout()
    
    plt.show() 
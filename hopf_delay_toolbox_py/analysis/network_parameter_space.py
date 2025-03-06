"""
Create parameter space plots for the Hopf Delay Model.
Visualizes Global Peak Frequency, Mean Peak Frequency, Metastability, and Synchrony.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

def network_parameter_space(a, MD, expK, C):
    """
    Create parameter space plots for the Hopf Delay Model.
    
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
    # Convert MD to seconds and calculate K
    MD = np.array(MD) * 1e-3
    K = 10**expK
    
    # Load simulation data
    data_dir = Path(__file__).parent.parent.parent / "Data"
    data = sio.loadmat(data_dir / "Model_Spectral_Features.mat")
    
    PeakFGlobal = data['PeakFGlobal']
    PeakFMean = data['PeakFMean']
    Sync = data['Sync']
    Meta = data['Meta']
    
    print(f"Running for a={a} 90AAL32")
    
    # Set up the plotting style
    plt.style.use('seaborn-white')
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Create figure for Global Peak Frequency
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(PeakFGlobal, 
              extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Frequency (Hz)')
    plt.title('Global Peak Frequency (Hz)', fontsize=20)
    plt.ylabel('Global Coupling K', fontsize=16)
    plt.xlabel('Mean Delay (ms)', fontsize=16)
    plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    plt.clim(0, 45)
    plt.tight_layout()
    # plt.savefig('PeakF.png')
    
    # Create figure for Synchrony
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(Sync,
              extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Synchrony', fontsize=20)
    plt.ylabel('Global Coupling K', fontsize=16)
    plt.xlabel('Mean Delay (ms)', fontsize=16)
    plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    plt.clim(0, 1)
    plt.tight_layout()
    # plt.savefig('Synch.png')
    
    # Create figure for Metastability
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(Meta,
              extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Metastability', fontsize=20)
    plt.ylabel('Global Coupling K', fontsize=16)
    plt.xlabel('Mean Delay (ms)', fontsize=16)
    plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    plt.tight_layout()
    # plt.savefig('Meta.png')
    
    # Optional: Create figure for Mean Peak Frequency
    # fig = plt.figure(figsize=(10, 8))
    # plt.imshow(PeakFMean,
    #           extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
    #           aspect='auto', origin='lower', cmap='jet')
    # plt.colorbar(label='Frequency (Hz)')
    # plt.title('Mean Peak Frequency (Hz)', fontsize=20)
    # plt.ylabel('Global Coupling K', fontsize=16)
    # plt.xlabel('Mean Delay (ms)', fontsize=16)
    # plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    # plt.clim(0, 45)
    # plt.tight_layout()
    # plt.savefig('MeanF.png')
    
    # Optional: Create figure for Spectral Entropy if available
    # if 'SpecEntropy_Global' in data:
    #     fig = plt.figure(figsize=(10, 8))
    #     plt.imshow(data['SpecEntropy_Global'],
    #               extent=[MD[0]*1e3, MD[-1]*1e3, np.log10(K[0]), np.log10(K[-1])],
    #               aspect='auto', origin='lower', cmap='jet')
    #     plt.colorbar()
    #     plt.title('Spectral Entropy', fontsize=20)
    #     plt.ylabel('Global Coupling K', fontsize=16)
    #     plt.xlabel('Mean Delay (ms)', fontsize=16)
    #     plt.yticks(np.log10(K), ['0.1', '', '1', '', '10', ''])
    #     plt.clim(0, 1)
    #     plt.tight_layout()
    #     plt.savefig('Spec_Entropy.png')
    
    plt.show() 
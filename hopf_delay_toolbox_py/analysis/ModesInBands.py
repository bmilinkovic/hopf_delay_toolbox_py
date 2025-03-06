"""
Analyze frequency-specific modes in different frequency bands.
Calculates and visualizes covariance matrices and eigenvalues for different simulation conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import hilbert
from scipy.linalg import eigh
from pathlib import Path
from ..simulation.bandpasshopf import bandpasshopf
from ..utils.subplot_tight import subplot_tight

def ModesInBands():
    """
    Analyze frequency-specific modes in different frequency bands.
    Calculates and visualizes covariance matrices and eigenvalues for different simulation conditions.
    """
    # Simulation files and names
    simulation_files = [
        'd4_HCP_Sim_Cluster_K1E1_MD0',
        'd4_HCP_Sim_Cluster_K1E-1_MD3',
        'Independent_AAL_HCP_Simu_K1E1_MD3',
        'd4_HCP_Sim_Cluster_K1E1p7_MD3',
        'AAL_HCP_Simu_K1E1_MD10'
    ]
    simulation_names = ['No delays', 'Weak K', 'Intermediate K', 'Strong K', 'Long delays']
    
    # Parameters
    N = 90
    Order = np.concatenate([np.arange(0, N, 2), np.arange(N-1, 0, -2)])
    
    # Frequency bands
    delta = [0.5, 4]
    theta = [4, 8]
    alpha = [8, 13]
    beta = [13, 30]
    gamma = [30, 60]
    
    n_simu = len(simulation_names)
    
    # Initialize arrays for results
    N_Modes_Delta = np.zeros(n_simu)
    N_Modes_Theta = np.zeros(n_simu)
    N_Modes_Alpha = np.zeros(n_simu)
    N_Modes_Beta = np.zeros(n_simu)
    N_Modes_Gamma = np.zeros(n_simu)
    
    Spatial_Modes_Delta = {}
    Spatial_Modes_Theta = {}
    Spatial_Modes_Alpha = {}
    Spatial_Modes_Beta = {}
    Spatial_Modes_Gamma = {}
    
    Val_Modes_Delta = {}
    Val_Modes_Theta = {}
    Val_Modes_Alpha = {}
    Val_Modes_Beta = {}
    Val_Modes_Gamma = {}
    
    # Create figure for covariance matrices
    plt.figure(figsize=(15, 12))
    
    # Load and process each simulation
    data_dir = Path(__file__).parent.parent.parent / "Data"
    
    for simu in range(n_simu):
        print(simulation_names[simu])
        
        # Load simulation data
        data = sio.loadmat(data_dir / f"{simulation_files[simu]}.mat")
        Zsave = data['Zsave']
        dt_save = data['dt_save']
        
        # Normalize and reorder data
        Zsave = Zsave / (5 * np.std(Zsave))
        Zsave = Zsave[Order, :]
        
        # Initialize filtered arrays
        Zdelta = np.zeros_like(Zsave)
        Ztheta = np.zeros_like(Zsave)
        Zalpha = np.zeros_like(Zsave)
        Zbeta = np.zeros_like(Zsave)
        Zgamma = np.zeros_like(Zsave)
        
        # Apply bandpass filtering
        for n in range(90):
            Zdelta[n, :] = bandpasshopf(Zsave[n, :], delta, 1/dt_save)
            Ztheta[n, :] = bandpasshopf(Zsave[n, :], theta, 1/dt_save)
            Zalpha[n, :] = bandpasshopf(Zsave[n, :], alpha, 1/dt_save)
            Zbeta[n, :] = bandpasshopf(Zsave[n, :], beta, 1/dt_save)
            Zgamma[n, :] = bandpasshopf(Zsave[n, :], gamma, 1/dt_save)
        
        # Remove edge effects
        edge_samples = int(1/dt_save)
        Zdelta = Zdelta[:, edge_samples:-edge_samples]
        Ztheta = Ztheta[:, edge_samples:-edge_samples]
        Zalpha = Zalpha[:, edge_samples:-edge_samples]
        Zbeta = Zbeta[:, edge_samples:-edge_samples]
        Zgamma = Zgamma[:, edge_samples:-edge_samples]
        
        # Calculate amplitude envelopes
        Env_Delta = np.abs(hilbert(Zdelta.T)).T
        Env_Theta = np.abs(hilbert(Ztheta.T)).T
        Env_Alpha = np.abs(hilbert(Zalpha.T)).T
        Env_Beta = np.abs(hilbert(Zbeta.T)).T
        Env_Gamma = np.abs(hilbert(Zgamma.T)).T
        
        # Calculate covariance matrices
        FC_Delta = np.cov(Env_Delta)
        FC_Theta = np.cov(Env_Theta)
        FC_Alpha = np.cov(Env_Alpha)
        FC_Beta = np.cov(Env_Beta)
        FC_Gamma = np.cov(Env_Gamma)
        
        # Calculate eigenvalues
        EigVal_Delta = np.sort(np.linalg.eigvals(FC_Delta))[::-1]
        EigVal_Theta = np.sort(np.linalg.eigvals(FC_Theta))[::-1]
        EigVal_Alpha = np.sort(np.linalg.eigvals(FC_Alpha))[::-1]
        EigVal_Beta = np.sort(np.linalg.eigvals(FC_Beta))[::-1]
        EigVal_Gamma = np.sort(np.linalg.eigvals(FC_Gamma))[::-1]
        
        # Plot covariance matrices
        Isubdiag = np.tril_indices(90, -1)
        
        # Beta band
        plt.subplot(4, n_simu, simu + 1)
        lim = 4 * np.std(np.abs(FC_Beta[Isubdiag]))
        plt.imshow(FC_Beta, cmap='jet', vmin=-lim, vmax=lim)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.title(simulation_names[simu], fontsize=12)
        plt.colorbar()
        
        # Alpha band
        plt.subplot(4, n_simu, simu + 1 + n_simu)
        lim = 4 * np.std(np.abs(FC_Alpha[Isubdiag]))
        plt.imshow(FC_Alpha, cmap='jet', vmin=-lim, vmax=lim)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        
        # Theta band
        plt.subplot(4, n_simu, simu + 1 + 2*n_simu)
        lim = 4 * np.std(np.abs(FC_Theta[Isubdiag]))
        plt.imshow(FC_Theta, cmap='jet', vmin=-lim, vmax=lim)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        
        # Delta band
        plt.subplot(4, n_simu, simu + 1 + 3*n_simu)
        lim = 5 * np.std(FC_Delta[Isubdiag])
        plt.imshow(FC_Delta, cmap='jet', vmin=-lim, vmax=lim)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        
        # Calculate number of modes above threshold
        if simu == 0:
            EigVal_Delta_Thres = EigVal_Delta[0]
            EigVal_Theta_Thres = EigVal_Theta[0]
            EigVal_Alpha_Thres = EigVal_Alpha[0]
            EigVal_Beta_Thres = EigVal_Beta[0]
            EigVal_Gamma_Thres = EigVal_Gamma[0]
        else:
            N_Modes_Delta[simu] = np.sum(EigVal_Delta > EigVal_Delta_Thres)
            N_Modes_Theta[simu] = np.sum(EigVal_Theta > EigVal_Theta_Thres)
            N_Modes_Alpha[simu] = np.sum(EigVal_Alpha > EigVal_Alpha_Thres)
            N_Modes_Beta[simu] = np.sum(EigVal_Beta > EigVal_Beta_Thres)
            N_Modes_Gamma[simu] = np.sum(EigVal_Gamma > EigVal_Gamma_Thres)
            
            # Calculate spatial modes
            if N_Modes_Delta[simu] > 0:
                Spatial_Modes_Delta[simu], Val_Delta = eigh(FC_Delta, subset_by_index=[N-N_Modes_Delta[simu], N-1])
                Val_Modes_Delta[simu] = np.diag(Val_Delta)
            
            if N_Modes_Theta[simu] > 0:
                Spatial_Modes_Theta[simu], Val_Theta = eigh(FC_Theta, subset_by_index=[N-N_Modes_Theta[simu], N-1])
                Val_Modes_Theta[simu] = np.diag(Val_Theta)
            
            if N_Modes_Alpha[simu] > 0:
                Spatial_Modes_Alpha[simu], Val_Alpha = eigh(FC_Alpha, subset_by_index=[N-N_Modes_Alpha[simu], N-1])
                Val_Modes_Alpha[simu] = np.diag(Val_Alpha)
            
            if N_Modes_Beta[simu] > 0:
                Spatial_Modes_Beta[simu], Val_Beta = eigh(FC_Beta, subset_by_index=[N-N_Modes_Beta[simu], N-1])
                Val_Modes_Beta[simu] = np.diag(Val_Beta)
            
            if N_Modes_Gamma[simu] > 0:
                Spatial_Modes_Gamma[simu], Val_Gamma = eigh(FC_Gamma, subset_by_index=[N-N_Modes_Gamma[simu], N-1])
                Val_Modes_Gamma[simu] = np.diag(Val_Gamma)
    
    plt.tight_layout()
    
    # Create figure for number of modes
    plt.figure(figsize=(15, 5))
    
    for simu in range(n_simu):
        plt.subplot(1, n_simu, simu + 1)
        plt.barh(1, N_Modes_Delta[simu], color=[1, 0.8, 0.5])
        plt.barh(2, N_Modes_Theta[simu], color=[1, 0.7, 0.7])
        plt.barh(3, N_Modes_Alpha[simu], color=[0.7, 0.7, 1])
        plt.barh(4, N_Modes_Beta[simu], color=[0.7, 1, 0.7])
        plt.yticks([1, 2, 3, 4], ['δ', 'θ', 'α', 'β'])
        plt.ylabel('Frequency bands', fontsize=12)
        plt.xlabel('Eigenvalues > baseline', fontsize=12)
        plt.title(simulation_names[simu], fontsize=12)
        plt.ylim(0, 5)
        plt.xlim(0, 28)
    
    plt.tight_layout()
    plt.show() 
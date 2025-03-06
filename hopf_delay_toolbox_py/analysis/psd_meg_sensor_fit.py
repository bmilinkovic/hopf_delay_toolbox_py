"""
Compare Hopf model simulations with MEG empirical data.
Calculates and compares power spectral densities (PSD) between model and empirical data.
"""

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from pathlib import Path

def psd_meg_sensor_fit(a, dt_save, MD, expK, C):
    """
    Compare Hopf model simulations with MEG empirical data.
    
    Parameters
    ----------
    a : float
        Bifurcation parameter
    dt_save : float
        Simulation resolution
    MD : array-like
        Range of mean delays in ms
    expK : array-like
        Range of couplings (expK)
    C : str
        Structural connectome identifier (e.g., 'AAL90n32s')
    """
    # Define simulation parameters
    K = 10**expK
    MD = np.array(MD) * 1e-3  # Convert to seconds
    fbins = 1000
    freqZ = np.arange(fbins) / (dt_save * fbins)
    ind150Hz = np.where(freqZ == 150)[0][0]  # Range of interest
    
    # Initialize arrays for results
    PSD_Simu_Global = np.zeros((len(K), len(MD), ind150Hz))
    Error_MEG_PSD = np.zeros((len(K), len(MD)))
    Fit_MEG_PSD = np.zeros((len(K), len(MD)))
    Pdist_Error_MEG_PSD = np.zeros((len(K), len(MD)))
    Error_MEG_PSD_Sub = np.zeros((len(K), len(MD), 89))
    Error_MEG_PSD_Sub_squared = np.zeros((len(K), len(MD), 89))
    Corr_Fit_MEG_PSD = np.zeros((len(K), len(MD)))
    Corr_MEG_PSD_Sub = np.zeros((len(K), len(MD), 89))
    
    # Load empirical data
    data_dir = Path(__file__).parent.parent.parent / "Data"
    meg_data = sio.loadmat(data_dir / "MEG_MeanPSD_Planar_89.mat")
    mean_data = sio.loadmat(data_dir / "Mean_PSD_Planar_89.mat")
    
    MEG_MeanPSD_Planar_89 = meg_data['MEG_MeanPSD_Planar_89']
    Mean_PSD_Planar_89 = mean_data['Mean_PSD_Planar_89']
    
    # Process each combination of K and MD
    for g, k in enumerate(K):
        for d, md in enumerate(MD):
            print(f"Processing K={k}, mean Delay = {md*1e3}ms")
            
            # Format K label for file loading
            K_label = str(np.log10(k))
            K_label = K_label.replace('.', 'p')
            
            # Load simulation data
            simu_file = f"a_Remote_K1E{K_label}_MD_{int(md*1e3)}a{a}.mat"
            data = sio.loadmat(data_dir / simu_file)
            Zsave = data['Zsave']
            
            # Calculate PSD for simulated data
            Fourier_Simu_Global = np.fft.fft(np.mean(Zsave, axis=0), fbins)
            Simu_PSD_MEAN = np.abs(Fourier_Simu_Global[:ind150Hz])**2
            Simu_PSD_MEAN = Simu_PSD_MEAN / np.sum(Simu_PSD_MEAN)
            
            # Calculate error metrics
            Error_MEG_PSD[g, d] = pdist([np.cumsum(Mean_PSD_Planar_89), 
                                        np.cumsum(Simu_PSD_MEAN)], 
                                       metric='sqeuclidean')
            
            Fit_MEG_PSD[g, d] = pearsonr(np.cumsum(Mean_PSD_Planar_89), 
                                        np.cumsum(Simu_PSD_MEAN))[0]
            
            Pdist_Error_MEG_PSD[g, d] = pdist([Mean_PSD_Planar_89, Simu_PSD_MEAN], 
                                             metric='sqeuclidean')
            
            Corr_Fit_MEG_PSD[g, d] = pearsonr(Mean_PSD_Planar_89, Simu_PSD_MEAN)[0]
            
            # Process individual subjects
            for s in range(89):
                Subj_PSD_Planar = MEG_MeanPSD_Planar_89[s, :] / np.sum(MEG_MeanPSD_Planar_89[s, :])
                Error_MEG_PSD_Sub[g, d, s] = pdist([np.cumsum(Subj_PSD_Planar), 
                                                   np.cumsum(Simu_PSD_MEAN)])
                Error_MEG_PSD_Sub_squared[g, d, s] = pdist([np.cumsum(Subj_PSD_Planar), 
                                                          np.cumsum(Simu_PSD_MEAN)], 
                                                         metric='sqeuclidean')
                Corr_MEG_PSD_Sub[g, d, s] = pearsonr(np.cumsum(Subj_PSD_Planar), 
                                                    np.cumsum(Simu_PSD_MEAN))[0]
            
            PSD_Simu_Global[g, d, :] = Simu_PSD_MEAN
    
    # Save results
    results = {
        'MD': MD,
        'expK': expK,
        'Error_MEG_PSD': Error_MEG_PSD,
        'PSD_Simu_Global': PSD_Simu_Global,
        'Error_MEG_PSD_Sub': Error_MEG_PSD_Sub,
        'Corr_Fit_MEG_PSD': Corr_Fit_MEG_PSD,
        'Fit_MEG_PSD': Fit_MEG_PSD,
        'Pdist_Error_MEG_PSD': Pdist_Error_MEG_PSD,
        'Corr_MEG_PSD_Sub': Corr_MEG_PSD_Sub,
        'Error_MEG_PSD_Sub_squared': Error_MEG_PSD_Sub_squared
    }
    
    output_file = data_dir / "MEG_sensor_PSD_Fitting.mat"
    sio.savemat(output_file, results) 
"""
Simulation of spontaneous whole-brain network activity using the Hopf model with delays.

Each brain area is represented by a dynamical unit with a supercritical Hopf bifurcation.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import os

def hopf_delays_simu(f, K, MD, SynDelay, sig, *args):
    """
    Run simulations of spontaneous whole-brain network activity.
    
    Parameters
    ----------
    f : float
        Fundamental frequency of the natural limit cycle in Hz (e.g., f=40Hz)
    K : float
        Global Coupling Weight
    MD : float
        Mean delay in seconds that scales the distance matrix
    SynDelay : float
        Additional synaptic delay in seconds
    sig : float
        Noise amplitude
    *args : tuple, optional
        Additional arguments in order: [C, D, tmax, t_prev, dt_save]
        If not provided, default values will be used
        
    Returns
    -------
    Zsave : ndarray
        Simulated brain activity
    dt_save : float
        Temporal resolution of saved data
    """
    save_data = True  # Set to False when called from another function
    
    # Handle variable input arguments
    if len(args) == 0:
        # Load default structural network
        data_dir = Path(__file__).parent.parent.parent / "Data" / "Structural Connectivity"
        mat_data = sio.loadmat(data_dir / "SC_90aal_32HCP.mat")
        red_mat = mat_data['mat']
        N = red_mat.shape[0]
        C = red_mat / np.mean(red_mat[~np.eye(N, dtype=bool)])
        
        # Load distance matrix
        D = mat_data['mat_D']
        D = D / 1000  # Convert to meters
        
        # Default simulation parameters (shorter for testing)
        tmax = 5  # seconds (reduced from 50)
        t_prev = 0.5  # seconds (reduced from 5)
        dt_save = 2e-3  # seconds
    else:
        C, D, tmax, t_prev, dt_save = args
    
    # Model parameters
    a = -5
    N = C.shape[0]  # Number of areas
    dt = 1e-4  # Resolution of model integration in seconds
    iomega = 1j * 2 * np.pi * f * np.ones(N)  # complex frequency in radians/second
    
    # Scale coupling matrix and noise
    kC = K * C * dt
    dsig = np.sqrt(dt) * sig
    
    # Calculate delays matrix
    if MD == 0:
        Delays = np.zeros((N, N))
    else:
        Delays = np.round((D / np.mean(D[C > 0]) * MD + SynDelay) / dt)
    Delays[C == 0] = 0
    
    Max_History = int(np.max(Delays)) + 1
    Delays = Max_History - Delays
    
    # Initialize history
    Z = dt * (np.random.randn(N, Max_History) + 1j * np.random.randn(N, Max_History))
    Zsave = np.zeros((N, int(tmax/dt_save)), dtype=complex)
    sumz = np.zeros(N, dtype=complex)
    
    print(f"Running simulation for K={K}, mean Delay = {MD*1e3}ms, a={a}")
    
    nt = 0
    for t in np.arange(dt, t_prev + tmax + dt, dt):
        Znow = Z[:, -1]  # Current state
        
        # Intrinsic dynamics
        dz = Znow * (a + iomega - np.abs(Znow**2)) * dt
        
        # Coupling term
        for n in range(N):
            sumzn = 0
            for p in range(N):
                if kC[n, p]:
                    sumzn += kC[n, p] * Z[p, int(Delays[n, p])-1]
            sumz[n] = sumzn - np.sum(kC[n, :] * Znow[n])
        
        if MD:
            # Slide history
            Z[:, :-1] = Z[:, 1:]
        
        # Update state
        Z[:, -1] = (Znow + dz + sumz + 
                   dsig * (np.random.randn(N) + 1j * np.random.randn(N)))
        
        # Save data
        if not t % dt_save and t > t_prev:
            Zsave[:, nt] = Z[:, -1]
            nt += 1
    
    if save_data:
        K_label = str(np.log10(K))
        K_label = K_label.replace('.', 'p')
        output_file = f"Hopf_Simu_K1E{K_label}_MD{int(MD*1e3)}.mat"
        sio.savemat(output_file, {
            'Zsave': Zsave,
            'dt_save': dt_save,
            'K': K,
            'MD': MD,
            'f': f,
            'sig': sig,
            'a': a
        })
    
    return Zsave, dt_save 
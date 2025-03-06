"""
Full analysis pipeline for the Hopf Delay Model.
This script demonstrates the complete workflow from simulation to visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from multiprocessing import Pool, cpu_count
import pandas as pd
from hopf_delay_toolbox_py import (
    hopf_delays_simu,
    hopf_delays_feature_extraction,
    network_parameter_space,
    psd_meg_sensor_fit,
    meg_psd_model_screening
)

def save_simulation_results(Zsave, dt_save, K, MD, output_dir):
    """Save simulation results in both .npy and .csv formats."""
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    base_filename = f"K{K:.2f}_MD{MD*1000:.1f}ms"
    
    # Save as .npy (numpy binary format)
    np.save(os.path.join(output_dir, f"{base_filename}.npy"), Zsave)
    
    # Save as .csv (more portable format)
    # Convert to DataFrame with time as index
    time = np.arange(Zsave.shape[1]) * dt_save
    df = pd.DataFrame(Zsave.T, index=time, columns=[f'Region_{i+1}' for i in range(Zsave.shape[0])])
    df.index.name = 'Time (s)'
    df.to_csv(os.path.join(output_dir, f"{base_filename}.csv"))
    
    # Save parameters in a separate metadata file
    metadata = {
        'K': K,
        'MD': MD,
        'dt_save': dt_save,
        'n_regions': Zsave.shape[0],
        'n_timepoints': Zsave.shape[1]
    }
    pd.Series(metadata).to_csv(os.path.join(output_dir, f"{base_filename}_metadata.csv"))

def run_single_simulation(params):
    """Run a single simulation with given parameters and return results."""
    expK, MD, i, j = params
    K = 10**expK
    print(f"Processing K={K:.2f}, MD={MD*1000:.1f}ms")
    
    # Run simulation
    Zsave, dt_save = hopf_delays_simu(
        f=40,          # Fundamental frequency in Hz
        K=K,           # Global coupling weight
        MD=MD,         # Mean delay in seconds
        SynDelay=0,    # Additional synaptic delay
        sig=0.1        # Noise amplitude
    )
    
    # Save simulation results
    save_simulation_results(Zsave, dt_save, K, MD, 'output/simulations')
    
    # Calculate features
    # Global Peak Frequency
    Fourier_Global = np.abs(np.mean(np.fft.fft(Zsave, axis=1), axis=0))**2
    PeakFGlobal = np.argmax(Fourier_Global) / (dt_save * Zsave.shape[1])
    
    # Mean Peak Frequency
    Fourier_Mean = np.mean(np.abs(np.fft.fft(Zsave, axis=1))**2, axis=0)
    PeakFMean = np.argmax(Fourier_Mean) / (dt_save * Zsave.shape[1])
    
    # Synchrony and Metastability
    Zb = np.zeros_like(Zsave)
    for n in range(Zsave.shape[0]):
        # Take real part before Hilbert transform
        Zb[n, :] = np.angle(hilbert(np.real(Zsave[n, :])))
    OP = np.abs(np.mean(np.exp(1j * Zb), axis=0))
    Sync = np.mean(OP)
    Meta = np.std(OP)
    
    return i, j, PeakFGlobal, PeakFMean, Sync, Meta, Zsave, dt_save

def main():
    # Create output directory for plots
    os.makedirs('output', exist_ok=True)

    # Define parameter ranges
    MD_range = np.array([0.001, 0.005, 0.01, 0.025])  # Delays from 1ms to 25ms
    expK_range = np.array([-1, -0.5, 0, 0.5, 1])  # Coupling strengths from 0.1 to 10

    # Initialize arrays for results
    n_MD = len(MD_range)
    n_K = len(expK_range)
    PeakFGlobal = np.zeros((n_K, n_MD))
    PeakFMean = np.zeros((n_K, n_MD))
    Sync = np.zeros((n_K, n_MD))
    Meta = np.zeros((n_K, n_MD))

    # Prepare parameters for parallel processing
    params = [(expK, MD, i, j) 
              for i, expK in enumerate(expK_range)
              for j, MD in enumerate(MD_range)]

    # Run simulations in parallel
    print("Running simulations in parallel...")
    n_cores = cpu_count()  # Use all available CPU cores
    with Pool(processes=n_cores) as pool:
        results = pool.map(run_single_simulation, params)

    # Process results
    print("Processing results...")
    last_Zsave = None
    last_dt_save = None
    for i, j, peak_f_global, peak_f_mean, sync, meta, Zsave, dt_save in results:
        PeakFGlobal[i, j] = peak_f_global
        PeakFMean[i, j] = peak_f_mean
        Sync[i, j] = sync
        Meta[i, j] = meta
        last_Zsave = Zsave  # Save the last simulation results
        last_dt_save = dt_save

    # Create network parameter space plots
    print("Creating network parameter space plots...")
    plt.figure(figsize=(15, 10))

    # Global Peak Frequency
    plt.subplot(2, 2, 1)
    plt.imshow(PeakFGlobal,
              extent=[MD_range[0]*1e3, MD_range[-1]*1e3, expK_range[0], expK_range[-1]],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Frequency (Hz)')
    plt.title('Global Peak Frequency (Hz)', fontsize=12)
    plt.ylabel('Global Coupling K (log10)', fontsize=10)
    plt.xlabel('Mean Delay (ms)', fontsize=10)
    plt.clim(0, 45)

    # Synchrony
    plt.subplot(2, 2, 2)
    plt.imshow(Sync,
              extent=[MD_range[0]*1e3, MD_range[-1]*1e3, expK_range[0], expK_range[-1]],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Synchrony', fontsize=12)
    plt.ylabel('Global Coupling K (log10)', fontsize=10)
    plt.xlabel('Mean Delay (ms)', fontsize=10)
    plt.clim(0, 1)

    # Metastability
    plt.subplot(2, 2, 3)
    plt.imshow(Meta,
              extent=[MD_range[0]*1e3, MD_range[-1]*1e3, expK_range[0], expK_range[-1]],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Metastability', fontsize=12)
    plt.ylabel('Global Coupling K (log10)', fontsize=10)
    plt.xlabel('Mean Delay (ms)', fontsize=10)

    # Mean Peak Frequency
    plt.subplot(2, 2, 4)
    plt.imshow(PeakFMean,
              extent=[MD_range[0]*1e3, MD_range[-1]*1e3, expK_range[0], expK_range[-1]],
              aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Frequency (Hz)')
    plt.title('Mean Peak Frequency (Hz)', fontsize=12)
    plt.ylabel('Global Coupling K (log10)', fontsize=10)
    plt.xlabel('Mean Delay (ms)', fontsize=10)
    plt.clim(0, 45)

    plt.tight_layout()
    plt.savefig('output/network_parameter_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create PSD plot for the last simulation
    print("Creating PSD plot...")
    plt.figure(figsize=(10, 6))
    # Calculate the correct frequency range based on the data length
    freq = np.fft.fftfreq(last_Zsave.shape[1], d=last_dt_save)
    # Only plot positive frequencies
    pos_freq_mask = freq > 0
    freq = freq[pos_freq_mask]
    psd = np.abs(np.fft.fft(last_Zsave))**2 / last_Zsave.shape[1]
    psd_mean = np.mean(psd[:, pos_freq_mask], axis=0)
    plt.plot(freq, psd_mean)
    plt.title('Power Spectral Density', fontsize=12)
    plt.xlabel('Frequency (Hz)', fontsize=10)
    plt.ylabel('Power', fontsize=10)
    plt.grid(True)
    plt.savefig('output/psd.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create time series plot for the last simulation
    print("Creating time series plot...")
    plt.figure(figsize=(12, 6))
    time = np.arange(last_Zsave.shape[1]) * last_dt_save
    plt.plot(time, np.real(last_Zsave[0, :]))  # Plot first brain region
    plt.title('Time Series (First Brain Region)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid(True)
    plt.savefig('output/time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Analysis complete! Plots have been saved to the 'output' directory.")

if __name__ == '__main__':
    main() 
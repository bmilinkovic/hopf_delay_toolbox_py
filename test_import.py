from hopf_delay_toolbox_py import hopf_delays_simu
import os

# Test the simulation function
try:
    # Get the absolute path to the data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Structural Connectivity')
    sc_file = os.path.join(data_dir, 'SC_90aal_32HCP.mat')
    
    print(f"Using structural connectivity file: {sc_file}")
    # Using more reasonable parameters for testing:
    # - f=40Hz (same as before)
    # - K=1.0 (same as before)
    # - MD=0.01s (10ms instead of 10s)
    # - SynDelay=0 (same as before)
    # - sig=0.1 (same as before)
    Zsave, dt_save = hopf_delays_simu(f=40, K=1.0, MD=0.01, SynDelay=0, sig=0.1)
    print("Package imported and function called successfully!")
    print(f"Simulation shape: {Zsave.shape}")
    print(f"Time step: {dt_save}")
except Exception as e:
    print(f"Error occurred: {e}") 
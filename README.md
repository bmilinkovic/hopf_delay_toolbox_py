# Hopf Delay Toolbox Python

This is a Python implementation of the Hopf Delay Toolbox, originally developed in MATLAB. The toolbox is designed for analysing brain network dynamics using the Hopf model with delays.

## Author

- Borjan Milinkovic

## Citation

If you use this toolbox, please cite:
```
Synchronisation in the Connectome: Metastable oscillatory modes emerge from interactions in the brain spacetime network
Joana Cabral, Francesca Castaldo, Jakub Vohryzek, Vladimir Litvak, Christian Bick, Renaud Lambiotte, Karl Friston, Morten L Kringelbach, Gustavo Deco
```

## Installation

You can install the package using pip:

```bash
pip install hopf_delay_toolbox_py
```

Or install from source:

```bash
git clone https://github.com/borjan/hopf_delay_toolbox_py.git
cd hopf_delay_toolbox_py
pip install -e .
```

## Features

The toolbox includes three main components:

1. **Simulation**: Generate simulations and extract spectral properties of simulated time series
2. **Analysis and Visualisation**: Tools for analysing and visualising results
3. **Demo Data**: Includes structural connectivity matrices and model spectral features

## Usage

Basic usage example:

```python
from hopf_delay_toolbox_py import hopf_delays_simu, hopf_delays_feature_extraction

# Generate simulations
Zsave = hopf_delays_simu(f=40, K=1.0, MD=10, SynDelay=0, sig=0.1)

# Extract features
features = hopf_delays_feature_extraction(Zsave, dt_save=0.001, MD=10, expK=1.0, C=1.0)
```

## Contact

For questions and support, please contact:
- Borjan Milinkovic: borjan.milinkovic@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
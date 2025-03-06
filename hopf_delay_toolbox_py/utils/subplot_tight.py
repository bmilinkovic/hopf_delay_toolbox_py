"""
A subplot function with user-tunable margins between subplots.
"""

import numpy as np
import matplotlib.pyplot as plt

def subplot_tight(m, n, p, margins=None, **kwargs):
    """
    Create a subplot with specified margins between neighboring axes.
    
    Parameters
    ----------
    m : int
        Number of rows in the subplot grid
    n : int
        Number of columns in the subplot grid
    p : int
        Subplot position (1-based index)
    margins : float or tuple, optional
        Margins between subplots [vertical, horizontal]. If float, same margin
        is used for both directions. Default is [0.04, 0.04].
    **kwargs : dict
        Additional arguments passed to plt.axes()
        
    Returns
    -------
    axes : matplotlib.axes.Axes
        The created axes object
    """
    # Set default margins
    if margins is None:
        margins = [0.04, 0.04]
    elif isinstance(margins, (int, float)):
        margins = [margins, margins]
    
    # Convert 1-based index to 0-based index for Python
    p = p - 1
    
    # Calculate subplot position (note: Python uses 0-based indexing)
    subplot_col = p % n
    subplot_row = p // n
    
    # Calculate dimensions
    height = (1 - (m + 1) * margins[0]) / m
    width = (1 - (n + 1) * margins[1]) / n
    
    # Calculate position
    bottom = (m - subplot_row - 1) * (height + margins[0]) + margins[0]
    left = subplot_col * (width + margins[1]) + margins[1]
    
    # Create axes with specified position
    ax = plt.axes([left, bottom, width, height], **kwargs)
    
    return ax 
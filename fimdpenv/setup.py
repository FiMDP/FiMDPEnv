"""
Function to setup the visualization environment in notebooks
"""

def setup():
    from matplotlib import animation, rc
    rc('animation', html='jshtml')
    
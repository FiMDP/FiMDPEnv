__version__ = "1.0.2"


def setup():
    """
    Function to setup the visualization environment in notebooks
    """
    
    from matplotlib import animation, rc
    rc('animation', html='jshtml')
    
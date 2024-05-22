import numpy as np


def get_initial_positions(cartesian_pos, r, n):
    """Initial position for actor in the form of a ring

    Parameters
    ----------
    cartesian_pos : array
        Initial cartesian pos
    r : float
        Radius of formation
    n : int
        Number of points in the circle

    Returns
    -------
    array
        An array of points in the form of a ring
    """
    positions = []
    t = np.linspace(0, 2 * np.pi, n)
    x = cartesian_pos[0] + r * np.cos(t)
    y = cartesian_pos[1] + r * np.sin(t)
    positions = np.asarray([x, y, x * 0 + 1]).T.tolist()
    return positions

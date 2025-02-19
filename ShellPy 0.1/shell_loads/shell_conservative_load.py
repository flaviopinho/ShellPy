import numpy as np


class ConcentratedForce:
    """
    This class represents a concentrated force applied at a specific location on the shell.
    The force is defined by its components in the x, y, and z directions, as well as its position in the xi1 and xi2 coordinates.
    """

    def __init__(self, Fx, Fy, Fz, xi1, xi2):
        """
        Initializes the concentrated force with its components and application position.

        :param Fx: The force component in the x-direction.
        :param Fy: The force component in the y-direction.
        :param Fz: The force component in the z-direction.
        :param xi1: The xi1 coordinate of the application point.
        :param xi2: The xi2 coordinate of the application point.
        """
        # Store the force components as a 3x1 numpy array (Fx, Fy, Fz)
        self.load_vector = np.array([[Fx], [Fy], [Fz]])

        # Store the position of the force application point as a 2x1 numpy array (xi1, xi2)
        self.position = np.array([[xi1], [xi2]])


class PressureLoad:
    """
    This class represents a pressure load applied uniformly over an area.
    The load is defined by its magnitude (pressure) applied to the surface.
    """

    def __init__(self, pressure):
        """
        Initializes the pressure load with its magnitude.

        :param pressure: The pressure magnitude applied uniformly over the surface.
        """
        # Store the pressure magnitude as a single value
        self.pressure = pressure

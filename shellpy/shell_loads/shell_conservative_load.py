import numpy as np


class ConcentratedForceGlobal:
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


class ConcentratedForceLocal:
    """
    This class represents a concentrated force applied at a specific location on the shell.
    The force is defined by its components in the M1, M2, and M3 directions, as well as its position in the xi1 and xi2 coordinates.
    """

    def __init__(self, F1, F2, F3, xi1, xi2):
        """
        Initializes the concentrated force with its components and application position.

        :param F1: The force component in the M1-direction.
        :param F2: The force component in the M2-direction.
        :param F3: The force component in the M3-direction.
        :param xi1: The xi1 coordinate of the application point.
        :param xi2: The xi2 coordinate of the application point.
        """
        # Store the force components as a 3x1 numpy array (F1, F2, F3)
        self.load_vector = np.array([[F1], [F2], [F3]])

        # Store the position of the force application point as a 2x1 numpy array (xi1, xi2)
        self.position = np.array([[xi1], [xi2]])


class LineLoadGlobal:
    """
    Represents a distributed line load applied along a specific parametric curve
    (either xi1 = constant or xi2 = constant) on the shell.
    The magnitude is force per unit length (e.g., N/m).
    """

    def __init__(self, qx, qy, qz, line_along: str, constant_coord: float, start_coord: float, end_coord: float):
        """
        :param qx: The load component in the global x-direction (scalar or callable).
        :param qy: The load component in the global y-direction (scalar or callable).
        :param qz: The load component in the global z-direction (scalar or callable).
        :param line_along: 'xi1' if the line runs parallel to the xi1 axis, 'xi2' if parallel to xi2.
        :param constant_coord: The value of the fixed coordinate (e.g., if line_along='xi1', this is the value of xi2).
        :param start_coord: The starting boundary of the line.
        :param end_coord: The ending boundary of the line.
        """
        self.qx = qx
        self.qy = qy
        self.qz = qz

        if line_along not in ['xi1', 'xi2']:
            raise ValueError("line_along must be either 'xi1' or 'xi2'")

        self.line_along = line_along
        self.constant_coord = constant_coord
        self.start_coord = start_coord
        self.end_coord = end_coord

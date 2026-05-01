import numpy as np


class ConcentratedForce:
    """
    Represents a concentrated force applied at a specific location on the shell.
    Can be defined in either Global (x, y, z) or Local (M1, M2, M3) coordinates.
    """

    def __init__(self, f1, f2, f3, xi1, xi2, is_local: bool = True):
        """
        :param f1, f2, f3: Force components. (x,y,z if global | M1,M2,M3 if local).
        :param xi1, xi2: Coordinates of the application point.
        :param is_local: True for Local basis, False for Global (Cartesian) basis.
        """
        self.load_vector = np.array([[f1], [f2], [f3]])
        self.position = np.array([[xi1], [xi2]])
        self.is_local = is_local


class PressureLoad:
    """
    Represents a pressure load applied uniformly over an area.
    (Pressure is typically local/normal to the surface).
    """

    def __init__(self, pressure):
        self.pressure = pressure


class LineLoad:
    """
    Represents a distributed line load applied along a parametric curve
    (xi1 = const or xi2 = const).
    """

    def __init__(self, q1, q2, q3, line_along: str, constant_coord: float,
                 start_coord: float, end_coord: float, is_local: bool = False):
        """
        :param q1, q2, q3: Load components (Force/Length).
        :param line_along: 'xi1' (runs parallel to xi1) or 'xi2' (parallel to xi2).
        :param constant_coord: Fixed coordinate value.
        :param start_coord: Starting boundary.
        :param end_coord: Ending boundary.
        :param is_local: True for Local basis, False for Global basis.
        """
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        if line_along not in ['xi1', 'xi2']:
            raise ValueError("line_along must be either 'xi1' or 'xi2'")

        self.line_along = line_along
        self.constant_coord = constant_coord
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.is_local = is_local


class ArbitraryLineLoad:
    """
    Represents a line load applied over an arbitrary parametric path xi1(t), xi2(t).
    """

    def __init__(self, q1, q2, q3, xi1_func, xi2_func, dxi1_dt, dxi2_dt,
                 t_start, t_end, is_local: bool = True):
        """
        :param q1, q2, q3: Load components (Force/Length).
        :param xi1_func, xi2_func: Functions defining the path (t).
        :param dxi1_dt, dxi2_dt: Derivatives of the path functions.
        :param t_start, t_end: Parametric interval.
        :param is_local: True for Local basis, False for Global basis.
        """
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.xi1_func = xi1_func
        self.xi2_func = xi2_func
        self.dxi1_dt = dxi1_dt
        self.dxi2_dt = dxi2_dt
        self.t_start = t_start
        self.t_end = t_end
        self.is_local = is_local


class LoadCollection:
    """
    A container to handle multiple loads simultaneously.
    """

    def __init__(self, loads=None):
        self.loads = loads if loads is not None else []

    def add_load(self, load):
        self.loads.append(load)

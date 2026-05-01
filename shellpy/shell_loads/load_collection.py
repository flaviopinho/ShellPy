
class LoadCollection:
    """
    This class represents a collection of multiple loads applied to the shell.
    The total energy functional for this collection is the sum of the energies
    of its individual load components.
    """

    def __init__(self, loads=None):
        """
        Initializes the load collection.

        :param loads: A list of load objects (e.g., ConcentratedForce, PressureLoad, etc.).
        """
        if loads is None:
            self.loads = []
        else:
            # Ensure it's a list so we can append to it later if needed
            self.loads = list(loads)

    def add_load(self, load):
        """
        Adds a new load to the collection.

        :param load: A load object to be added to the shell.
        """
        self.loads.append(load)
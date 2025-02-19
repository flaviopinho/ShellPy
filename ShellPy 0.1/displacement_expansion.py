from abc import ABC, abstractmethod

displacement_field_index = {
    "u1": 0,
    "u2": 1,
    "u3": 2,
}


class DisplacementExpansion(ABC):
    @abstractmethod
    def shape_function(self, n, xi1, xi2, derivative1=0, derivative2=0):
        """
        Determine the shape function and its derivatives for the coordinates xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :param derivative1: Optional. Positive integer that represents the derivative of xi1
        :param derivative2: Optional. Positive integer that represents the derivative of xi2
        :return: A vector that contains the value of the shape function for the coordinates xi1 and xi2
        """
        pass

    @abstractmethod
    def shape_function_first_derivatives(self, n, xi1, xi2):
        """
        Determine the first derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a matrix (numpy ndarray) that contains the derivative of the shape_functions
        """
        pass

    @abstractmethod
    def shape_function_second_derivatives(self, n, xi1, xi2):
        """
        Determine the second derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a tensor (numpy ndarray) that contains the derivative of the shape_functions
        """
        pass

    @abstractmethod
    def number_of_degrees_of_freedom(self):
        """
        :return: number of degrees of freedom
        """
        pass

    @abstractmethod
    def mapping(self, n):
        """
        Determine a tuple of (field: str, mode1: int, mode2: int)
        :param n: index of the n-th shape function
        :return: Returns a tuple of (field: str, mode1: int, mode2: int)
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        :param args: state_vector (1st), xi1 (2nd) and xi2 (3rd)
        :param kwargs:
        :return: return the position vector for a given state vector and curvilinear coordinates
        """
        pass


fully_clamped = {"u1": {"xi1": ("S", "S"),
                        "xi2": ("S", "S")},
                 "u2": {"xi1": ("S", "S"),
                        "xi2": ("S", "S")},
                 "u3": {"xi1": ("C", "C"),
                        "xi2": ("C", "C")}}

pinned = {"u1": {"xi1": ("S", "S"),
                 "xi2": ("S", "S")},
          "u2": {"xi1": ("S", "S"),
                 "xi2": ("S", "S")},
          "u3": {"xi1": ("S", "S"),
                 "xi2": ("S", "S")}}

simply_supported = {"u1": {"xi1": ("F", "F"),
                           "xi2": ("S", "S")},
                    "u2": {"xi1": ("S", "S"),
                           "xi2": ("F", "F")},
                    "u3": {"xi1": ("S", "S"),
                           "xi2": ("S", "S")}}

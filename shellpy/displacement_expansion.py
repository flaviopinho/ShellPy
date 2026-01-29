from abc import ABC, abstractmethod

displacement_field_index = {
    "u1": 0,
    "u2": 1,
    "u3": 2,
    "v1": 3,
    "v2": 4,
    "v3": 5
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

    def number_of_fields(self):
        """
        :return: return the number of fields of the displacement vector. 3 for thin shell theory and 6 for FOSD theory
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

simply_supported_fsdt6 = {"u1": {"xi1": ("F", "F"),
                                 "xi2": ("S", "S")},
                          "u2": {"xi1": ("S", "S"),
                                 "xi2": ("F", "F")},
                          "u3": {"xi1": ("S", "S"),
                                 "xi2": ("S", "S")},
                          "v1": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v2": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v3": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")}}

fully_campled_fsdt6 = {"u1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u3": {"xi1": ("C", "C"),
                              "xi2": ("C", "C")},
                       "v1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v3": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}}

simply_supported_fsdt5 = {"u1": {"xi1": ("F", "F"),
                                 "xi2": ("S", "S")},
                          "u2": {"xi1": ("S", "S"),
                                 "xi2": ("F", "F")},
                          "u3": {"xi1": ("S", "S"),
                                 "xi2": ("S", "S")},
                          "v1": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v2": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v3": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")}}

fully_campled_fsdt5 = {"u1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u3": {"xi1": ("C", "C"),
                              "xi2": ("C", "C")},
                       "v1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v3": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}}

# --- Condições de contorno FSDT6 por acrônimo (ordem: xi1=0, xi1=a, xi2=0, xi2=b)
#   F (livre):      nenhuma restrição; φ livre na borda.
#   S (apoio):      valor nulo na borda, φ(edge)=0.
#   C (engastado):  valor e derivada normal nulos: φ(edge)=0 e dφ/dn(edge)=0.
# Para cada dof, {"xi1": (ξ1=0, ξ1=a), "xi2": (ξ2=0, ξ2=b)}.

# CCSS: engastado em xi1; apoio simples em xi2
CCSS_fsdt6 = {"u1": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "u3": {"xi1": ("C", "C"), "xi2": ("S", "S")},
              "v1": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "v2": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "v3": {"xi1": ("S", "S"), "xi2": ("F", "F")}}

# SSCC: apoio simples em xi1; engastado em xi2
SSCC_fsdt6 = {"u1": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "u3": {"xi1": ("S", "S"), "xi2": ("C", "C")},
              "v1": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "v2": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "v3": {"xi1": ("F", "F"), "xi2": ("S", "S")}}

# CSCS: C em xi1=0, S em xi1=a; C em xi2=0, S em xi2=b
CSCS_fsdt6 = {"u1": {"xi1": ("S", "F"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("S", "F")},
              "u3": {"xi1": ("C", "S"), "xi2": ("C", "S")},
              "v1": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "v2": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "v3": {"xi1": ("S", "F"), "xi2": ("S", "F")}}

# SCSC: S em xi1=0, C em xi1=a; S em xi2=0, C em xi2=b
SCSC_fsdt6 = {"u1": {"xi1": ("F", "S"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("F", "S")},
              "u3": {"xi1": ("S", "C"), "xi2": ("S", "C")},
              "v1": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "v2": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "v3": {"xi1": ("F", "S"), "xi2": ("F", "S")}}

# CCCC: engastado nas 4 bordas
CCCC_fsdt6 = {"u1": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "u3": {"xi1": ("C", "C"), "xi2": ("C", "C")},
              "v1": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "v2": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "v3": {"xi1": ("S", "S"), "xi2": ("S", "S")}}

# SSSS: apoio simples nas 4 bordas
SSSS_fsdt6 = {"u1": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "u2": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "u3": {"xi1": ("S", "S"), "xi2": ("S", "S")},
              "v1": {"xi1": ("F", "F"), "xi2": ("F", "F")},
              "v2": {"xi1": ("F", "F"), "xi2": ("F", "F")},
              "v3": {"xi1": ("F", "F"), "xi2": ("F", "F")}}

# CCFF: engastado em xi1; livre em xi2
CCFF_fsdt6 = {"u1": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "u2": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "u3": {"xi1": ("C", "C"), "xi2": ("F", "F")},
              "v1": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "v2": {"xi1": ("S", "S"), "xi2": ("F", "F")},
              "v3": {"xi1": ("S", "S"), "xi2": ("F", "F")}}

# FFCC: livre em xi1; engastado em xi2
FFCC_fsdt6 = {"u1": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "u2": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "u3": {"xi1": ("F", "F"), "xi2": ("C", "C")},
              "v1": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "v2": {"xi1": ("F", "F"), "xi2": ("S", "S")},
              "v3": {"xi1": ("F", "F"), "xi2": ("S", "S")}}

# CFCF: C em xi1=0, F em xi1=a; C em xi2=0, F em xi2=b
CFCF_fsdt6 = {"u1": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "u2": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "u3": {"xi1": ("C", "F"), "xi2": ("C", "F")},
              "v1": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "v2": {"xi1": ("S", "F"), "xi2": ("S", "F")},
              "v3": {"xi1": ("S", "F"), "xi2": ("S", "F")}}

# FCFC: F em xi1=0, C em xi1=a; F em xi2=0, C em xi2=b
FCFC_fsdt6 = {"u1": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "u2": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "u3": {"xi1": ("F", "C"), "xi2": ("F", "C")},
              "v1": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "v2": {"xi1": ("F", "S"), "xi2": ("F", "S")},
              "v3": {"xi1": ("F", "S"), "xi2": ("F", "S")}}

simply_supported_fsdt5 = {"u1": {"xi1": ("F", "F"),
                                 "xi2": ("S", "S")},
                          "u2": {"xi1": ("S", "S"),
                                 "xi2": ("F", "F")},
                          "u3": {"xi1": ("S", "S"),
                                 "xi2": ("S", "S")},
                          "v1": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v2": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")},
                          "v3": {"xi1": ("F", "F"),
                                 "xi2": ("F", "F")}}

fully_campled_fsdt5 = {"u1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "u3": {"xi1": ("C", "C"),
                              "xi2": ("C", "C")},
                       "v1": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v2": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")},
                       "v3": {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}}

# SSSS: Simply Supported em todas as bordas
SSSS = {"u1": {"xi1": ("F", "F"),
               "xi2": ("S", "S")},
        "u2": {"xi1": ("S", "S"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("S", "S"),
               "xi2": ("S", "S")}}

# CCCC: Clamped (engastado) em todas as bordas
CCCC = {"u1": {"xi1": ("C", "C"),
               "xi2": ("C", "C")},
        "u2": {"xi1": ("C", "C"),
               "xi2": ("C", "C")},
        "u3": {"xi1": ("C", "C"),
               "xi2": ("C", "C")}}

# SSCC: Simply Supported nas bordas xi1, Clamped nas bordas xi2
SSCC = {"u1": {"xi1": ("F", "F"),
               "xi2": ("C", "C")},
        "u2": {"xi1": ("S", "S"),
               "xi2": ("C", "C")},
        "u3": {"xi1": ("S", "S"),
               "xi2": ("C", "C")}}

# CCSS: Clamped nas bordas xi1, Simply Supported nas bordas xi2
CCSS = {"u1": {"xi1": ("C", "C"),
               "xi2": ("S", "S")},
        "u2": {"xi1": ("C", "C"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("C", "C"),
               "xi2": ("S", "S")}}

# SCCC: Simply Supported na primeira borda xi1, Clamped nas outras
SCCC = {"u1": {"xi1": ("F", "C"),
               "xi2": ("C", "C")},
        "u2": {"xi1": ("S", "C"),
               "xi2": ("C", "C")},
        "u3": {"xi1": ("S", "C"),
               "xi2": ("C", "C")}}

# CSSS: Clamped na primeira borda xi1, Simply Supported nas outras
CSSS = {"u1": {"xi1": ("C", "F"),
               "xi2": ("S", "S")},
        "u2": {"xi1": ("C", "S"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("C", "S"),
               "xi2": ("S", "S")}}

# SCSC: Simply Supported nas bordas xi1=0 e xi2=b, Clamped nas outras
SCSC = {"u1": {"xi1": ("F", "C"),
               "xi2": ("C", "S")},
        "u2": {"xi1": ("S", "C"),
               "xi2": ("C", "F")},
        "u3": {"xi1": ("S", "C"),
               "xi2": ("C", "S")}}

# CSCS: Simply Supported nas bordas xi1=a e xi2=0, Clamped nas outras
CSCS = {"u1": {"xi1": ("C", "F"),
               "xi2": ("S", "C")},
        "u2": {"xi1": ("C", "S"),
               "xi2": ("F", "C")},
        "u3": {"xi1": ("C", "S"),
               "xi2": ("S", "C")}}

# SSFF: Simply Supported nas bordas xi1, Free nas bordas xi2
SSFF = {"u1": {"xi1": ("F", "F"),
               "xi2": ("F", "F")},
        "u2": {"xi1": ("S", "S"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("S", "S"),
               "xi2": ("F", "F")}}

# FFSS: Free nas bordas xi1, Simply Supported nas bordas xi2
FFSS = {"u1": {"xi1": ("F", "F"),
               "xi2": ("S", "S")},
        "u2": {"xi1": ("F", "F"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("F", "F"),
               "xi2": ("S", "S")}}

# CCFS: Clamped nas bordas xi1, Free na primeira borda xi2, Simply Supported na segunda borda xi2
CCFS = {"u1": {"xi1": ("C", "C"),
               "xi2": ("F", "S")},
        "u2": {"xi1": ("C", "C"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("C", "C"),
               "xi2": ("F", "S")}}

# CFCF: Clamped na primeira borda xi1, Free na segunda borda xi1, alternando
CFCF = {"u1": {"xi1": ("C", "F"),
               "xi2": ("C", "F")},
        "u2": {"xi1": ("C", "F"),
               "xi2": ("C", "F")},
        "u3": {"xi1": ("C", "F"),
               "xi2": ("C", "F")}}

# CCFF: Clamped nas bordas xi1, Free nas bordas xi2
CCFF = {"u1": {"xi1": ("C", "C"),
               "xi2": ("F", "F")},
        "u2": {"xi1": ("C", "C"),
               "xi2": ("F", "F")},
        "u3": {"xi1": ("C", "C"),
               "xi2": ("F", "F")}}

# FFCC: Free nas bordas xi1, Clamped nas bordas xi2
FFCC = {"u1": {"xi1": ("F", "F"),
               "xi2": ("C", "C")},
        "u2": {"xi1": ("F", "F"),
               "xi2": ("C", "C")},
        "u3": {"xi1": ("F", "F"),
               "xi2": ("C", "C")}}

# FCFC: Free na primeira borda xi1, Clamped na segunda borda xi1, alternando
FCFC = {"u1": {"xi1": ("F", "C"),
               "xi2": ("F", "C")},
        "u2": {"xi1": ("F", "C"),
               "xi2": ("F", "C")},
        "u3": {"xi1": ("F", "C"),
               "xi2": ("F", "C")}}

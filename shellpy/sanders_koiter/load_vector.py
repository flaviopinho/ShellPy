import numpy as np

from ..koiter_tensor.koiter_load_energy import koiter_load_energy_density
from ..shell import Shell
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z


def load_vector(shell: Shell,
                n_x=n_integral_default_x,
                n_y=n_integral_default_y,
                n_z=n_integral_default_z,
                integral_method=boole_weights_simple_integral):
    """
    Computes and returns the global external load vector of the shell.

    The potential energy of the loads, as calculated in `koiter_load_energy_density`,
    represents negative work (V = -F * U). To obtain the consistent nodal force vector F
    required for the equilibrium equations, we evaluate the energy density contribution
    for each generalized degree of freedom.

    Parameters:
    -----------
    shell : Shell
        The shell object containing geometry, load definitions, and displacement expansion.
    n_x, n_y, n_z : int, optional
        Number of integration points in the xi1, xi2, and thickness directions.
    integral_method : function, optional
        Numerical integration scheme (default is Boole's rule).

    Returns:
    --------
    f_vector : np.ndarray
        The global external force vector (size n_dof).
    """
    # Determine the number of generalized coordinates (DOF) for the Ritz expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize the global force vector
    f_vector = np.zeros(n_dof)

    for i in range(n_dof):
        # Calculate each force component F_i by evaluating the external potential energy
        # density associated with the i-th displacement mode.
        f_vector[i] = koiter_load_energy_density(i, shell.load, shell, n_x, n_y, n_z, integral_method)

    return f_vector
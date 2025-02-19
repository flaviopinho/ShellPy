import numpy as np
from functionals_shells.double_integral_booles_rule import boole_weights_double_integral
from functionals_shells.multiindex import MultiIndex
from functionals_shells.shell import Shell


def fast_koiter_kinetic_energy(shell: Shell) -> MultiIndex:
    xi1, xi2, W = boole_weights_double_integral(shell.boundary)
    n = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    rho = shell.material.density

    displacement_fields = np.zeros((n_dof, 3, n, n))
    for i in range(n_dof):
        displacement_fields[i] = shell.displacement_expansion.shape_function(i, xi1, xi2)

    G = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    C = shell.material.thin_shell_constitutive_tensor(G)
    h = shell.thickness(xi1, xi2)

    kinetic_energy_tensor = (rho * h / 2) * np.einsum('ijxy, aixy, bjxy, xy, xy->ab', G, displacement_fields, displacement_fields, sqrtG, W)

    return MultiIndex.tensor_to_functional_multi_index(kinetic_energy_tensor)
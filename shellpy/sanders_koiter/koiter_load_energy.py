from typing import Any
import numpy as np
from multipledispatch import dispatch

from ..shell import Shell
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z
from ..numeric_integration.integral_weights import double_integral_weights
from ..shell_loads import (ConcentratedForce,
                           PressureLoad,
                           LineLoad,
                           ArbitraryLineLoad,
                           LoadCollection)


def koiter_load_energy(shell: Shell,
                       n_x=n_integral_default_x,
                       n_y=n_integral_default_y,
                       n_z=n_integral_default_z,
                       integral_method=boole_weights_simple_integral):
    """
    Computes the energy functional associated with the applied loads on the shell.
    """
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    energy_functional = np.zeros(n_dof)

    for i in range(n_dof):
        energy_functional[i] = koiter_load_energy_density(i, shell.load, shell, n_x, n_y, n_z, integral_method)

    return energy_functional


@dispatch(int, ConcentratedForce, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, *args):
    """
    Computes energy for a concentrated force (Global or Local).
    """
    pos = load.position
    # Deslocamentos contravariantes da expansão
    U_contra = shell.displacement_expansion.shape_function(i, pos[0], pos[1])

    if load.is_local:
        # Lógica Local: Normalização pela métrica
        G = shell.mid_surface_geometry.metric_tensor_covariant_components(pos[0], pos[1])
        P = np.array([
            load.load_vector[0] / np.sqrt(G[0, 0]),
            load.load_vector[1] / np.sqrt(G[1, 1]),
            load.load_vector[2]
        ])
        return -np.dot(np.ravel(P), np.ravel(U_contra))
    else:
        # Lógica Global: Projeção via base recíproca
        N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(pos[0], pos[1])
        U_cart = U_contra[0] * N1 + U_contra[1] * N2 + U_contra[2] * N3
        return -np.dot(np.ravel(load.load_vector), np.ravel(U_cart))


@dispatch(int, PressureLoad, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, n_x, n_y, n_z, integral_method):
    """
    Computes energy for pressure (Assumed local/normal).
    """
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    Wxy_area = sqrtG * Wxy
    U_contra = shell.displacement_expansion.shape_function(i, xi1, xi2)
    # Pressão atua na direção normal (U3)
    return -np.sum(load.pressure * U_contra[2] * Wxy_area)


@dispatch(int, LineLoad, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, n_x, n_y, n_z, integral_method):
    """
    Computes energy for a line load along xi1 or xi2 (Global or Local).
    """
    n_pts = n_x if load.line_along == 'xi1' else n_y
    xi_var, W_var = integral_method((load.start_coord, load.end_coord), n_pts)

    if load.line_along == 'xi1':
        xi1, xi2 = xi_var, np.full_like(xi_var, load.constant_coord)
    else:
        xi2, xi1 = xi_var, np.full_like(xi_var, load.constant_coord)

    # Avaliação das componentes
    q1 = load.q1(xi1, xi2) if callable(load.q1) else load.q1 * np.ones_like(xi1)
    q2 = load.q2(xi1, xi2) if callable(load.q2) else load.q2 * np.ones_like(xi1)
    q3 = load.q3(xi1, xi2) if callable(load.q3) else load.q3 * np.ones_like(xi1)

    G = shell.mid_surface_geometry.metric_tensor_covariant_components(xi1, xi2)
    ds_multiplier = np.sqrt(G[0, 0]) if load.line_along == 'xi1' else np.sqrt(G[1, 1])
    W_arc = W_var * ds_multiplier

    U_contra = shell.displacement_expansion.shape_function(i, xi1, xi2)

    if load.is_local:
        # Trabalho Local
        dot_product = (q1 / np.sqrt(G[0, 0])) * U_contra[0] + \
                      (q2 / np.sqrt(G[1, 1])) * U_contra[1] + \
                      q3 * U_contra[2]
    else:
        # Trabalho Global (Projeção cartesiana)
        N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
        U_x = U_contra[0] * N1[0] + U_contra[1] * N2[0] + U_contra[2] * N3[0]
        U_y = U_contra[0] * N1[1] + U_contra[1] * N2[1] + U_contra[2] * N3[1]
        U_z = U_contra[0] * N1[2] + U_contra[1] * N2[2] + U_contra[2] * N3[2]
        dot_product = q1 * U_x + q2 * U_y + q3 * U_z

    return -np.sum(dot_product * W_arc)


@dispatch(int, ArbitraryLineLoad, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, n_x, n_y, n_z, integral_method):
    """
    Computes energy for a line load over an arbitrary path (Global or Local).
    """
    t_vals, W_t = integral_method((load.t_start, load.t_end), n_x)

    xi1, xi2 = load.xi1_func(t_vals), load.xi2_func(t_vals)
    dxi1, dxi2 = load.dxi1_dt(t_vals), load.dxi2_dt(t_vals)

    G = shell.mid_surface_geometry.metric_tensor_covariant_components(xi1, xi2)
    # Comprimento de arco ds para caminho arbitrário
    ds_dt = np.sqrt(G[0, 0] * dxi1 ** 2 + 2 * G[0, 1] * dxi1 * dxi2 + G[1, 1] * dxi2 ** 2)
    W_arc = W_t * ds_dt

    U_contra = shell.displacement_expansion.shape_function(i, xi1, xi2)

    q1 = load.q1(xi1, xi2) if callable(load.q1) else load.q1 * np.ones_like(xi1)
    q2 = load.q2(xi1, xi2) if callable(load.q2) else load.q2 * np.ones_like(xi1)
    q3 = load.q3(xi1, xi2) if callable(load.q3) else load.q3 * np.ones_like(xi1)

    if load.is_local:
        dot_product = (q1 / np.sqrt(G[0, 0])) * U_contra[0] + \
                      (q2 / np.sqrt(G[1, 1])) * U_contra[1] + \
                      q3 * U_contra[2]
    else:
        N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
        U_x = U_contra[0] * N1[0] + U_contra[1] * N2[0] + U_contra[2] * N3[0]
        U_y = U_contra[0] * N1[1] + U_contra[1] * N2[1] + U_contra[2] * N3[1]
        U_z = U_contra[0] * N1[2] + U_contra[1] * N2[2] + U_contra[2] * N3[2]
        dot_product = q1 * U_x + q2 * U_y + q3 * U_z

    return -np.sum(dot_product * W_arc)


@dispatch(int, LoadCollection, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load_collection, shell, n_x, n_y, n_z, integral_method):
    total_energy = 0.0
    for load in load_collection.loads:
        total_energy += koiter_load_energy_density(i, load, shell, n_x, n_y, n_z, integral_method)
    return total_energy
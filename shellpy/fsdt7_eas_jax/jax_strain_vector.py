import jax
import jax.numpy as jnp
import numpy as np

from ..midsurface_geometry import MidSurfaceGeometry

# Ensure double precision (Critical for numerical stability in finite element analysis)
jax.config.update("jax_enable_x64", True)


def fsdt7_total_strain_vector_jax(mid_surface_geometry: MidSurfaceGeometry,
                                  u_phys, du_phys, v_phys, dv_phys,
                                  xi1, xi2):
    """
    Computes the total strain vector (linear + non-linear) for a 7-parameter
    shell formulation (FSDT7) using JAX for automatic differentiation.
    """

    def to_voigt_jax(eps):
        """
        Helper function to convert a 3x3 strain tensor into a 6x1 Voigt notation vector.
        Order: [e11, e22, e33, gamma23, gamma13, gamma12]
        """
        return jnp.stack([
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            eps[1, 2] + eps[2, 1],
            eps[0, 2] + eps[2, 0],
            eps[0, 1] + eps[1, 0]
        ], axis=0)

    # Retrieve differential geometry properties at the integration points (xi1, xi2)
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)
    curvature_mixed = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)
    g_contra = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)

    # Compute covariant derivatives of mid-surface displacements (dcu) and director field/rotations (dcv)
    dcu = du_phys - jnp.einsum('jia..., j...->ia...', C, u_phys)
    dcv = dv_phys - jnp.einsum('jia..., j...->ia...', C, v_phys)

    shape = np.shape(xi1)

    # =========================================================================
    # 1. LINEAR STRAINS (FSDT6/7 Kinematics)
    # =========================================================================
    eps0_lin = jnp.zeros((3, 3) + shape)
    eps1_lin = jnp.zeros((3, 3) + shape)
    eps2_lin = jnp.zeros((3, 3) + shape)

    # --- epsilon0 (Membrane and constant Transverse Shear strains) ---
    eps0_lin = eps0_lin.at[0:2, 0:2].set(
        0.5 * (dcu[0:2, 0:2] + jnp.swapaxes(dcu[0:2, 0:2], 0, 1))
    )
    eps0_lin = eps0_lin.at[0:2, 2].set(
        0.5 * (dcu[2, 0:2] + v_phys[0:2])
    )
    eps0_lin = eps0_lin.at[2, 0:2].set(eps0_lin[0:2, 2])

    # Transverse normal strain (Thickness stretching - 7th parameter)
    eps0_lin = eps0_lin.at[2, 2].set(v_phys[2])

    # --- epsilon1 (Bending/Curvature changes and linear thickness effects) ---
    eps1_lin = eps1_lin.at[0:2, 0:2].set(
        0.5 * (dcv[0:2, 0:2] + jnp.swapaxes(dcv[0:2, 0:2], 0, 1))
    )

    # Curvature coupling corrections mapping 3D shell kinematics to the mid-surface
    upsilon_dcu = 0.5 * (
            jnp.einsum('oa..., ob...->ab...', curvature_mixed, dcu[0:2, 0:2]) +
            jnp.einsum('ob..., oa...->ab...', curvature_mixed, dcu[0:2, 0:2])
    )
    eps1_lin = eps1_lin.at[0:2, 0:2].add(upsilon_dcu)

    # Linear transverse shear varying with thickness (using dv instead of dcv as per original formulation)
    eps1_lin = eps1_lin.at[0:2, 2].set(0.5 * dv_phys[2, 0:2])
    eps1_lin = eps1_lin.at[2, 0:2].set(eps1_lin[0:2, 2])

    # --- epsilon2 (Second-order thickness effects / Quadratic dependency on z) ---
    upsilon_dcv = 0.5 * (
            jnp.einsum('oa..., ob...->ab...', curvature_mixed, dcv[0:2, 0:2]) +
            jnp.einsum('ob..., oa...->ab...', curvature_mixed, dcv[0:2, 0:2])
    )
    eps2_lin = eps2_lin.at[0:2, 0:2].set(upsilon_dcv)

    # =========================================================================
    # 2. NON-LINEAR STRAINS (Coupled kinematics based on Green-Lagrange formulation)
    # =========================================================================
    eps0_nl = jnp.zeros((3, 3) + shape)
    eps1_nl = jnp.zeros((3, 3) + shape)
    eps2_nl = jnp.zeros((3, 3) + shape)

    # --- epsilon0_nl (Non-linear Membrane and Shear terms) ---
    eps0_nl = eps0_nl.at[0:2, 0:2].set(
        0.5 * jnp.einsum('oa..., ot..., tb... -> ab...', dcu, g_contra, dcu)
    )
    eps0_nl = eps0_nl.at[2, 0:2].set(
        0.5 * jnp.einsum('t..., ot..., ob... -> b...', v_phys, g_contra, dcu)
    )
    eps0_nl = eps0_nl.at[0:2, 2].set(
        0.5 * jnp.einsum('ob..., ot..., t... -> b...', dcu, g_contra, v_phys)
    )

    # Non-linear thickness stretch term
    eps0_nl = eps0_nl.at[2, 2].set(
        0.5 * jnp.einsum('o..., ot..., t... -> ...', v_phys, g_contra, v_phys)
    )

    # --- epsilon1_nl (Non-linear Membrane-Bending coupling terms) ---
    eps1_nl = eps1_nl.at[0:2, 0:2].set(
        0.5 * jnp.einsum('oa..., ot..., tb... -> ab...', dcu, g_contra, dcv) +
        0.5 * jnp.einsum('oa..., ot..., tb... -> ab...', dcv, g_contra, dcu)
    )
    eps1_nl = eps1_nl.at[2, 0:2].set(
        0.5 * jnp.einsum('t..., ot..., ob... -> b...', v_phys, g_contra, dcv)
    )
    eps1_nl = eps1_nl.at[0:2, 2].set(
        0.5 * jnp.einsum('ob..., ot..., t... -> b...', dcv, g_contra, v_phys)
    )

    # --- epsilon2_nl (Non-linear Pure Bending/Higher-order terms) ---
    eps2_nl = eps2_nl.at[0:2, 0:2].set(
        0.5 * jnp.einsum('oa..., ot..., tb... -> ab...', dcv, g_contra, dcv)
    )

    # =========================================================================
    # 3. COMBINATION AND VOIGT CONVERSION
    # =========================================================================
    # Combine linear and non-linear parts and convert to Voigt notation
    eps0_total = to_voigt_jax(eps0_lin + eps0_nl)
    eps1_total = to_voigt_jax(eps1_lin + eps1_nl)
    eps2_total = to_voigt_jax(eps2_lin + eps2_nl)

    return eps0_total, eps1_total, eps2_total
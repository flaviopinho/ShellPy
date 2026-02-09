# ============================================================================
# Analysis of an orthotropic laminated parabolic conoid shell
#
# This model follows the formulation and parameters studied in:
# H. S. Das, D. Chakravorty,
# "Natural frequencies and mode shapes of composite conoids with complicated
# boundary conditions",
# Journal of Reinforced Plastics and Composites, 27(13), 2008, pp. 1397–1419.
# https://doi.org/10.1177/0731684407086508
#
# Main objectives of this script:
#  - Build a model of an orthotropic laminated conoid shell
#  - Compute natural frequencies and vibration mode shapes
#  - Evaluate multiple boundary conditions within a single loop
# ============================================================================

import sympy as sym
import numpy as np
from scipy.linalg import eigh

from shellpy.utils.shell_mode import shell_mode
from shellpy.cache_decorator import clear_cache
# --- ShellPy: expansions, materials, and operators ---
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy import (
    ConstantThickness,
    MidSurfaceGeometry,
    RectangularMidSurfaceDomain,
    Shell,
    xi1_, xi2_,
)

# --- Available boundary conditions (FSDT, 6 DOF) ---
from shellpy.displacement_expansion import (
    SSSS_fsdt6,
    CCCC_fsdt6,
    CCSS_fsdt6,
    SSCC_fsdt6,
    CSCS_fsdt6,
    SCSC_fsdt6,
    CCFF_fsdt6,
    FFCC_fsdt6,
    CFCF_fsdt6,
    FCFC_fsdt6,
)

# ============================================================================
# MAIN BLOCK
# ============================================================================
if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # Numerical integration parameters
    # ----------------------------------------------------------------------
    integral_x = 30
    integral_y = 30
    integral_z = 8

    # ----------------------------------------------------------------------
    # Geometrical parameters (Table 1 of the reference paper)
    # ----------------------------------------------------------------------
    a = 1.0          # length
    b = 1.0          # width
    h = a / 100      # thickness
    hh = a / 5     # maximum height
    h_l = hh * 0.25  # minimum height

    # Parabolic conoid parameters
    f1 = h_l
    f2 = hh

    # Parametric domain of the midsurface
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # ----------------------------------------------------------------------
    # Definition of the conoid midsurface geometry
    # ----------------------------------------------------------------------
    Z_conoid = f1 * (1 - (1 - f2 / f1) * xi1_ / a) * (1 - (2 * xi2_ / b - 1) ** 2)
    R_ = sym.Matrix([xi1_, xi2_, Z_conoid])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    # ----------------------------------------------------------------------
    # Orthotropic material properties (dimensionless form)
    # ----------------------------------------------------------------------
    E22 = 1.0
    density = 1.0
    E11 = 25.0 * E22
    E33 = E22
    G12 = 0.5 * E22
    G13 = 0.5 * E22
    G23 = 0.2 * E22
    nu12 = 0.25
    nu13 = 0.25
    nu23 = 0.25

    # ----------------------------------------------------------------------
    # Laminate definition
    # ----------------------------------------------------------------------
    t_lamina = 1 / 2

    def create_lamina(angle_deg):
        """Creates an orthotropic lamina with a given fiber orientation."""
        return Lamina(
            E_11=E11,
            E_22=E22,
            E_33=E33,
            nu_12=nu12,
            nu_13=nu13,
            nu_23=nu23,
            G_12=G12,
            G_13=G13,
            G_23=G23,
            density=density,
            angle=angle_deg * np.pi / 180.0,
            thickness=t_lamina,
        )

    # Stacking sequence
    angles = [0, 90, 0, 90]
    laminas = [create_lamina(angle) for angle in angles]

    material = LaminateOrthotropicMaterial(laminas, thickness)

    # ----------------------------------------------------------------------
    # Boundary conditions to be investigated
    # ----------------------------------------------------------------------
    boundary_conditions = {
        "SSSS": SSSS_fsdt6,
        "CCCC": CCCC_fsdt6,
        "CCSS": CCSS_fsdt6,
        "SSCC": SSCC_fsdt6,
        "CSCS": CSCS_fsdt6,
        "SCSC": SCSC_fsdt6,
        "CCFF": CCFF_fsdt6,
        "FFCC": FFCC_fsdt6,
        "CFCF": CFCF_fsdt6,
        "FCFC": FCFC_fsdt6,
    }

    # Number of terms in the displacement expansion
    n_modos = 15

    # ----------------------------------------------------------------------
    # Dictionary to store summary results
    summary_results = {}

    shell = None

    # ----------------------------------------------------------------------
    # Main loop over boundary conditions
    # ----------------------------------------------------------------------
    for bc_name, bc_function in boundary_conditions.items():

        print(f"\n================ Boundary condition: {bc_name} ================")

        # Displacement field expansion (primary variables)
        expansion_size = {k: (n_modos, n_modos)
                          for k in ("u1", "u2", "u3", "v1", "v2", "v3")}

        displacement_field = EnrichedCosineExpansion(
            expansion_size,
            rectangular_domain,
            bc_function,
        )

        # EAS (Enhanced Assumed Strain) field
        eas_field = LegendreSeries(
            {"u1": (n_modos, n_modos)},
            rectangular_domain,
            {"u1": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
        )

        clear_cache(shell)

        # Shell model assembly
        shell = Shell(
            mid_surface_geometry,
            thickness,
            rectangular_domain,
            material,
            displacement_field,
            None,
        )

        # ------------------------------------------------------------------
        # Mass and stiffness matrices
        # ------------------------------------------------------------------
        M = mass_matrix(shell, integral_x, integral_y, integral_z)
        K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

        # ------------------------------------------------------------------
        # Generalized eigenvalue problem: K φ = λ M φ
        # ------------------------------------------------------------------
        eigen_vals, eigen_vectors = eigh(K, M)
        omega = np.sqrt(eigen_vals)

        # Filtering spurious eigenvalues
        mask = np.isfinite(omega) & (np.abs(np.real(omega)) > 1e-2)
        omega = omega[mask].real
        eigen_vectors = np.real(eigen_vectors[:, mask])

        idx = np.argsort(omega)
        omega = omega[idx]
        eigen_vectors = eigen_vectors[:, idx]

        # Natural frequencies
        freq_Hz = omega / (2 * np.pi)
        omega_bar = omega * (a ** 2) * np.sqrt(density / (E22 * h ** 2))

        # Print first modes
        n_print = 5
        print("Natural frequencies (Hz):")
        print(freq_Hz[:n_print])
        print("Non-dimensional frequency ω̄:")
        print(omega_bar[:n_print])

        # Store first non-dimensional frequency for summary
        summary_results[bc_name] = omega_bar[0]

        file_name = f"mode1_{bc_name}.png"

        shell_mode(
            shell,
            np.real(eigen_vectors[:, 0]),
            file_name,
            n_1=20,
            n_2=20,
            n_3=4,
            max_deformation=10 * h,
        )

    # ==================================================================
    # Summary of results
    # ==================================================================
    print("          == == == == == == == == Summary of non - dimensional frequencies == == == == == == == == ")
    print(f"{'Boundary Condition':<15} | {'ω̄₁':>10}")
    print("-" * 32)
    for bc_name, omega_bar_1 in summary_results.items():
        print(f"{bc_name:<15} | {omega_bar_1:10.4f}")
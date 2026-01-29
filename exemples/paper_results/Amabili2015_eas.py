#Non-linearities in rotation and thickness deformation in a new
#third-order thickness deformation theory for static and dynamic
#analysis of isotropic and laminated doubly curved shells
#Amabili 2015

# Table 1

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig
import pyvista as pv

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    integral_x = 20
    integral_y = 20
    integral_z = 10

    R = 0.15
    L = 0.52
    h = 0.03
    E = 198E9
    nu = 0.3
    rho = 7800

    rectangular_domain = RectangularMidSurfaceDomain(0, L, 0, 2*np.pi)

    R_ = sym.Matrix([R * sym.cos(xi2_), R * sym.sin(xi2_), xi1_])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, rho)

    n_modos_c = 10
    n_modos = 10
    expansion_size = {"u1": (n_modos, n_modos_c),
                      "u2": (n_modos, n_modos_c),
                      "u3": (n_modos, n_modos_c),
                      "v1": (n_modos, n_modos_c),
                      "v2": (n_modos, n_modos_c),
                      "v3": (n_modos, n_modos_c)}

    boundary_conditions_u1 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}

    boundary_conditions_v1 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_v2 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_v3 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    eas_field = EasExpansion({"eas": (n_modos, n_modos_c)},
                               rectangular_domain,
                               {"eas": {"xi1": ("F", "F"), "xi2": ("R", "R")}})

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    M = mass_matrix(shell, integral_x, integral_y, integral_z)
    K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

    # Number of modes to be analyzed
    n_modes = 6

    # Solve generalized eigenvalue problem
    eigen_vals, eigen_vectors = eig(K, M)
    omega = np.sqrt(eigen_vals)

    # Keep only finite eigenvalues (remove NaN or Inf)
    finite_mask = np.isfinite(omega)

    tolerance = 1e-2
    real_part_non_zero_mask = np.abs(np.real(omega)) > tolerance

    final_mask = finite_mask & real_part_non_zero_mask
    omega = omega[final_mask]

    eigen_vectors = eigen_vectors[:, final_mask]

    # Sort eigenvalues in ascending order
    sorted_indices = np.argsort(omega.real)

    # Extract sorted finite eigenvalues and corresponding eigenvectors
    omega = omega[sorted_indices].real
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    freqHz = omega / (2.0 * np.pi)

    # Print the first five natural frequencies
    print("Frequencies (Hz):\n", freqHz[0:320:1])

    mode_output = 0


    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 40)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 80)
    xi3 = np.linspace(-h/2, h/2, 6)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    mode = shell.displacement_expansion(eigen_vectors[:, mode_output], x, y)

    M1, M2, M3 = shell.mid_surface_geometry.reciprocal_base(x, y)
    R = mid_surface_geometry(x, y)

    Rx3 = R[0, 0, :, :, None]
    Ry3 = R[1, 0, :, :, None]
    Rz3 = R[2, 0, :, :, None]

    M1x3 = M1[0][:, :, None]
    M1y3 = M1[1][:, :, None]
    M1z3 = M1[2][:, :, None]

    M2x3 = M2[0][:, :, None]
    M2y3 = M2[1][:, :, None]
    M2z3 = M2[2][:, :, None]

    M3x3 = M3[0, :, :, None]
    M3y3 = M3[1, :, :, None]
    M3z3 = M3[2, :, :, None]

    XI3 = xi3[None, None, :]

    X = Rx3 + XI3 * M3x3
    Y = Ry3 + XI3 * M3y3
    Z = Rz3 + XI3 * M3z3

    u1 = mode[0]
    u2 = mode[1]
    u3 = mode[2]

    v1 = mode[3]
    v2 = mode[4]
    v3 = mode[5]

    # Expandir para 3D
    u1_3 = u1[:, :, None]
    u2_3 = u2[:, :, None]
    u3_3 = u3[:, :, None]

    v1_3 = v1[:, :, None]
    v2_3 = v2[:, :, None]
    v3_3 = v3[:, :, None]

    U1 = (
            (u1_3 + XI3 * v1_3) * M1x3 +
            (u2_3 + XI3 * v2_3) * M2x3 +
            (u3_3 + XI3 * v3_3) * M3x3
    )

    U2 = (
            (u1_3 + XI3 * v1_3) * M1y3 +
            (u2_3 + XI3 * v2_3) * M2y3 +
            (u3_3 + XI3 * v3_3) * M3y3
    )

    U3 = (
            (u1_3 + XI3 * v1_3) * M1z3 +
            (u2_3 + XI3 * v2_3) * M2z3 +
            (u3_3 + XI3 * v3_3) * M3z3
    )

    U_mag = np.sqrt(U1 ** 2 + U2 ** 2 + U3 ** 2)
    U_max = np.max(U_mag)

    target = h  # 20% da espessura
    scale = target / (U_max + 1e-14)

    U1 *= scale
    U2 *= scale
    U3 *= scale

    X_def = X + U1
    Y_def = Y + U2
    Z_def = Z + U3

    grid = pv.StructuredGrid(X_def, Y_def, Z_def)

    grid["displacement"] = U_mag.ravel(order="F")

    plotter = pv.Plotter(window_size=(2000, 2000))
    plotter.add_mesh(
        grid,
        scalars="displacement",
        cmap="jet",
        opacity=1.0,
        lighting=False,
        show_edges=True,
        show_scalar_bar=False,
    )

    # Vista isométrica
    plotter.view_isometric()
    plotter.enable_parallel_projection()

    # Remover eixos, grade e qualquer anotação
    plotter.hide_axes()

    # Ajustar enquadramento automaticamente
    plotter.camera.zoom(1.1)

    # Salvar em PDF
    plotter.save_graphic("casca_isometrica.pdf")

    plotter.close()







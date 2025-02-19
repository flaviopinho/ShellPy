if __name__ == "__main__":
    from functionals_shells.koiter_shell_theory import *
    # from functionals_shells.koiter_shell_theory.koiter_shell_kinetic_energy import koiter_kinetic_energy
    from functionals_shells.koiter_shell_theory.koiter_shell_strain_energy import cubic_strain_energy, quadratic_strain_energy

    from multiprocessing import freeze_support
    from functionals_shells.displacement_expansion import GenericPolynomialSeries, simply_supported, \
        GenericTrigonometricSeries, \
        pinned

    from functionals_shells.boundary import RectangularBoundary

    from functionals_shells.material import LinearElasticMaterial

    from functionals_shells.shell import Shell

    from time import time

    import sympy as sym
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.linalg import eig

    from functionals_shells.thickness import ConstantThickness
    from functionals_shells.midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_

    freeze_support()  # Protege contra loops no Windows

    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001
    density = 7850

    E = 1
    E1 = 206E9
    nu = 0.3

    vibration_mode = 0

    boundary = RectangularBoundary(0, a, 0, b)

    expansion_size = {"u1": (5, 5),
                      "u2": (5, 5),
                      "u3": (5, 5)}

    boundary_conditions = pinned

    symmetry = {"xi1": True, "xi2": True}

    displacement_field = GenericTrigonometricSeries(expansion_size, boundary, boundary_conditions, symmetry)
    # displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, boundary, boundary_conditions, symmetry)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, boundary, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    start = time()
    T = koiter_kinetic_energy(shell)
    stop = time()
    print("tempo de cálculo do funcional de energia cinética: ", stop - start)

    start = time()
    U2p = quadratic_strain_energy(shell)
    stop = time()
    print("tempo de cálculo do funcional quadrático em paralelo: ", stop - start)

    # start = time()
    # U3 = cubic_strain_energy(shell)
    # stop = time()
    # print("tempo de cálculo do funcional cúbico: ", stop - start)

    # start = time()
    # U4 = quartic_strain_energy(shell)
    # stop = time()
    # print("tempo de cálculo do funcional quadrático: ", stop - start)

    M = T.jacobian().jacobian()(np.zeros((n_dof, 1)))
    K = U2p.jacobian().jacobian()(np.zeros((n_dof, 1))) * E1
    eigen_vals, eigen_vectors = eig(K, M, right=True, left=False)
    sorted_indices = np.argsort(eigen_vals.real)

    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    freq = np.sqrt(eigen_vals) / (2 * np.pi)
    mode1 = eigen_vectors[:, vibration_mode]

    print(freq)

    xi1 = np.linspace(*boundary.edges["xi1"], 100)
    xi2 = np.linspace(*boundary.edges["xi2"], 100)

    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    mode = shell.displacement_expansion(mode1, x, y)
    mode = mode / np.max(mode) * h * 3
    z = shell.mid_surface_geometry(x, y)

    ax = plt.axes(projection='3d')
    # Creating plot
    scmap = plt.cm.ScalarMappable(cmap='plasma')
    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2], facecolors=scmap.to_rgba(mode[2]))

    # show plot
    plt.axis('equal')
    plt.show()

import numpy as np
import sympy as sym

from shellpy import Shell, RectangularMidSurfaceDomain, ConstantThickness, MidSurfaceGeometry
from shellpy import xi1_, xi2_
from shellpy.utils.plot_undeformed_shell import plot_undeformed_shell


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Discretization and geometry parameters
    # --------------------------------------------------------------
    n1 = 100  # Radial points for plotting
    n2 = 1000  # Circumferential points for plotting
    n3 = 4  # Through-thickness points for plotting

    n = 10  # Number of circumferential waves
    p = 5  # Radial growth power exponent
    L = 1.0  # Radial length of the domain
    R_in = 0.3  # Inner radius

    H = 0.3  # Maximum corrugation amplitude

    # --------------------------------------------------------------
    # Shell properties
    # --------------------------------------------------------------
    thickness = ConstantThickness(L / 100)

    # --------------------------------------------------------------
    # Domain and mid-surface geometry definition
    # --------------------------------------------------------------
    rectangular_domain = RectangularMidSurfaceDomain(R_in, R_in + L, 0, 2 * np.pi)

    # Mid-surface position vector R(xi1, xi2)
    R_ = sym.Matrix([
        xi1_ * sym.cos(xi2_),
        xi1_ * sym.sin(xi2_),
        H * ((xi1_ - R_in) / L) ** p * sym.cos(n * xi2_)
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)

    # --------------------------------------------------------------
    # Shell object creation and plotting
    # --------------------------------------------------------------
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, None, None, None)

    print("Generating 3D plot. Saving to 'shell_plot_test.png'...")

    # Using the refined plot function with keyword arguments
    plot_undeformed_shell(
        shell=shell,
        file_name="shell_plot_test.png",
        n_1=n1,
        n_2=n2,
        n_3=n3,
        color=(0.55, 0.62, 0.70),
        wireframe_step_1=10,  # 1 a cada 10 pontos ao longo do raio L
        wireframe_step_2=50,  # 1 a cada 50 pontos ao longo da circunferência
        window_size=(1500, 1000),
        zoom=1.2
    )

    print("Done!")
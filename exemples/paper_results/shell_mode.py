import numpy as np
import pyvista as pv

from shellpy import Shell, RectangularMidSurfaceDomain


def shell_mode(
    shell: Shell,
    U,
    file_name,
    n_1,
    n_2,
    n_3,
    max_deformation=0,
):
    """
    Plot a 3D deformed shell mode shape using PyVista.

    Parameters
    ----------
    shell : Shell
        Shell object containing geometry, kinematics, and thickness definition.
    U : ndarray
        Eigenvector associated with the vibration mode.
    file_name : str
        Output PNG file name.
    n_1, n_2 : int
        Number of discretization points along the in-plane coordinates (xi1, xi2).
    n_3 : int
        Number of discretization points through the thickness.
    max_deformation : float, optional
        Maximum deformation amplitude used for visualization scaling.
    """

    # ------------------------------------------------------------------
    # Extract shell data
    # ------------------------------------------------------------------
    mid_surface_geometry = shell.mid_surface_geometry
    thickness = shell.thickness
    rectangular_domain = shell.mid_surface_domain

    # ------------------------------------------------------------------
    # In-plane discretization of the mid-surface
    # ------------------------------------------------------------------
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], n_1)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], n_2)
    x, y = np.meshgrid(xi1, xi2, indexing="ij")

    # ------------------------------------------------------------------
    # Through-thickness discretization
    # ------------------------------------------------------------------
    h = thickness(x, y)
    xi3 = np.linspace(-h / 2, h / 2, n_3)
    XI3 = np.transpose(xi3, (1, 2, 0))

    # ------------------------------------------------------------------
    # Evaluate displacement field of the mode
    # ------------------------------------------------------------------
    mode = shell.displacement_expansion(U, x, y)

    # ------------------------------------------------------------------
    # Geometry: position vector and reciprocal base vectors
    # ------------------------------------------------------------------
    M1, M2, M3 = mid_surface_geometry.reciprocal_base(x, y)
    R = mid_surface_geometry(x, y)

    # Mid-surface coordinates expanded to 3D
    Rx3 = R[0, 0, :, :, None]
    Ry3 = R[1, 0, :, :, None]
    Rz3 = R[2, 0, :, :, None]

    # Reciprocal base vectors expanded to 3D
    M1x3, M1y3, M1z3 = M1[0][:, :, None], M1[1][:, :, None], M1[2][:, :, None]
    M2x3, M2y3, M2z3 = M2[0][:, :, None], M2[1][:, :, None], M2[2][:, :, None]
    M3x3, M3y3, M3z3 = M3[0, :, :, None], M3[1, :, :, None], M3[2, :, :, None]

    # ------------------------------------------------------------------
    # Undeformed 3D configuration
    # ------------------------------------------------------------------
    X = Rx3 + XI3 * M3x3
    Y = Ry3 + XI3 * M3y3
    Z = Rz3 + XI3 * M3z3

    # ------------------------------------------------------------------
    # Kinematic variables (5- or 6-DOF shell theories)
    # ------------------------------------------------------------------
    n_dofs = mode.shape[0]

    u1, u2, u3 = mode[0], mode[1], mode[2]
    v1, v2 = mode[3], mode[4]

    if n_dofs == 6:
        v3 = mode[5]
    elif n_dofs == 5:
        # Reissiner-Mindlin type theory: no independent v3
        v3 = np.zeros_like(u1)
    else:
        raise ValueError(
            f"Incompatible number of kinematic parameters: {n_dofs}. "
            "Expected 5 or 6."
        )

    # ------------------------------------------------------------------
    # Expand kinematic fields to 3D
    # ------------------------------------------------------------------
    u1_3, u2_3, u3_3 = u1[:, :, None], u2[:, :, None], u3[:, :, None]
    v1_3, v2_3, v3_3 = v1[:, :, None], v2[:, :, None], v3[:, :, None]

    # ------------------------------------------------------------------
    # Displacement field in Cartesian coordinates
    # ------------------------------------------------------------------
    U1 = (
        (u1_3 + XI3 * v1_3) * M1x3
        + (u2_3 + XI3 * v2_3) * M2x3
        + (u3_3 + XI3 * v3_3) * M3x3
    )

    U2 = (
        (u1_3 + XI3 * v1_3) * M1y3
        + (u2_3 + XI3 * v2_3) * M2y3
        + (u3_3 + XI3 * v3_3) * M3y3
    )

    U3 = (
        (u1_3 + XI3 * v1_3) * M1z3
        + (u2_3 + XI3 * v2_3) * M2z3
        + (u3_3 + XI3 * v3_3) * M3z3
    )

    # ------------------------------------------------------------------
    # Deformation magnitude and scaling
    # ------------------------------------------------------------------
    U_mag = np.sqrt(U1**2 + U2**2 + U3**2)
    U_max = np.max(U_mag)

    scale = max_deformation / (U_max + 1e-14)

    U1 *= scale
    U2 *= scale
    U3 *= scale

    # ------------------------------------------------------------------
    # Deformed configuration
    # ------------------------------------------------------------------
    X_def = X + U1
    Y_def = Y + U2
    Z_def = Z + U3

    # ------------------------------------------------------------------
    # PyVista visualization
    # ------------------------------------------------------------------
    grid = pv.StructuredGrid(X_def, Y_def, Z_def)
    grid["displacement"] = U_mag.ravel(order="F")

    plotter = pv.Plotter(
        window_size=(3000, 2000),
        off_screen=True,
    )

    plotter.add_mesh(
        grid,
        scalars="displacement",
        cmap="turbo",
        opacity=1.0,
        lighting=False,
        show_edges=True,
        show_scalar_bar=False,
    )

    # Isometric view with parallel projection
    plotter.view_isometric()
    plotter.enable_parallel_projection()

    # Clean visualization
    plotter.hide_axes()
    plotter.camera.zoom(1.1)

    # Save figure
    plotter.screenshot(file_name)
    plotter.close()

import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors

from shellpy import Shell, RectangularMidSurfaceDomain
from shellpy.displacement_covariant_derivative import displacement_first_covariant_derivatives


def plot_deformed_shell(
        shell: Shell,
        U,
        file_name: str,
        n_1: int,
        n_2: int,
        n_3: int,
        max_deformation: float = 0,
        window_size: tuple = (1500, 1000),
        zoom: float = 1.1,
        wireframe_step_1: int = 5,
        wireframe_step_2: int = 5,
        wireframe_color: str = "black",
        wireframe_opacity: float = 1.0,
        edge_color: str = "black",
        image_scale: int = 4,
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
    window_size : tuple, optional
        Base resolution for the rendering window.
    zoom : float, optional
        Camera zoom level.
    wireframe_step_1, wireframe_step_2 : int, optional
        Step size for the wireframe lines along xi1 and xi2.
    wireframe_color, wireframe_opacity : str, float, optional
        Color and opacity for the wireframe.
    edge_color : str, optional
        Color of the outer boundaries.
    image_scale : int, optional
        Multiplier for super-resolution output.
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
    # Kinematic variables (3-, 5- or 6-DOF shell theories)
    # ------------------------------------------------------------------
    n_dofs = mode.shape[0]

    u1, u2, u3 = mode[0], mode[1], mode[2]

    if n_dofs == 6:
        v1, v2, v3 = mode[3], mode[4], mode[5]
    elif n_dofs == 5:
        # Reissner-Mindlin type theory: no independent v3
        v1, v2 = mode[3], mode[4]
        v3 = np.zeros_like(u1)
    elif n_dofs == 3:
        # Kirchhoff-Love type theory (3-parameter): rotations depend on mid-surface displacements
        u_field = mode[:3]  # Array com [u1, u2, u3]

        # Reconstrói a derivada total do deslocamento (du) iterando sobre as funções de forma
        # e realizando a combinação linear com o autovetor (U)
        n_dofs_total = shell.displacement_expansion.number_of_degrees_of_freedom()
        du = None
        for i in range(n_dofs_total):
            dn_i = shell.displacement_expansion.shape_function_first_derivatives(i, x, y)
            if du is None:
                du = U[i] * dn_i
            else:
                du += U[i] * dn_i

        # Calcula as derivadas covariantes
        dcu = displacement_first_covariant_derivatives(
            mid_surface_geometry, u_field, du, x, y
        )

        # Define as componentes rotacionais
        v1 = dcu[0]
        v2 = dcu[1]
        v3 = np.zeros_like(u1)
    else:
        raise ValueError(
            f"Incompatible number of kinematic parameters: {n_dofs}. "
            "Expected 3, 5, or 6."
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
    U_mag = np.sqrt(U1 ** 2 + U2 ** 2 + U3 ** 2)
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
    # PyVista Grid Setup
    # ------------------------------------------------------------------
    structured_grid = pv.StructuredGrid(X_def, Y_def, Z_def)
    structured_grid["displacement"] = U_mag.ravel(order="F")

    # Merge para otimizar detecção de bordas
    grid = structured_grid.merge(tolerance=1e-10)

    # ------------------------------------------------------------------
    # Criação do Wireframe na Configuração Deformada
    # ------------------------------------------------------------------
    points_list = []
    lines_list = []
    offset = 0

    idx_1 = list(range(0, n_1, wireframe_step_1))
    if idx_1[-1] != n_1 - 1: idx_1.append(n_1 - 1)

    idx_2 = list(range(0, n_2, wireframe_step_2))
    if idx_2[-1] != n_2 - 1: idx_2.append(n_2 - 1)

    for k in [0, n_3 - 1]:
        for j in idx_2:
            pts = np.column_stack((X_def[:, j, k], Y_def[:, j, k], Z_def[:, j, k]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

        for i in idx_1:
            pts = np.column_stack((X_def[i, :, k], Y_def[i, :, k], Z_def[i, :, k]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

    for i in idx_1:
        for j in idx_2:
            pts = np.column_stack((X_def[i, j, :], Y_def[i, j, :], Z_def[i, j, :]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

    high_res_wireframe = pv.PolyData(np.vstack(points_list), lines=np.hstack(lines_list))

    # ------------------------------------------------------------------
    # PyVista Renderização
    # ------------------------------------------------------------------
    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=True,
    )

    plotter.enable_anti_aliasing('msaa')
    plotter.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)

    colors = [
        (0, 0, 1),  # azul
        (0, 1, 1),  # ciano
        (0, 1, 0),  # verde
        (1, 1, 0),  # amarelo
        (1, 0, 0)  # vermelho
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("my_jet", colors, N=500)

    # Malha deformada (cores sem distorção por reflexo/sombreamento usando ambient=1)
    plotter.add_mesh(
        grid,
        scalars="displacement",
        cmap=cmap,
        opacity=1.0,
        lighting=True,
        ambient=1.0,
        diffuse=0.0,
        specular=0.0,
        show_edges=False,
        show_scalar_bar=False,
    )

    # Contorno externo na malha deformada
    edges = grid.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(
        edges,
        color=edge_color,
        render_lines_as_tubes=True,
        line_width=4,
    )

    # Grade de linhas do wireframe na malha deformada
    plotter.add_mesh(
        high_res_wireframe,
        color=wireframe_color,
        render_lines_as_tubes=True,
        line_width=2,
        opacity=wireframe_opacity,
    )

    # Configuração de câmera e visualização
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.hide_axes()
    plotter.camera.zoom(zoom)

    # Salva figura com a escala controlada
    plotter.screenshot(file_name, scale=image_scale)
    plotter.close()
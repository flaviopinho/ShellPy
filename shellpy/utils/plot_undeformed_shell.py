import numpy as np
import pyvista as pv

from shellpy import Shell, RectangularMidSurfaceDomain


def plot_undeformed_shell(
        shell: Shell,
        file_name: str,
        n_1: int,
        n_2: int,
        n_3: int,
        color: tuple = (0.55, 0.62, 0.70),
        wireframe_step_1: int = 10,
        wireframe_step_2: int = 50,
        wireframe_color: str = "black",
        wireframe_opacity: float = 1,
        edge_color: str = "black",
        window_size: tuple = (1500, 1000),  # Resolução base segura para VRAM
        zoom: float = 1.1,
):
    """
    Plots a 3D undeformed shell geometry using PyVista and saves it to a high-res image file.
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
    # Geometry: position vector and reciprocal base vectors
    # ------------------------------------------------------------------
    M1, M2, M3 = mid_surface_geometry.reciprocal_base(x, y)
    R = mid_surface_geometry(x, y)

    Rx3 = R[0, 0, :, :, None]
    Ry3 = R[1, 0, :, :, None]
    Rz3 = R[2, 0, :, :, None]

    M3x3 = M3[0, :, :, None]
    M3y3 = M3[1, :, :, None]
    M3z3 = M3[2, :, :, None]

    # ------------------------------------------------------------------
    # Undeformed 3D configuration
    # ------------------------------------------------------------------
    X = Rx3 + XI3 * M3x3
    Y = Ry3 + XI3 * M3y3
    Z = Rz3 + XI3 * M3z3

    # ------------------------------------------------------------------
    # PyVista visualization setup
    # ------------------------------------------------------------------
    structured_grid = pv.StructuredGrid(X, Y, Z)
    grid = structured_grid.merge(tolerance=1e-10)

    # Criação do Wireframe Curvo (incluindo as bordas naturalmente)
    points_list = []
    lines_list = []
    offset = 0

    idx_1 = list(range(0, n_1, wireframe_step_1))
    if idx_1[-1] != n_1 - 1: idx_1.append(n_1 - 1)

    idx_2 = list(range(0, n_2, wireframe_step_2))
    if idx_2[-1] != n_2 - 1: idx_2.append(n_2 - 1)

    for k in [0, n_3 - 1]:
        for j in idx_2:
            pts = np.column_stack((X[:, j, k], Y[:, j, k], Z[:, j, k]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

        for i in idx_1:
            pts = np.column_stack((X[i, :, k], Y[i, :, k], Z[i, :, k]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

    for i in idx_1:
        for j in idx_2:
            pts = np.column_stack((X[i, j, :], Y[i, j, :], Z[i, j, :]))
            points_list.append(pts)
            lines_list.append(np.hstack([len(pts), np.arange(offset, offset + len(pts))]))
            offset += len(pts)

    high_res_wireframe = pv.PolyData(np.vstack(points_list), lines=np.hstack(lines_list))

    # ------------------------------------------------------------------
    # Renderização no Plotter
    # ------------------------------------------------------------------
    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=True,
    )

    # MSAA: Suaviza as linhas perfeitamente sem estourar a memória de vídeo
    plotter.enable_anti_aliasing('msaa')
    plotter.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)

    # Superfície lisa
    plotter.add_mesh(
        grid,
        color=color,
        opacity=1.0,
        lighting=True,
        ambient=0.35,
        diffuse=0.6,
        specular=0.05,
        show_edges=False,
    )

    # Contorno externo
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
        line_width=4, # Contorno externo levemente mais forte
    )

    # Grade de linhas brancas
    plotter.add_mesh(
        high_res_wireframe,
        color=wireframe_color,
        render_lines_as_tubes=True,
        line_width=2,
        opacity=wireframe_opacity,
    )

    # ------------------------------------------------------------------
    # Finalização
    # ------------------------------------------------------------------
    plotter.view_isometric()
    plotter.enable_parallel_projection()

    plotter.hide_axes()
    plotter.camera.zoom(zoom)

    # Salva com super-resolução (Scale 4 = 6000x4000 pixels baseados na janela)
    plotter.screenshot(file_name, scale=4)
    plotter.close()
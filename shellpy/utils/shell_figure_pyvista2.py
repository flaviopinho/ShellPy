import sympy as sym
import numpy as np
import pyvista as pv

from shellpy import RectangularMidSurfaceDomain, ConstantThickness
from shellpy import xi1_, xi2_, MidSurfaceGeometry


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    # ----------------------------
    # Parâmetros numéricos
    # ----------------------------
    n_int_x = 400
    n_int_y = 400
    n_int_z = 2

    aRx = 1
    aRy = -1
    p = 4
    ah = 10

    a = 1
    b = 1
    Rx = a / aRx
    Ry = a / aRy

    h = a / ah

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([
        xi1_,  # x
        xi2_,  # y
        # 0  # plate
        # 1 / (2 * Rx) * (xi1_ - a / 2) ** 2 # cylindrical
        1 / (2 * Rx) * (xi1_ - a / 2) ** 2 + 1 / (2 * Ry) * (xi2_ - b / 2) ** 2  # z # doubly curved
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    # ----------------------------
    # Malha paramétrica (superfícies principais)
    # ----------------------------
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], n_int_x)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], n_int_y)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    R = mid_surface_geometry(x, y)
    _, _, M3 = mid_surface_geometry.natural_base(x, y)
    h = thickness(x, y)

    # Superfícies interna e externa
    R_interno = R.squeeze() + np.einsum('axy,xy->axy', M3, -h / 2)
    R_externo = R.squeeze() + np.einsum('axy,xy->axy', M3,  h / 2)

    grid_in = pv.StructuredGrid(R_interno[0], R_interno[1], R_interno[2])
    grid_out = pv.StructuredGrid(R_externo[0], R_externo[1], R_externo[2])

    # ----------------------------
    # Superfícies laterais (xi1 = const)
    # ----------------------------
    def lateral_grid_x(x_const):
        y = xi2
        R = mid_surface_geometry(x_const, y).squeeze()
        _, _, M3 = mid_surface_geometry.reciprocal_base(x_const, y)
        h = thickness(x_const, y)
        z = np.linspace(-h / 2, h / 2, n_int_z)

        Rx = (
                np.einsum('ax,z->axz', R, z ** 0)
                + np.einsum('ax,z->axz', M3, z)
        )

        return pv.StructuredGrid(Rx[0], Rx[1], Rx[2])


    def lateral_grid_y(y_const):
        x = xi1
        R = mid_surface_geometry(x, y_const).squeeze()
        _, _, M3 = mid_surface_geometry.reciprocal_base(x, y_const)
        h = thickness(x, y_const)
        z = np.linspace(-h / 2, h / 2, n_int_z)

        Rx = (
                np.einsum('ax,zx->axz', R, z ** 0)
                + np.einsum('ax,zx->axz', M3, z)
        )

        return pv.StructuredGrid(Rx[0], Rx[1], Rx[2])


    grid_x1 = lateral_grid_x(rectangular_domain.edges["xi1"][0])
    grid_x2 = lateral_grid_x(rectangular_domain.edges["xi1"][1])

    grid_y1 = lateral_grid_y(rectangular_domain.edges["xi2"][0])
    grid_y2 = lateral_grid_y(rectangular_domain.edges["xi2"][1])

    # ----------------------------
    # Bordas (extraídas das superfícies laterais)
    # ----------------------------
    edges_x1 = grid_x1.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )

    edges_x2 = grid_x2.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )

    edges_y1 = grid_y1.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )

    edges_y2 = grid_y2.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False
    )

    # ----------------------------
    # Visualização
    # ----------------------------
    plotter = pv.Plotter(off_screen=True, window_size=(6000, 4000))
    plotter.enable_anti_aliasing()
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    plotter.enable_parallel_projection()

    shell_color = "lightgray"

    plotter.add_mesh(grid_in, opacity=0.7,  color=shell_color, smooth_shading=True)
    plotter.add_mesh(grid_out, opacity=0.7, color=shell_color, smooth_shading=True)

    plotter.add_mesh(grid_x1, opacity=0.7, color=shell_color, smooth_shading=True)
    plotter.add_mesh(grid_x2, opacity=0.7, color=shell_color, smooth_shading=True)

    plotter.add_mesh(grid_y1, opacity=0.7, color=shell_color, smooth_shading=True)
    plotter.add_mesh(grid_y2, opacity=0.7, color=shell_color, smooth_shading=True)

    plotter.add_mesh(edges_x1, color="black", line_width=24)
    plotter.add_mesh(edges_x2, color="black", line_width=24)

    plotter.add_mesh(edges_y1, color="black", line_width=24)
    plotter.add_mesh(edges_y2, color="black", line_width=24)

    plotter.show_axes()
    plotter.show(screenshot="shell4.png")

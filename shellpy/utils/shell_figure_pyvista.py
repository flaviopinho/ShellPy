import sympy as sym
import numpy as np
import pyvista as pv

from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    # ----------------------------
    # Parâmetros numéricos
    # ----------------------------
    n_int_x = 200
    n_int_y = 400
    n_int_z = 2

    factor = 0.5
    alpha = np.deg2rad(30)
    R2 = 0.4
    L = R2 * factor / np.sin(alpha)
    L2 = R2 / np.sin(alpha)
    L1 = L2 - L

    H = 0.05 * R2

    # ----------------------------
    # Espessura variável
    # ----------------------------
    """
    def thickness(xi1, xi2):
        s1 = 0.5
        s2 = 1.0
        qi0 = 0.5
        cutoff = L1 + qi0 * L
        return np.where(xi1 <= cutoff, s1 * H, s2 * H)

    """
    def thickness(xi1, xi2):
        p = 1
        q = 1
        return H * (p + q * xi1)


    # ----------------------------
    # Domínio e geometria
    # ----------------------------
    rectangular_domain = RectangularMidSurfaceDomain(L1, L2, 0+3*np.pi/2, 2 * np.pi+3*np.pi/2)

    R_ = sym.Matrix([
        xi1_ * sym.sin(alpha) * sym.cos(xi2_),
        xi1_ * sym.sin(alpha) * sym.sin(xi2_),
        xi1_ * sym.cos(alpha)
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)

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
            np.einsum('ax,z->axz', R, z**0)
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
            np.einsum('ax,z->axz', R, z**0)
            + np.einsum('ax,z->axz', M3, z)
        )

        return pv.StructuredGrid(Rx[0], Rx[1], Rx[2])

    grid_x1 = lateral_grid_x(rectangular_domain.edges["xi1"][0])
    grid_x2 = lateral_grid_x(rectangular_domain.edges["xi1"][1])

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

    # ----------------------------
    # Visualização
    # ----------------------------
    plotter = pv.Plotter(off_screen=True, image_scale=3)
    plotter.enable_anti_aliasing()
    plotter.set_background("white")
    plotter.enable_depth_peeling()

    shell_color = "lightgray"

    plotter.add_mesh(grid_in, opacity=0.7,  color=shell_color, smooth_shading=True)
    plotter.add_mesh(grid_out, opacity=0.7, color=shell_color, smooth_shading=True)

    plotter.add_mesh(grid_x1, opacity=0.7, color=shell_color, smooth_shading=True)
    plotter.add_mesh(grid_x2, opacity=0.7, color=shell_color, smooth_shading=True)

    plotter.add_mesh(edges_x1, color="black", line_width=3)
    plotter.add_mesh(edges_x2, color="black", line_width=3)

    actor = plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        xtitle=' ',
        ytitle=' ',
        ztitle=' ',
        use_3d_text=False,
        bold=False
    )
    plotter.show(auto_close=False)
    plotter.screenshot("shell2a.png")
    plotter.close()

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry

def plot_surface_edges(ax, R, color='k', lw=1.0):
    # xi1 = const (início e fim)
    ax.plot(R[0][0, :],  R[1][0, :],  R[2][0, :],  color=color, lw=lw)
    ax.plot(R[0][-1, :], R[1][-1, :], R[2][-1, :], color=color, lw=lw)

    # xi2 = const (início e fim)
    ax.plot(R[0][:, 0],  R[1][:, 0],  R[2][:, 0],  color=color, lw=lw)
    ax.plot(R[0][:, -1], R[1][:, -1], R[2][:, -1], color=color, lw=lw)


# Main execution block
if __name__ == "__main__":
    n_int_x = 200
    n_int_y = 400
    n_int_z = 2

    factor = 0.5
    alpha = np.deg2rad(30)
    R2 = 0.4
    L = R2 * factor / np.sin(alpha)
    L2 = R2 / np.sin(alpha)
    L1 = L2 - L
    R1 = L1 * np.sin(alpha)

    H = 0.05 * R2


    def thickness(xi1, xi2):
        s1 = 0.5
        s2 = 1.0
        qi0 = 0.5

        cutoff = L1 + qi0 * L

        # np.where works elementwise, so it handles 2D arrays naturally
        h = np.where(xi1 <= cutoff, s1 * H, s2 * H)

        return h


    rectangular_domain = RectangularMidSurfaceDomain(L1, L2, 0, 2 * np.pi)

    R_ = sym.Matrix([
        xi1_ * sym.sin(alpha) * sym.cos(xi2_),  # x
        xi1_ * sym.sin(alpha) * sym.sin(xi2_),  # y
        xi1_ * sym.cos(alpha)  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    xi1 = np.linspace(*rectangular_domain.edges["xi1"], n_int_x)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], n_int_y)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    R = mid_surface_geometry(x, y)
    M1, M2, M3 = mid_surface_geometry.natural_base(x, y)

    h = thickness(x, y)

    R_interno = R.squeeze() + np.einsum('axy, xy->axy', M3, -h / 2)
    R_externo = R.squeeze() + np.einsum('axy, xy->axy', M3, h / 2)

    x, y = rectangular_domain.edges["xi1"][0], xi2
    R = mid_surface_geometry(x, y).squeeze()
    _, _, M3 = mid_surface_geometry.reciprocal_base(x, y)
    h = thickness(x, y)
    z = np.linspace(-h / 2, h / 2, n_int_z)
    Rx1 = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3, z)
    Rx1_bordas = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3*1.1, z)

    x, y = rectangular_domain.edges["xi1"][1], xi2
    R = mid_surface_geometry(x, y).squeeze()
    _, _, M3 = mid_surface_geometry.reciprocal_base(x, y)
    h = thickness(x, y)
    z = np.linspace(-h / 2, h / 2, n_int_z)
    Rx2 = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3, z)
    Rx2_bordas = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3 * 1.1, z)

    x, y = xi1, rectangular_domain.edges["xi2"][0]
    R = mid_surface_geometry(x, y).squeeze()
    _, _, M3 = mid_surface_geometry.reciprocal_base(x, y)
    h = thickness(x, y)
    z = np.linspace(-h / 2, h / 2, n_int_z)
    #Ry1 = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3, z)
    #Ry1_bordas = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3 * 1.001, z)

    x, y = xi1, rectangular_domain.edges["xi2"][1]
    R = mid_surface_geometry(x, y).squeeze()
    _, _, M3 = mid_surface_geometry.reciprocal_base(x, y)
    h = thickness(x, y)
    z = np.linspace(-h / 2, h / 2, n_int_z)
    #Ry2 = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3, z)
    #Ry2_bordas = np.einsum('ax, z->axz', R, z ** 0) + np.einsum('ax, z->axz', M3 * 1.001, z)

    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure()
    ax = plt.axes(projection='3d',computed_zorder=True)

    # Remove grid
    ax.grid(False)

    plt.rcParams['text.usetex'] = True

    ax.set_xlabel(r'$\mathdefault{x^{1}}$', fontsize=10)
    ax.set_ylabel(r'$\mathdefault{x^{2}}$', fontsize=10)
    ax.set_zlabel(r'$\mathdefault{x^{3}}$', fontsize=10)

    # Superfície inferior
    ax.plot_surface(
        R_interno[0], R_interno[1], R_interno[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True
    )

    # Superfície superior
    ax.plot_surface(
        R_externo[0], R_externo[1], R_externo[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True
    )

    # Superfície lateral 1
    ax.plot_surface(
        Rx1[0], Rx1[1], Rx1[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True
    )

    # Superfície lateral 2
    ax.plot_surface(
        Rx2[0], Rx2[1], Rx2[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True
    )
    """
    # Superfície lateral 3
    ax.plot_surface(
        Ry1[0], Ry1[1], Ry1[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True,
        alpha=0.7
    )

    # Superfície lateral 4
    ax.plot_surface(
        Ry2[0], Ry2[1], Ry2[2],
        color=(0.9, 0.9, 0.9),
        edgecolor='none',
        antialiased=False,
        rstride=1,
        cstride=1,
        shade=True,
        alpha=0.7
    )
    """

    plot_surface_edges(ax, Rx1_bordas, color='k', lw=2)
    plot_surface_edges(ax, Rx2_bordas, color='k', lw=2)

    # Proporção correta
    ax.set_box_aspect([
        np.ptp(R_externo[0]),
        np.ptp(R_externo[1]),
        np.ptp(R_externo[2])
    ])

    plt.savefig("shell.png", dpi=900)

    plt.tight_layout()
    plt.show()

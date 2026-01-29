import numpy as np
import pyvista as pv

# Geometria
R = 1.0        # raio médio
h = 0.2        # espessura
L = 5.0        # comprimento

# Discretização
nr = 6         # pontos na espessura
nt = 80        # pontos circunferenciais
nz = 50        # pontos axiais

# Deformação
alpha = 0.15

# Coordenadas paramétricas
r = np.linspace(R - h/2, R + h/2, nr)
theta = np.linspace(0, 2*np.pi, nt)
z = np.linspace(-L/2, L/2, nz)

# Malha 3D paramétrica
Rr, Tt, Zz = np.meshgrid(r, theta, z, indexing="ij")

# Coordenadas cartesianas
X = Rr * np.cos(Tt)
Y = Rr * np.sin(Tt)
Z = Zz

grid = pv.StructuredGrid(X, Y, Z)

campo = (Rr - R) * np.sin(2*Tt) * np.cos(np.pi * Zz / L)

grid["campo"] = campo.ravel(order="F")

ur = alpha * np.sin(2*Tt) * np.sin(np.pi * Zz / L)

ux = ur * np.cos(Tt)
uy = ur * np.sin(Tt)
uz = np.zeros_like(ux)

grid_def = grid.copy()
grid_def.points[:, 0] += ux.ravel(order="F")
grid_def.points[:, 1] += uy.ravel(order="F")
grid_def.points[:, 2] += uz.ravel(order="F")

plotter = pv.Plotter()
plotter.add_mesh(
    grid_def,
    scalars="campo",
    cmap="viridis",
    opacity=1.0,
    show_edges=False
)

plotter.add_axes()
plotter.show()


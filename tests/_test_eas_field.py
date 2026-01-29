import numpy as np
import matplotlib.pyplot as plt
from shellpy import RectangularMidSurfaceDomain
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion

# -----------------------------
# Domínio da placa
# -----------------------------
a = 2*np.pi
b = 1.0

domain = RectangularMidSurfaceDomain(0, a, 0, b)

# -----------------------------
# Expansão
# -----------------------------
expansion_size = {
    "eas": (10, 10)
}

boundary_conditions = {
    "eas": {"xi1": ("R", "R"), "xi2": ("S", "S")}
}

expansion = EasExpansion(
    expansion_size,
    domain,
    boundary_conditions
)

# -----------------------------
# Malha
# -----------------------------
nx, ny = 100, 100
xi1 = np.linspace(0, a, nx)
xi2 = np.linspace(0, b, ny)
XI1, XI2 = np.meshgrid(xi1, xi2)

# -----------------------------
# Vetor de coeficientes
# -----------------------------
ndof = expansion.number_of_degrees_of_freedom()
U = np.random.rand(ndof)

# -----------------------------
# Avaliação do campo
# -----------------------------
u3 = np.zeros_like(XI1)

for n in range(ndof):
    u = expansion.shape_function(n, XI1, XI2)
    u3 += u * U[n]

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    XI1, XI2, u3,
    cmap="viridis",
    linewidth=0,
    antialiased=True
)

fig.colorbar(surf, ax=ax, shrink=0.6, label=r"$u_3$")

ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
ax.set_zlabel(r"$u_3$")
ax.set_title("Placa – superfície 3D (Legendre)")

plt.tight_layout()
plt.show()

erro_periodicidade = np.max(
    np.abs(u3[:, 0] - u3[:, -1])
)

print("Erro máximo de periodicidade:", erro_periodicidade)

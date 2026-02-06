import numpy as np
import matplotlib.pyplot as plt
from shellpy import RectangularMidSurfaceDomain
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion

# -----------------------------
# Domínio
# -----------------------------
a = 2 * np.pi
b = 1.0

domain = RectangularMidSurfaceDomain(0, a, 0, b)

expasion_size=(10, 3)

# -----------------------------
# Expansão EAS
# -----------------------------
expansion_eas = EasExpansion(
    {"eas": expasion_size},
    domain,
    {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
)

# -----------------------------
# Expansão Legendre
# -----------------------------
expansion_leg = LegendreSeries(
    {"u1": expasion_size},
    domain,
    {"u1": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
)

# -----------------------------
# Malha
# -----------------------------
nx, ny = 100, 100
xi1 = np.linspace(0, a, nx)
xi2 = np.linspace(0, b, ny)
XI1, XI2 = np.meshgrid(xi1, xi2)

# -----------------------------
# Coeficientes
# -----------------------------
U = np.random.rand(expansion_eas.number_of_degrees_of_freedom())

print(expansion_leg.number_of_degrees_of_freedom())
print(expansion_eas.number_of_degrees_of_freedom())

# -----------------------------
# Avaliação dos campos
# -----------------------------
u_eas = np.zeros_like(XI1)
u_leg = np.zeros_like(XI1)

for n in range(len(U)):
    u_eas += expansion_eas.shape_function(n, XI1, XI2) * U[n]

for n in range(len(U)):
    u_leg += expansion_leg.shape_function(n, XI1, XI2)[0] * U[n]

print(np.max(np.abs(u_leg - u_eas)))

# -----------------------------
# Plot comparativo (mesmo eixo)
# -----------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

surf_eas = ax.plot_surface(
    XI1, XI2, u_eas,
    cmap="viridis",
    alpha=0.7,
    linewidth=0
)

surf_leg = ax.plot_surface(
    XI1, XI2, u_leg,
    cmap="plasma",
    alpha=0.7,
    linewidth=0
)

ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
ax.set_zlabel(r"$u$")
ax.set_title("Comparação: EAS (viridis) × Legendre (plasma)")

plt.tight_layout()
plt.show()


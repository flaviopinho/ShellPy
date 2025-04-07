import numpy as np
import matplotlib.pyplot as plt
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial, power_law_distribution1

E0 = 1
E1 = 2
nu0 = 0.2
nu1 = 0.3
density0 = 3
density1 = 3

h = 0.01
a = 1
b = 0.5
c = 2
p = 2

func = lambda z: power_law_distribution1(z, h, a, b, c, p)

material = FunctionallyGradedMaterial(E0, E1, nu0, nu1, density0, density1, func)

z = np.linspace(-h/2, h/2, 100)

Vc = func(z)
E = material.E(z)

# Plotando Vc e E
plt.figure(figsize=(8, 5))
plt.plot(z/h, Vc, label=r'$V_c(z)$')
plt.plot(z/h, E, label=r'$E(z)$', linestyle='dashed')
plt.xlabel(r'$z/h$')
plt.ylabel(r'$Valores$')
plt.title(r'Propriedades em função de $z/h$')
plt.legend()
plt.grid()
plt.show()



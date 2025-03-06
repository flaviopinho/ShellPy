import numpy as np
from matplotlib import pyplot as plt

from shellpy.expansions.simple_expansions import fourier_expansion_for_periodic_solutions

edges = (0, 3)
n_modos = 5

f = fourier_expansion_for_periodic_solutions(edges, 3, n_modos)

x = np.linspace(edges[0], edges[1], 200)

plt.figure(figsize=(8, 6))

for i in range(n_modos):  # De 0 a 5
    fi = f[(i, 0)](x)
    plt.plot(x, fi, label=f"i = {i+1}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title(f'Funções f[(i, 0)] para i de 1 a {n_modos}')
plt.show()
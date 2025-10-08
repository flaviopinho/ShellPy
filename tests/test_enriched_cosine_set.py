import numpy as np
import matplotlib.pyplot as plt

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion

if __name__ == "__main__":
    a = 2
    boundary_conditions = ("F", "F")
    boundary = (0, a)
    n_modos = 20

    f = EnrichedCosineExpansion._generate_set_C1(boundary_conditions, boundary, 3, n_modos+1)

    x = np.linspace(boundary[0], boundary[1], 200)

    plt.figure(figsize=(8, 6))

    for i in range(n_modos):  # De 0 a 5
        fi = f[(i, 0)](x)
        plt.plot(x, fi, label=f"i = {i+1}")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title(f'Funções f[(i, 0)] para i de 1 a {n_modos}')
    plt.show()




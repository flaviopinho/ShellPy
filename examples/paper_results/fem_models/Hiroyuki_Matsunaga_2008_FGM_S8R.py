# Example based on the shell with functionally graded material from
# the article: https://doi:10.1016/j.compstruct.2007.07.006
# Natural frequency reported is in Table 2 from various types of shells


import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from examples.paper_results.fem_models.generate_boundary_conditions import generate_bc_lines
from examples.paper_results.fem_models.generate_boundary_transformation import generate_boundary_transformation
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, simply_supported, MidSurfaceGeometry

if __name__ == "__main__":
    # Define boundary conditions
    boundary_conditions = simply_supported

    # === PARÂMETROS ===

    aRx = 0
    aRy = 0
    p = 4
    ah = 2

    a = 1
    b = 1
    Rx = 0 # a / aRx
    Ry = 0 # a / aRy

    h = a / ah

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([
        xi1_,  # x
        xi2_,  # y
        0 # 1 / (2 * Rx) * (xi1_ - a / 2) ** 2 #+ 1 / (2 * Ry) * (xi2_ - b / 2) ** 2  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    u1, u2 = rectangular_domain.edges["xi1"]
    v1, v2 = rectangular_domain.edges["xi2"]
    n_u, n_v = 20, 20

    periodic_u = False
    periodic_v = False

    # === PROPRIEDADES DO MATERIAL ===

    E_M = 70E9
    nu_M = 0.3
    rho_M = 2710

    E_C = 380E9
    nu_C = 0.3
    rho_C = 3800

    Vc = lambda z: (0.5+z/h)**p
    E = lambda z: (E_C - E_M) * Vc(z) + E_M
    nu = lambda z: (nu_C - nu_M) * Vc(z) + nu_M
    rho = lambda z: (rho_C - rho_M) * Vc(z) + rho_M

    number_of_layers = 50

    # === GERAÇÃO DA MALHA ===
    num_u_nodes = n_u * 2 if periodic_u else n_u * 2 + 1
    num_v_nodes = n_v * 2 if periodic_v else n_v * 2 + 1

    u_vals = np.linspace(u1, u2, num_u_nodes, endpoint=not periodic_u)
    v_vals = np.linspace(v1, v2, num_v_nodes, endpoint=not periodic_v)

    nodes = []
    node_ids = {}
    node_id = 1

    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            x, y, z = mid_surface_geometry(u, v).flatten()
            MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)

            if j % 2 != 0 and i % 2 != 0:
                node_ids[(i, j)] = -1
            else:
                nodes.append((node_id, x, y, z, MR1, MR2, MR3))
                node_ids[(i, j)] = node_id
                node_id += 1

    elements = []
    elem_id = 1

    # Define limites do laço dependendo da periodicidade
    iu_max = num_u_nodes if periodic_u else num_u_nodes - 1
    iv_max = num_v_nodes if periodic_v else num_v_nodes - 1

    for j in range(0, iv_max, 2):
        for i in range(0, iu_max, 2):
            i_next = (i + 2)
            if i_next >= num_u_nodes:
                i_next = 0

            j_next = (j + 2)
            if j_next >= num_v_nodes:
                j_next = 0

            n1 = node_ids[(i, j)]
            n2 = node_ids[(i_next, j)]
            n3 = node_ids[(i_next, j_next)]
            n4 = node_ids[(i, j_next)]
            n5 = node_ids[(i+1, j)]
            n6 = node_ids[(i_next, j+1)]
            n7 = node_ids[(i+1, j_next)]
            n8 = node_ids[(i, j+1)]
            elements.append((elem_id, n1, n2, n3, n4, n5, n6, n7, n8))
            elem_id += 1

    # === VISUALIZAÇÃO 3D ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem in elements:
        _, n1, n2, n3, n4, n5, n6, n7, n8 = elem
        coords = np.array([
            nodes[n1 - 1][1:4], nodes[n2 - 1][1:4], nodes[n3 - 1][1:4], nodes[n4 - 1][1:4], nodes[n1 - 1][1:4]
        ])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)

    base = f"Hiroyuki_Matsunaga_2008_FGM_S8R_{ah:.2f}_{aRx:.2f}_{aRy:.2f}_{p:.2f}"
    safe_base = base.replace('.', 'p')
    filename = f"{safe_base}.inp"

    # === ESCRITA DO ARQUIVO ABAQUS .inp ===
    with open(filename, "w") as f:
        f.write("*HEADING\n")
        f.write("Modelo de Casca Gerado Automaticamente\n")
        f.write("Sistema de Unidades: SI (m, kg, s)\n")

        # Nós
        f.write("*NODE, NSET=ALLNODES\n")
        for node in nodes:
            if node[0] != -1:
                f.write(f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}, ")
                #f.write(f"{node[4][0]:.6f}, {node[4][1]:.6f}, {node[4][2]:.6f}, ")
                #f.write(f"{node[5][0]:.6f}, {node[5][1]:.6f}, {node[5][2]:.6f}, ")
                #f.write(f"{node[6][0]:.6f}, {node[6][1]:.6f}, {node[6][2]:.6f}")
                f.write("\n")

        nx = u_vals.size
        ny = v_vals.size

        f.write("*NSET, NSET=X0\n")
        for i in range(ny):
            if i % 4 == 0 and i != 0:
                f.write("\n")
            f.write(f"{node_ids[(0, i)]}, ")
        f.write("\n")
        f.write("*NSET, NSET=X1\n")
        for i in range(ny):
            if i % 4 == 0 and i != 0:
                f.write("\n")
            f.write(f"{node_ids[(nx-1, i)]}, ")
        f.write("\n")

        f.write("*NSET, NSET=Y0\n")
        for i in range(nx):
            if i % 4 == 0 and i != 0:
                f.write("\n")
            f.write(f"{node_ids[(i, 0)]}, ")
        f.write("\n")
        f.write("*NSET, NSET=Y1\n")
        for i in range(nx):
            if i % 4 == 0 and i != 0:
                f.write("\n")
            f.write(f"{node_ids[(i, ny-1)]}, ")
        f.write("\n")


        # Elementos
        f.write("*ELEMENT, TYPE=S8R, ELSET=ALLELEMENTS\n")
        for elem in elements:
            f.write(f"{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}, {elem[8]}\n")

        # Propriedades do Material
        for l in range(number_of_layers):
            z = -h / 2 + h / number_of_layers * (l + 1 / 2)
            print(z)
            f.write(f"*MATERIAL, NAME=MATERIAL_ELASTICO_{l}\n")
            f.write("*ELASTIC\n")
            f.write(f"{E(z)}, {nu(z)}\n")
            f.write("*DENSITY\n")
            f.write(f"{rho(z)}\n")

        # Seção da Casca
        f.write(f"*SHELL SECTION, ELSET=ALLELEMENTS, COMPOSITE\n")
        for l in range(number_of_layers):
            f.write(f"{h/number_of_layers}, 3, MATERIAL_ELASTICO_{l}, 0\n")

        generate_boundary_transformation(node_ids, u_vals, v_vals, nx, ny, mid_surface_geometry, f)

        lines = generate_bc_lines(boundary_conditions)
        for line in lines:
            f.write(line + "\n")

        # Passo de análise
        f.write("*STEP\n")
        f.write("*FREQUENCY\n")
        f.write("10, 100, 100000\n")
        f.write("*END STEP\n")

    ax.set_title("Visualização da malha 3D")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.show()

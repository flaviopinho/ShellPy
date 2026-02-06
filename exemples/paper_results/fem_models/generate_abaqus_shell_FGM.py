import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from exemples.paper_results.fem_models.generate_boundary_conditions import generate_bc_lines
from exemples.paper_results.fem_models.generate_boundary_transformation import generate_boundary_transformation
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, simply_supported, MidSurfaceGeometry

if __name__ == "__main__":
    # Define boundary conditions
    boundary_conditions = simply_supported

    # === PARÂMETROS ===

    a = 1
    b = 1
    Rx = a / 2
    Ry = -a / 2

    h = a / 5

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([
        xi1_,  # x
        xi2_,  # y
        1 / (2 * Rx) * (xi1_ - a / 2) ** 2 + 1 / (2 * Ry) * (xi2_ - b / 2) ** 2  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)


    u1, u2 = rectangular_domain.edges["xi1"]
    v1, v2 = rectangular_domain.edges["xi2"]
    n_u, n_v = 100, 100

    periodic_u = False
    periodic_v = False

    # === PROPRIEDADES DO MATERIAL ===

    E_M = 70E9
    nu_M = 0.3
    rho_M = 2710

    E_C = 380E9
    nu_C = 0.3
    rho_C = 3800

    p = 4
    Vc = lambda z: (0.5+z/h)**p
    E = lambda z: (E_C - E_M) * Vc(z) + E_M
    nu = lambda z: (nu_C - nu_M) * Vc(z) + nu_M
    rho = lambda z: (rho_C - rho_M) * Vc(z) + rho_M

    number_of_layers = 20

    # === GERAÇÃO DA MALHA ===
    num_u_nodes = n_u if periodic_u else n_u + 1
    num_v_nodes = n_v if periodic_v else n_v + 1

    u_vals = np.linspace(u1, u2, num_u_nodes, endpoint=not periodic_u)
    v_vals = np.linspace(v1, v2, num_v_nodes, endpoint=not periodic_v)

    nodes = []
    node_ids = {}
    node_id = 1

    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            x, y, z = mid_surface_geometry(u, v).flatten()
            MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)

            nodes.append((node_id, x, y, z, MR1, MR2, MR3))
            node_ids[(i, j)] = node_id
            node_id += 1

    elements = []
    elem_id = 1

    # Define limites do laço dependendo da periodicidade
    iu_max = num_u_nodes if periodic_u else num_u_nodes - 1
    iv_max = num_v_nodes if periodic_v else num_v_nodes - 1

    for j in range(iv_max):
        for i in range(iu_max):
            i_next = (i + 1)
            if i_next >= num_u_nodes:
                i_next = 0

            j_next = (j + 1)
            if j_next >= num_v_nodes:
                j_next = 0

            n1 = node_ids[(i, j)]
            n2 = node_ids[(i_next, j)]
            n3 = node_ids[(i_next, j_next)]
            n4 = node_ids[(i, j_next)]
            elements.append((elem_id, n1, n2, n3, n4))
            elem_id += 1

    # === VISUALIZAÇÃO 3D ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem in elements:
        _, n1, n2, n3, n4 = elem
        coords = np.array([
            nodes[n1 - 1][1:4], nodes[n2 - 1][1:4], nodes[n3 - 1][1:4], nodes[n4 - 1][1:4], nodes[n1 - 1][1:4]
        ])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)

    # === ESCRITA DO ARQUIVO ABAQUS .inp ===
    with open("malha_abaqus_casca_FGM.inp", "w") as f:
        f.write("*HEADING\n")
        f.write("Modelo de Casca Gerado Automaticamente\n")
        f.write("Sistema de Unidades: SI (m, kg, s)\n")

        # Nós
        f.write("*NODE, NSET=ALLNODES\n")
        for node in nodes:
            f.write(f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}, ")
            f.write(f"{node[4][0]:.6f}, {node[4][1]:.6f}, {node[4][2]:.6f}, ")
            f.write(f"{node[5][0]:.6f}, {node[5][1]:.6f}, {node[5][2]:.6f}, ")
            f.write(f"{node[6][0]:.6f}, {node[6][1]:.6f}, {node[6][2]:.6f}\n")

        nx = u_vals.size
        ny = v_vals.size

        f.write("*NSET, NSET=X0, GENERATE\n")
        f.write(f"{node_ids[(0, 0)]} , {node_ids[(0, ny - 1)]}, {nx}\n")
        f.write("*NSET, NSET=X1, GENERATE\n")
        f.write(f"{node_ids[(nx - 1, 0)]} , {node_ids[(nx - 1, ny - 1)]}, {nx}\n")

        f.write("*NSET, NSET=Y0, GENERATE\n")
        f.write(f"{node_ids[(0, 0)]} , {node_ids[(nx - 1, 0)]}, 1\n")
        f.write("*NSET, NSET=Y1, GENERATE\n")
        f.write(f"{node_ids[(0, ny - 1)]} , {node_ids[(nx - 1, ny - 1)]}, 1\n")

        # Elementos
        f.write("*ELEMENT, TYPE=S4R, ELSET=ALLELEMENTS\n")
        for elem in elements:
            f.write(f"{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}\n")

        # Propriedades do Material
        for l in range(number_of_layers):
            z = -h / 2 + h / number_of_layers * (l + 1 / 2)
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

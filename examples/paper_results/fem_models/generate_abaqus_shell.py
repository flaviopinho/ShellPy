import numpy as np
import matplotlib.pyplot as plt

from examples.paper_results.fem_models.generate_boundary_conditions import generate_bc_lines


# === DEFINIÇÃO DA SUPERFÍCIE ===
def R(xi1_, xi2_):
    alpha = np.deg2rad(45)
    return xi1_ * np.sin(alpha) * np.cos(xi2_), xi1_ * np.sin(alpha) * np.sin(xi2_), xi1_ * np.cos(alpha)


if __name__ == "__main__":
    # Define boundary conditions
    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("C", "C"),
                              "xi2": ("R", "R")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    # === PARÂMETROS ===

    factor = 0.5
    alpha = np.deg2rad(45)
    L = 1
    R2 = L * np.sin(alpha) / factor
    L2 = R2 / np.sin(alpha)
    L1 = L2 - L
    R1 = L1 * np.sin(alpha)

    h = 0.01 * R2

    b = 2 * np.pi

    u1, u2 = L1, L2
    v1, v2 = 0, b
    nu, nv = 100, 100

    periodic_u = False
    periodic_v = True

    # === PROPRIEDADES DO MATERIAL ===
    rho = 2710
    E = 70E9
    nu_mat = 0.3

    thickness = h  # Espessura da casca [m]

    # === GERAÇÃO DA MALHA ===
    num_u_nodes = nu if periodic_u else nu + 1
    num_v_nodes = nv if periodic_v else nv + 1

    u_vals = np.linspace(u1, u2, num_u_nodes, endpoint=not periodic_u)
    v_vals = np.linspace(v1, v2, num_v_nodes, endpoint=not periodic_v)

    nodes = []
    node_ids = {}
    node_id = 1

    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            x, y, z = R(u, v)
            nodes.append((node_id, x, y, z))
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
            nodes[n1 - 1][1:], nodes[n2 - 1][1:], nodes[n3 - 1][1:], nodes[n4 - 1][1:], nodes[n1 - 1][1:]
        ])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)

    ax.set_title("Visualização da malha 3D")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.show()

    # === ESCRITA DO ARQUIVO ABAQUS .inp ===
    with open("malha_abaqus_casca_SR4.inp", "w") as f:
        f.write("*HEADING\n")
        f.write("Modelo de Casca Gerado Automaticamente\n")
        f.write("Sistema de Unidades: SI (m, kg, s)\n")

        # Nós
        f.write("*NODE, NSET=ALLNODES\n")
        for node in nodes:
            f.write(f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}\n")

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
        f.write("*MATERIAL, NAME=MATERIAL_ELASTICO\n")
        f.write("*ELASTIC\n")
        f.write(f"{E}, {nu_mat}\n")
        f.write("*DENSITY\n")
        f.write(f"{rho}\n")

        # Seção da Casca
        f.write(f"*SHELL SECTION, ELSET=ALLELEMENTS, MATERIAL=MATERIAL_ELASTICO\n")
        f.write(f"{thickness}\n")

        lines = generate_bc_lines(boundary_conditions)
        for line in lines:
            f.write(line + "\n")

        # Passo de análise
        f.write("*STEP\n")
        f.write("*FREQUENCY\n")
        f.write("10, 100, 100000\n")
        f.write("*END STEP\n")

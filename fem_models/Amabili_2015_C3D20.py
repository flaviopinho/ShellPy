# Example based on the shell with orthotropic laminated material from
# the article: Qatu and Leissa 1991 (Table 6.12)


import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from fem_models.generate_boundary_conditions import generate_bc_lines
from fem_models.generate_boundary_transformation import generate_boundary_transformation
from fem_models.generate_kinematic_constraint import generate_kinematic_constraint
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, simply_supported, MidSurfaceGeometry, ConstantThickness
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial

if __name__ == "__main__":
    # Define boundary conditions
    boundary_conditions_u1 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    # === PARÂMETROS ===
    periodic_u = False
    periodic_v = True

    R = 0.15
    L = 0.52
    h = 0.03
    E = 198E9
    nu = 0.3
    rho = 7800

    number_of_layers = 6

    rectangular_domain = RectangularMidSurfaceDomain(0, L, 0, 2 * np.pi)

    R_ = sym.Matrix([R * sym.cos(xi2_), R * sym.sin(xi2_), xi1_])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    u1, u2 = rectangular_domain.edges["xi1"]
    v1, v2 = rectangular_domain.edges["xi2"]
    n_u, n_v, n_w = 40, 80, number_of_layers

    # === GERAÇÃO DA MALHA ===
    num_u_nodes = n_u * 2 if periodic_u else n_u * 2 + 1
    num_v_nodes = n_v * 2 if periodic_v else n_v * 2 + 1
    num_w_nodes = n_w * 2 + 1

    u_vals = np.linspace(u1, u2, num_u_nodes, endpoint=not periodic_u)
    v_vals = np.linspace(v1, v2, num_v_nodes, endpoint=not periodic_v)
    w_vals = np.linspace(-h / 2, h / 2, num_w_nodes)

    nodes = []
    node_ids = {}
    node_id = 1
    for k, w in enumerate(w_vals):
        for j, v in enumerate(v_vals):
            for i, u in enumerate(u_vals):
                MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)

                R = mid_surface_geometry(u, v).squeeze()
                R1 = R + w * MR3
                x, y, z = R1.flatten()

                nodes.append((node_id, x, y, z))
                node_ids[(i, j, k)] = node_id
                node_id += 1

    node_ids2 = {}
    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            if (i % 2 == 1) and (j % 2 == 1):
                continue

            if (k % 2 == 1) and (i % 2 == 1) and (i % 2 == 0):
                continue

            if (k % 2 == 1) and (i % 2 == 0) and (i % 2 == 1):
                continue
            node_ids2[(i, j)] = node_ids[(i, j, (n_w // 2) * 2)]

    elements = []
    elem_id = 1

    # Define limites do laço dependendo da periodicidade
    iu_max = num_u_nodes if periodic_u else num_u_nodes - 1
    iv_max = num_v_nodes if periodic_v else num_v_nodes - 1

    for k in range(n_w):
        for j in range(n_v):
            for i in range(n_u):

                i0 = i * 2
                i1 = i * 2 + 1
                i2 = i * 2 + 2
                if i2 >= num_u_nodes:
                    i2 = 0

                j0 = j * 2
                j1 = j * 2 + 1
                j2 = j * 2 + 2
                if j2 >= num_v_nodes:
                    j2 = 0

                k0 = k * 2
                k1 = k * 2 + 1
                k2 = k * 2 + 2

                n1 = node_ids[(i0, j0, k0)]
                n2 = node_ids[(i2, j0, k0)]
                n3 = node_ids[(i2, j2, k0)]
                n4 = node_ids[(i0, j2, k0)]

                n5 = node_ids[(i0, j0, k2)]
                n6 = node_ids[(i2, j0, k2)]
                n7 = node_ids[(i2, j2, k2)]
                n8 = node_ids[(i0, j2, k2)]

                n9 = node_ids[(i1, j0, k0)]
                n10 = node_ids[(i2, j1, k0)]
                n11 = node_ids[(i1, j2, k0)]
                n12 = node_ids[(i0, j1, k0)]

                n13 = node_ids[(i1, j0, k2)]
                n14 = node_ids[(i2, j1, k2)]
                n15 = node_ids[(i1, j2, k2)]
                n16 = node_ids[(i0, j1, k2)]

                n17 = node_ids[(i0, j0, k1)]
                n18 = node_ids[(i2, j0, k1)]
                n19 = node_ids[(i2, j2, k1)]
                n20 = node_ids[(i0, j2, k1)]

                elements.append((elem_id, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20))
                elem_id += 1

    # === VISUALIZAÇÃO 3D ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem in elements:
        _, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20 = elem
        coords = np.array([
            nodes[n1 - 1][1:4], nodes[n2 - 1][1:4], nodes[n3 - 1][1:4], nodes[n4 - 1][1:4], nodes[n1 - 1][1:4]
        ])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)
        coords = np.array([
            nodes[n5 - 1][1:4], nodes[n6 - 1][1:4], nodes[n7 - 1][1:4], nodes[n8 - 1][1:4], nodes[n5 - 1][1:4]
        ])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)

    base = f"Amabili_2015_C3D20_90"
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
            f.write(f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}, ")
            # f.write(f"{node[4][0]:.6f}, {node[4][1]:.6f}, {node[4][2]:.6f}, ")
            # f.write(f"{node[5][0]:.6f}, {node[5][1]:.6f}, {node[5][2]:.6f}, ")
            # f.write(f"{node[6][0]:.6f}, {node[6][1]:.6f}, {node[6][2]:.6f}")
            f.write("\n")

        nx = u_vals.size
        ny = v_vals.size

        if not periodic_u:
            f.write("*NSET, NSET=X0\n")
            for i in range(ny):
                f.write(f"{node_ids2[(0, i)]} , ")
            f.write("\n")

            f.write("*NSET, NSET=X1\n")
            for i in range(ny):
                f.write(f"{node_ids2[(nx - 1, i)]} , ")
            f.write("\n")

        if not periodic_v:
            f.write("*NSET, NSET=Y0\n")
            for i in range(nx):
                f.write(f"{node_ids2[(i, 0)]} , ")
            f.write("\n")

            f.write("*NSET, NSET=Y1\n")
            for i in range(nx):
                f.write(f"{node_ids2[(i, ny-1)]} , ")
            f.write("\n")

        # Elementos
        f.write("*ELEMENT, TYPE=C3D20, ELSET=ALLELEMENTS\n")
        for elem in elements:
            f.write(
                f"{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}, {elem[8]}, {elem[9]}, {elem[10]}, {elem[11]}, {elem[12]}, {elem[13]}, {elem[14]}, {elem[15]}, {elem[16]}, {elem[17]}, {elem[18]}, {elem[19]}, {elem[20]}\n")

        generate_kinematic_constraint(f, node_ids, num_u_nodes, num_v_nodes, num_w_nodes, periodic_u, periodic_v)

        # Propriedades do Material
        f.write(f"*MATERIAL, NAME=MATERIAL_ELASTICO\n")
        f.write("*ELASTIC\n")
        f.write(f"{E}, {nu}\n")
        f.write("*DENSITY\n")
        f.write(f"{rho}\n")

        # Seção da Casca
        f.write(f"*SOLID SECTION, MATERIAL=MATERIAL_ELASTICO, ELSET=ALLELEMENTS\n")
        f.write(f"1.0\n")

        generate_boundary_transformation(node_ids2, u_vals, v_vals, nx, ny, mid_surface_geometry, f)

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

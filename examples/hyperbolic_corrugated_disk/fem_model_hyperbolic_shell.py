import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from examples.paper_results.fem_models.generate_boundary_conditions import generate_bc_lines
from examples.paper_results.fem_models.generate_boundary_transformation import generate_boundary_transformation
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry

if __name__ == "__main__":
    # Define boundary conditions
    boundary_conditions_u1 = {
        "xi1": ("S", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions_u2 = {
        "xi1": ("S", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions_u3 = {
        "xi1": ("C", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions = {
        "u1": boundary_conditions_u1,
        "u2": boundary_conditions_u2,
        "u3": boundary_conditions_u3
    }

    # --------------------------------------------------------------
    # Geometry parameters (corrugated shell)
    # --------------------------------------------------------------
    n = 0  # circumferential waves
    p = 1  # radial exponent
    L = 1.0  # radial length
    R_in = 0.3  # inner radius
    H = 0.3  # corrugation amplitude

    mode_theta = max(20, n * 16)

    h = L / 100
    density = 1

    # --------------------------------------------------------------
    # Material
    # --------------------------------------------------------------
    E = 1
    nu = 0.3

    # --------------------------------------------------------------
    # Integration points
    # --------------------------------------------------------------
    n_int_x = 20
    n_int_y = mode_theta
    n_int_z = 4

    # --------------------------------------------------------------
    # Domain (radial + angular)
    # --------------------------------------------------------------
    rectangular_domain = RectangularMidSurfaceDomain(R_in, R_in + L, 0, 2 * np.pi)

    # --------------------------------------------------------------
    # Mid-surface geometry
    # --------------------------------------------------------------
    R_ = sym.Matrix([
        xi1_ * sym.cos(xi2_),
        xi1_ * sym.sin(xi2_),
        H * ((xi1_ - R_in) / L)**p * sym.cos(n * xi2_)
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)

    u1, u2 = rectangular_domain.edges["xi1"]
    v1, v2 = rectangular_domain.edges["xi2"]
    n_u, n_v = n_int_x, n_int_y

    periodic_u = False
    periodic_v = True

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

    filename = f"hyperbolic_shell_S8R_n{n}_p{p}_L{L:g}_Rin{R_in:g}_H{H:g}_h{h:g}".replace('.', 'p') + ".inp"

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

        f.write(f"*MATERIAL, NAME=MATERIAL_ELASTICO\n")
        f.write("*ELASTIC\n")
        f.write(f"{E}, {nu}\n")
        f.write("*DENSITY\n")
        f.write(f"{density}\n")

        # Seção da Casca
        f.write(f"*SHELL SECTION, ELSET=ALLELEMENTS, MATERIAL=MATERIAL_ELASTICO\n")
        f.write(f"{h}, 3\n")

        generate_boundary_transformation(node_ids, u_vals, v_vals, nx, ny, mid_surface_geometry, f)

        lines = generate_bc_lines(boundary_conditions)
        for line in lines:
            f.write(line + "\n")

        # Passo de análise
        f.write("*STEP\n")
        f.write("*FREQUENCY\n")
        f.write("200, 0, 100000000\n")
        f.write("*END STEP\n")

    ax.set_title("Visualização da malha 3D")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.show()

"""
casca canoide parabólica laminada com elementos C3D20.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fem_models.generate_boundary_conditions import generate_bc_lines
from fem_models.generate_boundary_transformation import generate_boundary_transformation
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, ConstantThickness
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy.displacement_expansion import (SSSS, CCCC, SSCC, CCSS, SCCC, CSSS, SCSC, CSCS,
SSFF, FFSS, CCFS, CFCF, CCFF, FFCC, FCFC)

if __name__ == "__main__":

    boundary_conditions = FCFC

    # Parâmetros geométricos
    a = 1.0
    b = 1.0
    h = a / 100
    hh = a / 2.5
    h_l = hh * 0.25
    f1 = h_l
    f2 = hh

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Superfície canoide parabólica
    Z_conoid = f1 * (1 - (1 - f2/f1) * xi1_ / a) * (1 - (2 * xi2_ / b - 1)**2)
    R_ = sym.Matrix([xi1_, xi2_, Z_conoid])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    # Propriedades do material ortotrópico
    E22 = 1.0
    density = 1.0
    E11 = 25.0 * E22
    E33 = E22
    G12 = 0.5 * E22
    G13 = 0.5 * E22
    G23 = 0.2 * E22
    nu12 = 0.25
    nu13 = 0.25
    nu23 = 0.25

    number_of_layers = 8

    # Criar lâminas
    t_lamina = h/number_of_layers
        
    def create_lamina(angle_deg):
        return Lamina(
            E_11=E11,
            E_22=E22,
            E_33=E33,
            nu_12=nu12,
            nu_13=nu13,
            nu_23=nu23,
            G_12=G12,
            G_13=G13,
            G_23=G23,
            density=density,
            angle=angle_deg * np.pi / 180.0,
            thickness=t_lamina,)

    angles = [0, 90, 0, 90, 0, 90, 0, 90]
    laminas = [create_lamina(angle) for angle in angles]
    material = LaminateOrthotropicMaterial(laminas, thickness)

    u1, u2 = rectangular_domain.edges["xi1"]
    v1, v2 = rectangular_domain.edges["xi2"]

    # Número de elementos em cada direção
    n_u = 30
    n_v = 30
    n_w = number_of_layers

    periodic_u = False
    periodic_v = False

    num_u_nodes = n_u * 2 if periodic_u else n_u * 2 + 1
    num_v_nodes = n_v * 2 if periodic_v else n_v * 2 + 1
    num_w_nodes = n_w * 2 + 1

    u_vals = np.linspace(u1, u2, num_u_nodes, endpoint=not periodic_u)
    v_vals = np.linspace(v1, v2, num_v_nodes, endpoint=not periodic_v)
    w_vals = np.linspace(-h / 2, h / 2, num_w_nodes)

    # Cache de geometria da superfície média
    geometry_cache = {}
    for j, v in enumerate(v_vals):
        for i, u in enumerate(u_vals):
            MN1, MN2, MN3 = mid_surface_geometry.natural_base(u, v)
            MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)
            R = mid_surface_geometry(u, v).squeeze()
            geometry_cache[(i, j)] = (MN1, MN2, MN3, MR1, MR2, MR3, R)

    nodes = []
    node_ids = {}
    material_orientation = {}
    node_id = 1

    # Calcula orientação do material
    def calculate_material_orientation(i, j, k, u, v, w, MN1, MN2, MN3, MR2, R, MR3):
        angle = material.angle(u, v, w)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        R1 = R + w * MR3
        
        cross_MN3_MN1 = np.cross(MN3, MN1)
        dot_MN3_MN1 = np.dot(MN3, MN1)
        a1 = MN1 * cos_a + cross_MN3_MN1 * sin_a + MN3 * dot_MN3_MN1 * (1.0 - cos_a)
        a1 /= np.linalg.norm(a1)
        a1 += R1
        
        cross_MN3_MR2 = np.cross(MN3, MR2)
        dot_MN3_MR2 = np.dot(MN3, MR2)
        a2 = MR2 * cos_a + cross_MN3_MR2 * sin_a + MN3 * dot_MN3_MR2 * (1.0 - cos_a)
        a2 /= np.linalg.norm(a2)
        a2 += R1
        
        lamina = material.lamina_index(u, v, w)
        return (a1, a2, R1, lamina[0])

    # Geração dos nós da malha
    for k, w in enumerate(w_vals):
        for j, v in enumerate(v_vals):
            for i, u in enumerate(u_vals):
                MN1, MN2, MN3, MR1, MR2, MR3, R = geometry_cache[(i, j)]
                
                R1 = R + w * MR3
                x, y, z = R1.flatten()

                # Cache de orientação apenas para nós centrais dos elementos
                if (i % 2 == 1) and (j % 2 == 1) and (k % 2 == 1):
                    material_orientation[(i, j, k)] = calculate_material_orientation(
                        i, j, k, u, v, w, MN1, MN2, MN3, MR2, R, MR3
                    )

                # Nós intermediários nas arestas do plano médio não são criados
                if (i % 2 == 1) and (j % 2 == 1):
                    continue

                nodes.append((node_id, x, y, z))
                node_ids[(i, j, k)] = node_id
                node_id += 1

    # Mapeamento 2D para nós da superfície média
    node_ids2 = {}
    k_mid = (n_w // 2) * 2
    u_len = len(u_vals)
    v_len = len(v_vals)
    is_edge = lambda i, j: (i == 0 or i == u_len - 1 or j == 0 or j == v_len - 1)
    
    for j_idx in range(v_len):
        for i_idx in range(u_len):
            key_mid = (i_idx, j_idx, k_mid)
            if key_mid in node_ids:
                node_ids2[(i_idx, j_idx)] = node_ids[key_mid]
            elif is_edge(i_idx, j_idx):
                for k_test in range(num_w_nodes):
                    key_test = (i_idx, j_idx, k_test)
                    if key_test in node_ids:
                        node_ids2[(i_idx, j_idx)] = node_ids[key_test]
                        break

    # Geração dos elementos C3D20
    elements = []
    elem_id = 1

    for k in range(n_w):
        k0 = k * 2
        k1 = k * 2 + 1
        k2 = k * 2 + 2
        
        for j in range(n_v):
            j0 = j * 2
            j1 = j * 2 + 1
            j2 = j * 2 + 2
            if j2 >= num_v_nodes:
                j2 = 0
            
            for i in range(n_u):
                i0 = i * 2
                i1 = i * 2 + 1
                i2 = i * 2 + 2
                if i2 >= num_u_nodes:
                    i2 = 0

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

                orientation = material_orientation.get((i1, j1, k1), None)
                if orientation is None:
                    u_mid = u_vals[i1] if i1 < u_len else u_vals[i0]
                    v_mid = v_vals[j1] if j1 < v_len else v_vals[j0]
                    w_mid = w_vals[k1] if k1 < num_w_nodes else w_vals[k0]
                    
                    cache_key = (i1, j1) if i1 < u_len and j1 < v_len else (i0, j0)
                    if cache_key in geometry_cache:
                        MN1, MN2, MN3, MR1, MR2, MR3, R_mid = geometry_cache[cache_key]
                    else:
                        MN1, MN2, MN3 = mid_surface_geometry.natural_base(u_mid, v_mid)
                        MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u_mid, v_mid)
                        R_mid = mid_surface_geometry(u_mid, v_mid).squeeze()
                    
                    orientation = calculate_material_orientation(
                        i1, j1, k1, u_mid, v_mid, w_mid, MN1, MN2, MN3, MR2, R_mid, MR3
                    )

                elements.append((elem_id, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, orientation))
                elem_id += 1

    # Geração do arquivo Abaqus
    filename = os.path.join(os.path.dirname(__file__), "Sadhan_Chakravort_C3D20.inp")

    # Visualização da malha
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem in elements:
        _, n1, n2, n3, n4, n5, n6, n7, n8 = elem[:9]
        for face in ([n1, n2, n3, n4, n1], [n5, n6, n7, n8, n5]):
            coords = np.array([nodes[n - 1][1:4] for n in face])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'k-', linewidth=0.5)

    with open(filename, "w") as f:
        f.write("*HEADING\n")
        f.write("Modelo de Casca Canoide Parabolica Laminada - Sadhan Chakravort\n")
        f.write("Elementos C3D20 com tratamento de bordas\n")
        f.write("Sistema de Unidades: SI (m, kg, s)\n")

        f.write("*NODE, NSET=ALLNODES\n")
        node_lines = [f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}\n" for node in nodes]
        f.writelines(node_lines)

        nx = u_len
        ny = v_len

        def write_nset(nset_name, node_list):
            if not node_list:
                return
            f.write(f"*NSET, NSET={nset_name}\n")
            node_list = sorted(set(node_list))
            batch_size = 16
            for idx in range(0, len(node_list), batch_size):
                batch = node_list[idx:idx + batch_size]
                f.write(", ".join(map(str, batch)) + "\n")

        nodes_x0_mid = [node_ids2[(0, j_idx)] for j_idx in range(v_len) if (0, j_idx) in node_ids2]
        write_nset("X0", nodes_x0_mid)

        u_last = u_len - 1
        nodes_x1_mid = [node_ids2[(u_last, j_idx)] for j_idx in range(v_len) if (u_last, j_idx) in node_ids2]
        write_nset("X1", nodes_x1_mid)

        nodes_y0_mid = [node_ids2[(i_idx, 0)] for i_idx in range(u_len) if (i_idx, 0) in node_ids2]
        write_nset("Y0", nodes_y0_mid)

        v_last = v_len - 1
        nodes_y1_mid = [node_ids2[(i_idx, v_last)] for i_idx in range(u_len) if (i_idx, v_last) in node_ids2]
        write_nset("Y1", nodes_y1_mid)

        f.write("*ELEMENT, TYPE=C3D20, ELSET=ALLELEMENTS\n")
        elem_lines = []
        for elem in elements:
            elem_lines.append(
                f"{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, "
                f"{elem[6]}, {elem[7]}, {elem[8]}, {elem[9]}, {elem[10]}, {elem[11]}, "
                f"{elem[12]}, {elem[13]}, {elem[14]}, {elem[15]}, {elem[16]}, {elem[17]}, "
                f"{elem[18]}, {elem[19]}, {elem[20]}\n"
            )
        f.writelines(elem_lines)

        elset_lines = []
        for elem in elements:
            elset_lines.append(f"*ELSET, ELSET=EL_{elem[0]}\n{elem[0]}\n")
        f.writelines(elset_lines)

        for k in range(number_of_layers):
            z = -h / 2 + h / number_of_layers * (k + 1 / 2)
            f.write(f"*MATERIAL, NAME=MATERIAL_ORTO_{k}\n")
            f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
            idx = material.lamina_index(0, 0, z)
            lamina = material.laminas[idx[0]]
            f.write(f"{lamina.E_11}, {lamina.E_22}, {lamina.E_33}, {lamina.nu_12}, {lamina.nu_13}, {lamina.nu_23}, {lamina.G_12}, {lamina.G_13}\n")
            f.write(f"{lamina.G_23}\n")
            f.write("*DENSITY\n")
            f.write(f"{density}\n")

        section_lines = []
        for elem in elements:
            a1, a2, a3, lamina = elem[-1]
            section_lines.append(f"*ORIENTATION, NAME=ORI_{elem[0]}\n")
            section_lines.append(
                f"{a1[0]}, {a1[1]}, {a1[2]}, "
                f"{a2[0]}, {a2[1]}, {a2[2]}, "
                f"{a3[0]}, {a3[1]}, {a3[2]}\n"
            )
            section_lines.append(
                f"*SOLID SECTION, MATERIAL=MATERIAL_ORTO_{lamina}, "
                f"ORIENTATION=ORI_{elem[0]}, ELSET=EL_{elem[0]}\n"
            )
            section_lines.append("1.0\n")
        f.writelines(section_lines)

        generate_boundary_transformation(node_ids2, u_vals, v_vals, nx, ny, mid_surface_geometry, f)

        lines = generate_bc_lines(boundary_conditions)
        for line in lines:
            f.write(line + "\n")

        f.write("*STEP\n")
        f.write("*FREQUENCY\n")
        f.write("10\n")
        f.write("*END STEP\n")

    ax.set_title("Visualização da malha 3D - Casca Canoide")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.show()

    print(f"\nArquivo Abaqus gerado: {filename}")
    print(f"Total de nós: {len(nodes)}")
    print(f"Total de elementos C3D20: {len(elements)}")
    print(f"Número de camadas: {number_of_layers}")

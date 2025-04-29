import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === DEFINIÇÃO DA SUPERFÍCIE ===
def R(xi1_, xi2_):
    # Define geometric parameters of the shell
    RR = 1
    beta = np.radians(45)

    x = xi1_
    y = (RR-xi1_*np.tan(beta))*np.cos(xi2_)
    z = (RR-xi1_*np.tan(beta))*np.sin(xi2_)

    return x, y, z

# === PARÂMETROS ===
RR=1
beta = np.radians(45)
L = 0.5/(np.sin(beta)*np.cos(beta))*RR

u1, u2 = 0, L
v1, v2 = 0, 2 * np.pi
nu, nv = 50, 50
periodic_u = False
periodic_v = True

# === PROPRIEDADES DO MATERIAL ===
E = 210e9         # Módulo de elasticidade [Pa]
nu_mat = 0.3      # Coeficiente de Poisson
rho = 7800        # Densidade [kg/m³]
thickness = 0.01  # Espessura da casca [m]

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
        nodes[n1-1][1:], nodes[n2-1][1:], nodes[n3-1][1:], nodes[n4-1][1:], nodes[n1-1][1:]
    ])
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='black', linewidth=0.5)

ax.set_title("Visualização da malha 3D")
ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
plt.show()

# === ESCRITA DO ARQUIVO ABAQUS .inp ===
with open("malha_abaqus_truncated_shell.inp", "w") as f:
    f.write("*HEADING\n")
    f.write("Modelo de Casca Gerado Automaticamente\n")
    f.write("Sistema de Unidades: SI (m, kg, s)\n")

    # Nós
    f.write("*NODE, NSET=ALLNODES\n")
    for node in nodes:
        f.write(f"{node[0]}, {node[1]:.6f}, {node[2]:.6f}, {node[3]:.6f}\n")

    # Elementos
    f.write("*ELEMENT, TYPE=S4, ELSET=ALLELEMENTS\n")
    for elem in elements:
        f.write(f"{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}\n")

    # Propriedades do Material
    f.write("*MATERIAL, NAME=STEEL\n")
    f.write("*ELASTIC\n")
    f.write(f"{E}, {nu_mat}\n")
    f.write("*DENSITY\n")
    f.write(f"{rho}\n")

    # Seção da Casca
    f.write(f"*SHELL SECTION, ELSET=ALLELEMENTS, MATERIAL=STEEL, THICKNESS={thickness}\n")

    # Passo de análise
    f.write("*STEP\n")
    f.write("*STATIC\n")
    f.write("*END STEP\n")
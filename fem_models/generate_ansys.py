import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === DEFINIÇÃO DA SUPERFÍCIE ===
def R(u, v):
    R0 = 1.0  # raio maior
    r = 0.3   # raio menor
    x = (R0 + r * np.cos(v)) * np.cos(u)
    y = (R0 + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

# === PARÂMETROS ===
u1, u2 = 0, 2 * np.pi
v1, v2 = 0, 2 * np.pi
nu, nv = 40, 20
periodic_u = True
periodic_v = True

# === MALHA ===
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

for j in range(nv):
    for i in range(nu):
        i_next = (i + 1) % num_u_nodes if periodic_u else i + 1
        j_next = (j + 1) % num_v_nodes if periodic_v else j + 1

        if not periodic_u and i_next >= num_u_nodes:
            continue
        if not periodic_v and j_next >= num_v_nodes:
            continue

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
plt.show()


with open("malha_ansys_shell.cdb", "w") as f:
    f.write("! Arquivo de malha de casca para ANSYS (SHELL181)\n")
    f.write("/PREP7\n")
    f.write("ET,1,181\n")  # Tipo de elemento: SHELL181
    f.write("SECTYPE,1,SHELL\n")
    f.write("SECDATA,0.01\n")  # Espessura da casca (exemplo: 0.01 m)

    f.write("NBLOCK,6,SOLID\n(1i9,3e20.13)\n")
    for node in nodes:
        f.write(f"{node[0]:9d}{node[1]:20.13E}{node[2]:20.13E}{node[3]:20.13E}\n")
    f.write("-1\n")

    f.write("EBLOCK,19,SOLID\n(1i9,8i9)\n")
    for elem in elements:
        eid, n1, n2, n3, n4 = elem
        # SHELL181: 4 nós por elemento
        f.write(f"{eid:9d}{1:9d}{0:9d}{0:9d}{0:9d}{n1:9d}{n2:9d}{n3:9d}{n4:9d}\n")
    f.write("-1\n")
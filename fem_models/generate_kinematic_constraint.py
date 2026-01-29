def generate_kinematic_constraint(f, node_ids, num_u_nodes, num_v_nodes, num_w_nodes, periodic_u=False,
                                  periodic_v=False):
    k_mid = (num_w_nodes - 1) // 2

    if not periodic_u:
        i = 0
        for j in range(num_v_nodes):
            ref_node = node_ids[(i, j, k_mid)]
            f.write(f"*SURFACE, NAME=SURF_{i}_{j}, TYPE=NODE\n")
            for k in range(num_w_nodes):
                n1 = node_ids[(i, j, k)]
                if n1 != ref_node:
                    f.write(f"{n1}, 1\n")

            f.write(f"*COUPLING, NAME =CONST_{i}_{j}, REF NODE={ref_node}, SURFACE=SURF_{i}_{j}\n")
            f.write("*KINEMATIC\n")

        i = num_u_nodes - 1
        for j in range(num_v_nodes):
            ref_node = node_ids[(i, j, k_mid)]
            f.write(f"*SURFACE, NAME=SURF_{i}_{j}, TYPE=NODE\n")
            for k in range(num_w_nodes):
                n1 = node_ids[(i, j, k)]
                if n1 != ref_node:
                    f.write(f"{n1}, 1\n")

            f.write(f"*COUPLING, REF NODE={ref_node}, SURFACE=SURF_{i}_{j}\n")
            f.write("*KINEMATIC\n")

    if not periodic_v:
        j = 0
        for i in range(num_u_nodes):
            ref_node = node_ids[(i, j, k_mid)]
            f.write(f"*SURFACE, NAME=SURF_{i}_{j}, TYPE=NODE\n")
            for k in range(num_w_nodes):
                n1 = node_ids[(i, j, k)]
                if n1 != ref_node:
                    f.write(f"{n1}, 1\n")

            f.write(f"*COUPLING, REF NODE={ref_node}, SURFACE=SURF_{i}_{j}\n")
            f.write("*KINEMATIC\n")

        j = num_v_nodes - 1
        for i in range(num_u_nodes):
            ref_node = node_ids[(i, j, k_mid)]
            f.write(f"*SURFACE, NAME=SURF_{i}_{j}, TYPE=NODE\n")
            for k in range(num_w_nodes):
                n1 = node_ids[(i, j, k)]
                if n1 != ref_node:
                    f.write(f"{n1}, 1\n")

            f.write(f"*COUPLING, REF NODE={ref_node}, SURFACE=SURF_{i}_{j}\n")
            f.write("*KINEMATIC\n")

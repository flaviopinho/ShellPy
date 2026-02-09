def generate_boundary_transformation(node_ids, u_vals, v_vals, nx, ny, mid_surface_geometry, f):
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            # u = u_vals[i]
            # v = v_vals[j]
            node_id = node_ids.get((i, j), -1)  # retorna -1 se a chave não existe

            if node_id != -1:
                f.write(f"*NSET, NSET=NB_{i}_{j}\n")
                f.write(f"{node_id}\n")

    for i in [0, nx - 1]:
        for j, v in enumerate(v_vals):
            node_id = node_ids.get((i, j), -1)  # retorna -1 se a chave não existe

            if node_id != -1:
                u = u_vals[i]
                # v = v_vals[j]
                M1, M2, M3 = mid_surface_geometry.natural_base(u, v)
                MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)
                f.write(f"*TRANSFORM, NSET=NB_{i}_{j}, TYPE=R\n")
                f.write(f"{MR1[0]}, {MR1[1]}, {MR1[2]}, {M2[0]}, {M2[1]}, {M2[2]}\n")

    for j in [0, ny - 1]:
        for i, u in enumerate(u_vals[1:-1]):
            node_id = node_ids.get((i, j), -1)  # retorna -1 se a chave não existe

            if node_id != -1:
                # u = u_vals[i]
                v = v_vals[j]
                M1, M2, M3 = mid_surface_geometry.natural_base(u, v)
                MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(u, v)
                f.write(f"*TRANSFORM, NSET=NB_{i + 1}_{j}, TYPE=R\n")
                f.write(f"{M1[0]}, {M1[1]}, {M1[2]}, {MR2[0]}, {MR2[1]}, {MR2[2]}\n")

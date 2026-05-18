def generate_bc_lines(boundary_conditions):
    """
    Gera linhas *BOUNDARY para Abaqus usando os Node Sets X0, X1, Y0, Y1.

    boundary_conditions deve ser:
    {
        "u1": {"xi1": ("R", "R"), "xi2": ("S", "F")},
        "u2": {"xi1": ("R", "R"), "xi2": ("S", "F")},
        "u3": {"xi1": ("R", "R"), "xi2": ("C", "F")},
    }

    Retorna uma lista de strings para escrever no arquivo .inp
    """

    nset_mapping = {"xi1": {0: "X0", 1: "X1"},
                    "xi2": {0: "Y0", 1: "Y1"}}

    dof_map = {"u1": 1, "u2": 2, "u3": 3}

    lines = []

    for dof, bc_dict in boundary_conditions.items():
        for xi_dir, (bc0, bc1) in bc_dict.items():
            # Nó inicial
            nset0 = nset_mapping[xi_dir][0]
            nset1 = nset_mapping[xi_dir][1]

            def add_bc_line(nset_name, bc_type):
                if bc_type.upper() == "S":
                    # Restringe translacional
                    lines.append(f"*BOUNDARY, TYPE=DISPLACEMENT\n{nset_name}, {dof_map[dof]}, {dof_map[dof]}")
                elif bc_type.upper() == "C":
                    # Restringe translacional + rotações
                    lines.append(f"*BOUNDARY, TYPE=DISPLACEMENT\n{nset_name}, {dof_map[dof]}, {dof_map[dof]}")
                    lines.append(f"*BOUNDARY, TYPE=DISPLACEMENT\n{nset_name}, 4, 6")  # DOFs rotacionais
                elif bc_type.upper() == "FC":
                    # Restringe rotações
                    lines.append(f"*BOUNDARY, TYPE=DISPLACEMENT\n{nset_name}, 4, 6")  # DOFs rotacionais

            add_bc_line(nset0, bc0)
            add_bc_line(nset1, bc1)

    return lines
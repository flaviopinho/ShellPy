import numpy as np
from multipledispatch import dispatch

from shellpy import MidSurfaceGeometry, cache_function
from shellpy.fsdt6.transformation_matrix import transformation_matrix_rotation
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.materials.laminate_orthotropic_material import LaminateOrthotropicMaterial


@dispatch(MidSurfaceGeometry, IsotropicHomogeneousLinearElasticMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry, material, position=None):
    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------
    E = material.E
    nu = material.nu

    D = E / ((1 + nu) * (1 - 2 * nu))

    # Initialize matrix with zeros, shape (6,6)+E.shape
    C = np.zeros((6, 6))

    # Fill in normal components
    C[0, 0] = (1 - nu) * D
    C[1, 1] = (1 - nu) * D
    C[2, 2] = (1 - nu) * D

    C[0, 1] = nu * D
    C[1, 0] = nu * D
    C[0, 2] = nu * D
    C[2, 0] = nu * D
    C[1, 2] = nu * D
    C[2, 1] = nu * D

    # Shear components
    C[3, 3] = (1 - 2 * nu) * D
    C[4, 4] = (1 - 2 * nu) * D
    C[5, 5] = (1 - 2 * nu) * D

    return C


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry: MidSurfaceGeometry,
                                          material: FunctionallyGradedMaterial, position):
    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------

    xi1 = position[0]
    xi2 = position[1]
    xi3 = position[2]

    E = material.E(xi3)
    nu = material.nu(xi3)
    G = E / (2 * (1 + nu))
    C = np.zeros((6, 6) + E.shape)
    plane_stress = False
    if plane_stress:
        E3 = material.E(xi3)*10
        nu13 = 0.01
        D = E / (1 - nu ** 2)
        S = np.zeros((6, 6) + E.shape)
        # Fill in normal components
        S[0, 0] = 1/E
        S[1, 1] = 1/E
        S[2, 2] = 1/E

        S[0, 1] = -nu / E
        S[1, 0] = -nu / E
        S[0, 2] = 0
        S[2, 0] = 0
        S[1, 2] = 0
        S[2, 1] = 0

        # Shear components
        S[3, 3] = 1 / G
        S[4, 4] = 1 / G
        S[5, 5] = 1 / G

        n_dim = C.ndim

        # Eixos que serão movidos: 0 e 1 (as dimensões 6x6)
        eixos_originais = [0, 1]

        # Novas posições: as últimas duas posições (N-2, N-1)
        novas_posicoes = [n_dim - 2, n_dim - 1]

        S_permutado = np.moveaxis(S, eixos_originais, novas_posicoes)

        C_permutado = np.linalg.inv(S_permutado)

        eixos_originais_reverso = [n_dim - 2, n_dim - 1]
        novas_posicoes_reverso = [0, 1]

        C = np.moveaxis(C_permutado, eixos_originais_reverso, novas_posicoes_reverso)

    else:
        C[0, 0] = (1 / E)
        C[0, 1] = -nu / E
        C[0, 2] = -nu / E

        C[1, 0] = -nu / E
        C[1, 1] = 1 / E
        C[1, 2] = -nu / E

        C[2, 0] = -nu / E
        C[2, 1] = -nu / E
        C[2, 2] = 1 / E

        C[3, 3] = 1 / G
        C[4, 4] = 1 / G
        C[5, 5] = 1 / G

        n_dim = C.ndim

        # Eixos que serão movidos: 0 e 1 (as dimensões 6x6)
        eixos_originais = [0, 1]

        # Novas posições: as últimas duas posições (N-2, N-1)
        novas_posicoes = [n_dim - 2, n_dim - 1]

        C_permutado = np.moveaxis(C, eixos_originais, novas_posicoes)

        S_permutado = np.linalg.inv(C_permutado)

        eixos_originais_reverso = [n_dim - 2, n_dim - 1]
        novas_posicoes_reverso = [0, 1]

        C = np.moveaxis(S_permutado, eixos_originais_reverso, novas_posicoes_reverso)

    return C


@dispatch(MidSurfaceGeometry, LaminateOrthotropicMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry: MidSurfaceGeometry,
                                          material: LaminateOrthotropicMaterial, position):
    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------

    xi1 = position[0]
    xi2 = position[1]
    xi3 = position[2]

    # Obtém o índice da lâmina correspondente a cada ponto
    index = material.lamina_index(xi1, xi2, xi3)  # shape = xi3 broadcastado

    # Shape final do array de saída
    shape_out = (6, 6) + np.shape(index)
    C_all = np.zeros(shape_out, dtype=float)

    # Monta as matrizes C para cada lâmina
    C_laminas = []
    for lamina in material.laminas:
        angle = lamina.angle

        E1, E2, E3 = lamina.E_11, lamina.E_22, lamina.E_33
        nu12 = lamina.nu_12
        nu13 = lamina.nu_13
        nu23 = lamina.nu_23

        G12 = lamina.G_12
        G13 = lamina.G_13
        G23 = lamina.G_23

        nu21 = nu12 * E2 / E1
        nu31 = nu13 * E3 / E1
        nu32 = nu23 * E3 / E2

        C = np.zeros((6, 6))

        plane_stress = False
        if plane_stress:
            C[0, 0] = E1 / (1 - nu12 * nu21)
            C[1, 1] = E2 / (1 - nu12 * nu21)
            C[0, 1] = nu12 * E2 / (1 - nu12 * nu21)
            C[1, 0] = nu12 * E2 / (1 - nu12 * nu21)

            C[2, 2] = E3

            C[3, 3] = G23
            C[4, 4] = G13
            C[5, 5] = G12
        else:
            C[0, 0] = (1 / E1)
            C[0, 1] = -nu21 / E2
            C[0, 2] = -nu31 / E3

            C[1, 0] = -nu12 / E1
            C[1, 1] = 1 / E2
            C[1, 2] = -nu32 / E3

            C[2, 0] = -nu13 / E1
            C[2, 1] = -nu23 / E2
            C[2, 2] = 1 / E3

            C[3, 3] = 1 / G23
            C[4, 4] = 1 / G13
            C[5, 5] = 1 / G12
            C = np.linalg.inv(C)

        T = transformation_matrix_rotation(lamina.angle)

        C_rot = np.einsum('ji,jk,kl->il', T, C, T)

        C_laminas.append(C_rot)

    C_laminas = np.array(C_laminas)  # shape = (n_laminas, 6, 6)

    # Preenche C_all usando os índices de cada ponto
    for i_lam, C in enumerate(C_laminas):
        mask = (index == i_lam)
        if np.any(mask):
            # Broadcasting para todos os pontos que pertencem à lâmina
            C_all[:, :, mask] = C[:, :, np.newaxis]

    return C_all


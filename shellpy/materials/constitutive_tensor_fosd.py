import numpy as np
from multipledispatch import dispatch

from shellpy import MidSurfaceGeometry, cache_function
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial


@dispatch(MidSurfaceGeometry, IsotropicHomogeneousLinearElasticMaterial, np.ndarray, np.ndarray, np.ndarray)
@cache_function
def constitutive_tensor_for_fosd(mid_surface_geometry: MidSurfaceGeometry, material: IsotropicHomogeneousLinearElasticMaterial, xi1, xi2,
                                 xi3):
    n_xy = np.shape(xi1)
    n_xyz = n_xy + (np.shape(xi3)[-1],)

    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components_extended(
        xi1, xi2)

    shifter_tensor_inverse = mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)

    metric_tensor2 = np.zeros((3, 3) + n_xyz)

    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor_contravariant_components[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the first term of the constitutive tensor (T1)
    # This is a contraction of the metric tensor with itself.
    T1 = np.einsum('il...,jk...->ijkl...', metric_tensor2, metric_tensor2)

    # Calculate the second term of the constitutive tensor (T2)
    # This is a contraction of the metric tensor with itself, but with different indices.
    T2 = np.einsum('ik...,jl...->ijkl...', metric_tensor2, metric_tensor2)

    # Calculate the third term of the constitutive tensor (T3)
    # This is a contraction of the metric tensor with itself, with all indices intact.
    T3 = np.einsum('ij...,kl...->ijkl...', metric_tensor2, metric_tensor2)

    # Compute the constitutive tensor using the material properties E (Young's modulus) and nu (Poisson's ratio)
    # The formula incorporates the material's behavior and the metric tensor's influence.
    return material.E / (2 * (1 + material.nu)) * (T1 + T2) + material.E * material.nu / (
                (1 + material.nu) * (1 - 2 * material.nu)) * T3


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, np.ndarray, np.ndarray, np.ndarray)
@cache_function
def constitutive_tensor_for_fosd(mid_surface_geometry: MidSurfaceGeometry, material: FunctionallyGradedMaterial, xi1, xi2,
                                 xi3):
    n_xy = np.shape(xi1)
    n_xyz = n_xy + (np.shape(xi3)[-1],)

    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components_extended(
        xi1, xi2)

    shifter_tensor_inverse = mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)

    metric_tensor2 = np.zeros((3, 3) + n_xyz)

    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor_contravariant_components[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the first term of the constitutive tensor (T1)
    # This is a contraction of the metric tensor with itself.
    T1 = np.einsum('il...,jk...->ijkl...', metric_tensor2, metric_tensor2)

    # Calculate the second term of the constitutive tensor (T2)
    # This is a contraction of the metric tensor with itself, but with different indices.
    T2 = np.einsum('ik...,jl...->ijkl...', metric_tensor2, metric_tensor2)

    # Calculate the third term of the constitutive tensor (T3)
    # This is a contraction of the metric tensor with itself, with all indices intact.
    T3 = np.einsum('ij...,kl...->ijkl...', metric_tensor2, metric_tensor2)

    Ta = T1 + T2

    # Compute the constitutive tensor using the material properties E (Young's modulus) and nu (Poisson's ratio)
    # The formula incorporates the material's behavior and the metric tensor's influence.
    E = material.E(xi3)
    nu = material.E(xi3)

    aux1 = E / (2 * (1 + nu))
    aux2 = E * nu / ((1 + nu) * (1 - 2 * nu))

    return np.einsum('z, ijklxyz->ijklxyz', aux1, Ta) + np.einsum('z, ijklxyz->ijklxyz', aux2, T3)

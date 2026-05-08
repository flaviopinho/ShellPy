import numpy as np
from multipledispatch import dispatch

from ..cache_decorator import cache_function
from ..midsurface_geometry import MidSurfaceGeometry
from ..materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial


@dispatch(MidSurfaceGeometry, IsotropicHomogeneousLinearElasticMaterial, object, object, object)
@cache_function
def plane_stress_constitutive_tensor_for_koiter_theory(mid_surface_geometry: MidSurfaceGeometry, material: IsotropicHomogeneousLinearElasticMaterial, xi1, xi2, xi3=0):
    xi1 = np.atleast_2d(xi1)
    xi2 = np.atleast_2d(xi2)
    n_xy = np.shape(xi1)
    xi3 = np.atleast_1d(xi3)
    n_xyz = n_xy + (np.shape(xi3)[-1],)

    # If no metric tensor is provided, initialize it as a 2x2 identity tensor for the default case.
    metric_tensor = mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)

    # Calculate the first term of the constitutive tensor (T1)
    # This is a contraction of the metric tensor with itself.
    T1 = np.einsum('ij...,kl...->iklj...', metric_tensor, metric_tensor)

    # Calculate the second term of the constitutive tensor (T2)
    # This is a contraction of the metric tensor with itself, but with different indices.
    T2 = np.einsum('ij...,kl...->ikjl...', metric_tensor, metric_tensor)

    # Calculate the third term of the constitutive tensor (T3)
    # This is a contraction of the metric tensor with itself, with all indices intact.
    T3 = np.einsum('ij...,kl...->ijkl...', metric_tensor, metric_tensor)

    # Compute the constitutive tensor using the material properties E (Young's modulus) and nu (Poisson's ratio)
    # The formula incorporates the material's behavior and the metric tensor's influence.
    contitutive_tensor = material.E / (2 * (1 - material.nu ** 2)) * ((1 - material.nu) * (T1 + T2) + 2 * material.nu * T3)

    contitutive_tensor2 = np.repeat(contitutive_tensor[:, :, :, :, :, :, np.newaxis], n_xyz[2], axis=6)

    return np.squeeze(contitutive_tensor2)


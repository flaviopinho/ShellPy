import numpy as np
from multipledispatch import dispatch

from shellpy import cache_function, MidSurfaceGeometry
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.materials.laminate_orthotropic_material import LaminateOrthotropicMaterial
from shellpy.materials.orthotropic_material import OrthotropicMaterial


@cache_function
@dispatch(IsotropicHomogeneousLinearElasticMaterial, object, object, object)
def shell_density(material: IsotropicHomogeneousLinearElasticMaterial, xi1, xi2, xi3):
    xi3 = np.atleast_1d(xi3)
    return np.ones(np.shape(xi1) + (np.shape(xi3)[-1],)) * material.density


@cache_function
@dispatch(LaminateOrthotropicMaterial, object, object, object)
def shell_density(material: LaminateOrthotropicMaterial, xi1, xi2, xi3):
    indices = material.lamina_index(xi1, xi2, xi3)
    # Array de ângulos das lâminas
    density = np.array([l.density for l in material.laminas])

    # Retorna o ângulo correspondente a cada ponto
    return density[indices]

@cache_function
@dispatch(FunctionallyGradedMaterial, object, object, object)
def shell_density(material: FunctionallyGradedMaterial, xi1, xi2, xi3):
    xi3 = np.atleast_1d(xi3)
    return np.ones(np.shape(xi1) + (np.shape(xi3)[-1],)) * material.density(xi3)


@cache_function
@dispatch(OrthotropicMaterial, object, object, object)
def shell_density(material: OrthotropicMaterial, xi1, xi2, xi3):
    xi3 = np.atleast_1d(xi3)
    return np.ones(np.shape(xi1) + (np.shape(xi3)[-1],)) * material.density
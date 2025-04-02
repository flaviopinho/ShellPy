import numpy as np
from multipledispatch import dispatch

from shellpy import cache_function, MidSurfaceGeometry
from shellpy.materials.linear_elastic_material import LinearElasticMaterial


@cache_function
@dispatch(LinearElasticMaterial, object, object, object)
def shell_density(material: LinearElasticMaterial, xi1, xi2, xi3):
    return np.ones(np.shape(xi1) + np.shape(xi3)) * material.density

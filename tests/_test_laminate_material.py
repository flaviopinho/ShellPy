import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_, ConstantThickness
import sympy as sym
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy.fosd_theory6.fsdt6_transformation_matrix import transformation_matrix_fosd2_local

if __name__ == "__main__":

    # Cria duas lâminas idênticas
    lamina1 = Lamina(
        E_11=150e9,  # Pa
        E_22=10e9,
        E_33=10e9,
        nu_12=0.25,
        nu_21=0.25 * 10e9 / 150e9,  # regra de simetria nu21 = nu12*E2/E1
        nu_13=0.25,
        nu_31=0.25 * 10e9 / 150e9,
        nu_23=0.3,
        nu_32=0.3 * 10e9 / 10e9,  # aqui fica 0.3 mesmo
        density=1600,  # kg/m³
        theta=np.pi/3,  # orientação (graus ou rad, depende da convenção)
        thickness=1  # 1 mm
    )

    lamina2 = Lamina(
        E_11=150e9,
        E_22=10e9,
        E_33=10e9,
        nu_12=0.25,
        nu_21=0.25 * 10e9 / 150e9,
        nu_13=0.25,
        nu_31=0.25 * 10e9 / 150e9,
        nu_23=0.3,
        nu_32=0.3,
        density=1600,
        theta=np.pi/3,  # por exemplo, segunda lâmina girada
        thickness=1
    )

    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    # Cria o laminado
    material = LaminateOrthotropicMaterial([lamina1, lamina2], ConstantThickness(h))

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    xi1_lin = np.linspace(0, a, 10)
    xi2_lin = np.linspace(0, b, 10)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')
    xi3 = np.linspace(-h / 2, h / 2, 3)

    alpha = material.angle(xi1, xi2, xi3)
    T = transformation_matrix_fosd2_local(mid_surface_geometry, xi1, xi2, xi3, alpha)

    print("shape of transformation matrix:", T.shape)

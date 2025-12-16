import numpy as np


def shear_correction_factor(C_local, xi3, Wz, det_shifter_tensor):

    C0 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 0 * Wz)
    C1 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 1 * Wz)
    C2 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 2 * Wz)

    shape_xy = np.shape(C0)[2:]
    shape_xyz = np.shape(xi3)

    M = np.zeros((12, 12) + shape_xy)
    M[0:6, 0:6] = C0
    M[0:6, 6:12] = C1

    M[6:12, 0:6] = C1
    M[6:12, 6:12] = C2

    M_swapped = np.moveaxis(M, source=(0, 1), destination=(2, 3))
    H_swapped = np.linalg.inv(M_swapped)
    H = np.moveaxis(H_swapped, source=(0, 1), destination=(2, 3))

    Bx1 = np.einsum('jxyz, jxy -> xyz', C_local[0], H[0:6, 6])
    Bx2 = np.einsum('jxyz, jxy -> xyz', C_local[0], H[6:12, 6])

    By1 = np.einsum('jxyz, jxy -> xyz', C_local[1], H[0:6, 7])
    By2 = np.einsum('jxyz, jxy -> xyz', C_local[1], H[6:12, 7])

    aux_x = (Bx1 + xi3 * Bx2) * Wz
    aux_y = (By1 + xi3 * By2) * Wz

    tau_xz = np.zeros(shape_xyz)
    tau_xz[:, :, 0] = aux_x[:, :, 0] / 2

    tau_yz = np.zeros(shape_xyz)
    tau_yz[:, :, 0] = aux_y[:, :, 0] / 2

    for i in range(1, shape_xyz[-1]):
        tau_xz[:, :, i] = tau_xz[:, :, i - 1] + aux_x[:, :, i - 1] / 2 + aux_x[:, :, i] / 2
        tau_yz[:, :, i] = tau_yz[:, :, i - 1] + aux_y[:, :, i - 1] / 2 + aux_y[:, :, i] / 2

    Cs = C_local[3:5, 3:5]

    Ux = 0.5 * np.einsum('xyz, xyz -> xy', tau_xz ** 2 / Cs[1, 1], Wz * det_shifter_tensor)
    Uy = 0.5 * np.einsum('xyz, xyz -> xy', tau_yz ** 2 / Cs[0, 0], Wz * det_shifter_tensor)

    Cs_integral = np.einsum('abxyz, xyz -> abxy', Cs, Wz * det_shifter_tensor)

    """
    Ux = 0.5 * np.einsum('xyz, xyz -> xy', tau_xz ** 2 / Cs[1, 1], Wz)
    Uy = 0.5 * np.einsum('xyz, xyz -> xy', tau_yz ** 2 / Cs[0, 0], Wz)

    Cs_integral = np.einsum('abxyz, xyz -> abxy', Cs, Wz)
    """

    Delta = Cs_integral[0, 0] * Cs_integral[1, 1] - Cs_integral[0, 1] ** 2

    k1 = 0.5 * Cs_integral[0, 0] / Ux / Delta
    k2 = 0.5 * Cs_integral[1, 1] / Uy / Delta
    k3 = np.sqrt(k1*k2)

    return k1, k2, k3

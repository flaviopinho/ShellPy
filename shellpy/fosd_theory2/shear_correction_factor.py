import numpy as np


def shear_correction_factor(C_local, xi3, Wz, det_shifter_tensor):
    C0 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 0 * det_shifter_tensor * Wz)
    C1 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 1 * det_shifter_tensor * Wz)
    C2 = np.einsum('ijxyz, xyz->ijxy', C_local, xi3 ** 2 * det_shifter_tensor * Wz)

    shape_xyz = np.shape(C_local)[2:]
    shape_xy = np.shape(C_local)[2:-1]

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

    aux_x = (Bx1 + xi3 * Bx2) * det_shifter_tensor * Wz
    aux_y = (By1 + xi3 * By2) * det_shifter_tensor * Wz

    tau_xz = np.zeros(shape_xyz)
    tau_xz[:, :, 0] = aux_x[:, :, 0] / 2

    tau_yz = np.zeros(shape_xyz)
    tau_yz[:, :, 0] = aux_x[:, :, 0] / 2

    for i in range(1, shape_xyz[-1]):
        tau_xz[:, :, i] = tau_xz[:, :, i - 1] + aux_x[:, :, i-1] / 2 + aux_x[:, :, i] / 2
        tau_yz[:, :, i] = tau_yz[:, :, i - 1] + aux_y[:, :, i-1] / 2 + aux_y[:, :, i] / 2

    Gx = np.einsum('xyz, xyz -> xy', C_local[4, 4] * det_shifter_tensor, Wz)
    Gy = np.einsum('xyz, xyz -> xy', C_local[3, 3] * det_shifter_tensor, Wz)

    Uxz_eq = np.einsum('xyz, xyz -> xy', tau_xz ** 2 / C_local[4, 4] * det_shifter_tensor, Wz)
    Uyz_eq = np.einsum('xyz, xyz -> xy', tau_yz ** 2 / C_local[3, 3] * det_shifter_tensor, Wz)

    Uxz_FSDT = 1 / Gx
    Uyz_FSDT = 1 / Gy

    return Uxz_FSDT / Uxz_eq, Uyz_FSDT / Uyz_eq

import numpy as np


def test_voigt_notation():
    """Test if the 4th-order constitutive tensor satisfies minor and major symmetries."""

    # --- Initialization ---
    C = np.random.rand(3, 3, 3, 3)
    epsilon = np.random.rand(3, 3)
    epsilon = epsilon + epsilon.T  # make strain tensor symmetric

    # --- Impose minor symmetries: C_{ijkl} = C_{jikl} = C_{ijlk} ---
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[j, i, k, l] = C[i, j, k, l]
                    C[i, j, l, k] = C[i, j, k, l]
                    C[j, i, l, k] = C[i, j, k, l]

    # --- Impose major symmetry: C_{ijkl} = C_{klij} ---
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = 0.5 * (C[i, j, k, l] + C[k, l, i, j])
                    C[i, j, k, l] = val
                    C[k, l, i, j] = val

    # --- Assertions ---

    # 1️⃣ Check shape and data type
    assert C.shape == (3, 3, 3, 3), "Constitutive tensor must have shape (3,3,3,3)"
    assert np.issubdtype(C.dtype, np.floating), "Tensor must contain real numbers"

    # 2️⃣ Check minor symmetries
    assert np.allclose(C, np.swapaxes(C, 0, 1)), "Minor symmetry failed: C_ijkl ≠ C_jikl"
    assert np.allclose(C, np.swapaxes(C, 2, 3)), "Minor symmetry failed: C_ijkl ≠ C_ijlk"

    # 3️⃣ Check major symmetry
    assert np.allclose(C, np.transpose(C, (2, 3, 0, 1))), "Major symmetry failed: C_ijkl ≠ C_klij"

    # 4️⃣ Compute stress tensor and check basic properties
    sigma = np.einsum('ijkl,kl->ij', C, epsilon)
    assert sigma.shape == (3, 3), "Sigma must be a 3x3 tensor"
    assert np.allclose(sigma, sigma.T), "Stress tensor must be symmetric"

    # 5️⃣ Verify consistency between 4D tensor form and 2D matrix form
    C_matrix = C.reshape(9, 9)
    epsilon_vector = epsilon.reshape(9)
    sigma_vector = C_matrix @ epsilon_vector
    sigma_tensor = np.einsum('ijkl,kl->ij', C, epsilon)

    assert np.allclose(sigma_vector.reshape(3, 3), sigma_tensor, atol=1e-10), \
        "Mismatch between sigma computed in tensor and matrix form"

    print("✅ All tensor symmetry tests passed successfully!")

    epsilon_voigt_expected = np.array(
        [epsilon[0, 0], epsilon[1, 1], epsilon[2, 2], 2 * epsilon[1, 2], 2 * epsilon[0, 2], 2 * epsilon[0, 1]])
    sigma_voigt_expected = np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[1, 2], sigma[0, 2], sigma[0, 1]])

    # permutation matrix: transformation from 9x1 vector to 6x1 Voigt notation
    permutation_voigt = np.zeros((6, 9))
    permutation_voigt[0, 0] = 1
    permutation_voigt[1, 4] = 1
    permutation_voigt[2, 8] = 1
    permutation_voigt[3, 5] = 1
    permutation_voigt[3, 7] = 1
    permutation_voigt[4, 2] = 1
    permutation_voigt[4, 6] = 1
    permutation_voigt[5, 1] = 1
    permutation_voigt[5, 3] = 1

    # inverse permutation matrix: transformation from 6x1 Voigt to 9x1 vector
    inverse_permutation_voigt = np.zeros((9, 6))
    inverse_permutation_voigt[0, 0] = 1
    inverse_permutation_voigt[1, 5] = 0.5
    inverse_permutation_voigt[2, 4] = 0.5
    inverse_permutation_voigt[3, 5] = 0.5
    inverse_permutation_voigt[4, 1] = 1
    inverse_permutation_voigt[5, 3] = 0.5
    inverse_permutation_voigt[6, 4] = 0.5
    inverse_permutation_voigt[7, 3] = 0.5
    inverse_permutation_voigt[8, 2] = 1

    C_voigt = inverse_permutation_voigt.T @ C_matrix @ inverse_permutation_voigt

    epsilon_voigt = permutation_voigt @ epsilon_vector

    sigma_voigt = C_voigt @ epsilon_voigt

    assert np.allclose(epsilon_voigt, epsilon_voigt_expected, atol=1e-10), \
        "Mismatch between epsilon in Voigt notation"

    assert np.allclose(sigma_voigt, sigma_voigt_expected, atol=1e-10), \
        "Mismatch between sigma in Voigt notation"

    print("Sigma in tensor notation")
    print(sigma)

    print("Sigma in vector notation")
    print(sigma_vector)

    print("Sigma in voigt notation")
    print(sigma_voigt)


if __name__ == "__main__":
    test_voigt_notation()


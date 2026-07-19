import numpy as np
from scipy.linalg import null_space
from circuits.basic.householder_reflection import (
    create_householder_circuit_matrix,
)


def test_householder_reflection_properties():
    """Verify that Householder reflection is Unitary, Hermitian, and Involutory (H^2 = I)."""
    dim = 8
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v / np.linalg.norm(v)

    H = create_householder_circuit_matrix(v, verbose=False)

    # 1. Unitary: H^dagger @ H = I
    assert np.allclose(H.conj().T @ H, np.eye(dim))
    # 2. Hermitian: H^dagger = H
    assert np.allclose(H.conj().T, H)
    # 3. Involutory: H @ H = I
    assert np.allclose(H @ H, np.eye(dim))


def test_householder_reflection_action():
    """Verify that H|v> = -|v> and H|v_perp> = |v_perp>."""
    dim = 8
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v / np.linalg.norm(v)

    H = create_householder_circuit_matrix(v, verbose=False)

    # Reflection inverts phase of |v>: H|v> = -|v>
    assert np.allclose(H @ v, -v)

    # Reflection leaves orthogonal vectors unchanged: H|v_perp> = |v_perp>
    v_perp = null_space(v.reshape(1, -1).conj())[:, 0]
    assert np.allclose(H @ v_perp, v_perp)

import numpy as np
from algorithms.simulation.ansatz_hea import (
    I,
    Ry,
    kron_n,
    get_ansatz_matrix,
    get_ansatz_state,
    objective_function,
    H,
)


def test_ry_identity():
    """Verify Ry(0) yields the identity matrix."""
    assert np.allclose(Ry(0), I)


def test_ry_pi():
    """Verify Ry(pi) yields a rotation of [[0, -1], [1, 0]]."""
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(Ry(np.pi), expected)


def test_kron_n_shape():
    """Verify kron_n returns the correct tensor product size."""
    m1 = np.eye(2)
    m2 = np.eye(2)
    m3 = np.eye(2)
    res = kron_n(m1, m2, m3)
    assert res.shape == (8, 8)


def test_ansatz_matrix_unitary():
    """Verify that get_ansatz_matrix generates an 8x8 unitary matrix."""
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    U = get_ansatz_matrix(params)
    assert U.shape == (8, 8)
    # Check unitariness: U^dagger * U = I
    U_dagger = U.conj().T
    assert np.allclose(U_dagger @ U, np.eye(8))


def test_ansatz_state_normalized():
    """Verify get_ansatz_state outputs a normalized statevector."""
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    state = get_ansatz_state(params)
    assert state.shape == (8,)
    # Norm must be 1
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0)


def test_objective_function_range():
    """Verify expectation energy is within the eigenvalue bounds of the Hamiltonian."""
    eigenvalues = np.linalg.eigvalsh(H)
    min_eigenval = eigenvalues[0]
    max_eigenval = eigenvalues[-1]

    # Test random parameters
    params = np.random.uniform(0, 2 * np.pi, 6)
    energy = objective_function(params)

    # The expectation value of H must lie within the range [min_eigenvalue, max_eigenvalue]
    assert min_eigenval - 1e-9 <= energy <= max_eigenval + 1e-9

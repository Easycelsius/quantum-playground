import numpy as np
from circuits.basic.bell_states import create_bell_state_phi_plus


def test_bell_state_dimensions():
    """Verify that the generated state vector has correct dimensions for 2 qubits."""
    state = create_bell_state_phi_plus()
    assert state.shape == (4,)


def test_bell_state_normalization():
    """Verify that the quantum state is properly normalized (norm equals 1)."""
    state = create_bell_state_phi_plus()
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0)


def test_bell_state_amplitudes():
    """Verify the mathematical amplitudes of the |Φ+⟩ state: 1/√2(|00⟩ + |11⟩)."""
    state = create_bell_state_phi_plus()

    # Expected amplitudes: [1/√2, 0, 0, 1/√2]
    expected = np.array([1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)])

    assert np.allclose(state, expected)

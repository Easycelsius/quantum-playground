import numpy as np


def create_bell_state_phi_plus():
    """
    Creates the Bell state |Φ+⟩ = 1/√2 (|00⟩ + |11⟩) from scratch using NumPy.

    Mathematical details:
    - Initial state: |00⟩ = |0⟩ ⊗ |0⟩ = [1, 0, 0, 0]^T
    - Hadamard on Qubit 0: H ⊗ I
    - CNOT on Qubit 0 (control) and Qubit 1 (target):
      CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
    """
    print("--- Starting Bell State (|Φ+⟩) Simulation ---")

    # 1. Define basis states
    ket_0 = np.array([1.0, 0.0])
    ket_1 = np.array([0.0, 1.0])

    # Initial state |00⟩
    state = np.kron(ket_0, ket_0)
    print(f"Initial State |00⟩:\n{state}\n")

    # 2. Define single-qubit gates
    I = np.eye(2)  # noqa: E741
    H = (1.0 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    # 3. Apply Hadamard to qubit 0 (first qubit)
    # Operator: H ⊗ I
    H_tensor_I = np.kron(H, I)
    state = np.dot(H_tensor_I, state)
    print(f"State after H ⊗ I (Superposition on Qubit 0):\n{state}")
    print("Analytical: 1/√2 (|00⟩ + |10⟩)\n")

    # 4. Define and apply CNOT gate
    # In the matrix representation for control=Q0, target=Q1:
    # CNOT = [ [1, 0, 0, 0],
    #          [0, 1, 0, 0],
    #          [0, 0, 0, 1],
    #          [0, 0, 1, 0] ]
    # Projectors:
    P0 = np.outer(ket_0, ket_0)  # |0⟩⟨0|
    P1 = np.outer(ket_1, ket_1)  # |1⟩⟨1|

    CNOT = np.kron(P0, I) + np.kron(P1, X)

    # Apply CNOT
    final_state = np.dot(CNOT, state)
    print(f"State after CNOT (Entangled Bell State |Φ+⟩):\n{final_state}")
    print("Analytical: 1/√2 (|00⟩ + |11⟩)\n")

    return final_state


def simulate_measurement(state, shots=1000):
    """
    Simulates measurement of the 2-qubit state.
    """
    print(f"--- Simulating Measurement ({shots} shots) ---")

    # Calculate probabilities: |a_i|^2
    probabilities = np.abs(state) ** 2
    states = ["00", "01", "10", "11"]

    print("Ideal Probabilities:")
    for s, p in zip(states, probabilities):
        print(f"  |{s}⟩: {p * 100:.1f}%")
    print()

    # Simulate random sampling based on probabilities
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    samples = np.random.choice(states, size=shots, p=probabilities)

    for sample in samples:
        counts[sample] += 1

    print("Measurement Results:")
    for s, count in counts.items():
        percentage = (count / shots) * 100
        bar = "█" * int(percentage // 2)
        print(f"  |{s}⟩: {count:4d} shots ({percentage:5.1f}%) {bar}")


if __name__ == "__main__":
    bell_state = create_bell_state_phi_plus()
    simulate_measurement(bell_state, shots=1000)

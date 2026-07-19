import numpy as np
from scipy.linalg import null_space

# Set NumPy print options for clean matrix visualization
np.set_printoptions(precision=4, suppress=True)


def create_householder_circuit_matrix(
    v: np.ndarray, verbose: bool = True
) -> np.ndarray:
    """Assembles the Householder reflection matrix for an arbitrary state vector v

    using quantum gate operations (U, X^⊗n, and MCZ).

    Mathematical Formulation:
        R_v = I - 2|v><v| = U @ R_0 @ U^dagger
        where R_0 = X^⊗n @ MCZ @ X^⊗n = I - 2|0...0><0...0|
        and U|0...0> = |v>.
    """
    # Ensure normalized input statevector: ||v|| = 1
    v = v / np.linalg.norm(v)

    n_qubits = int(np.log2(len(v)))
    dim = 2**n_qubits

    # ------------------------------------------------------------------------
    # 1. State preparation unitary U where U|0...0> = |v>
    # Matrix Structure:
    # Column 0 is |v>, and columns 1..dim-1 form an orthonormal basis
    # spanning the orthogonal complement space of |v>.
    # ------------------------------------------------------------------------
    U = np.zeros((dim, dim), dtype=complex)
    U[:, 0] = v
    # Find orthonormal basis for null space of v^\dagger
    ns = null_space(v.reshape(1, -1).conj())
    U[:, 1:] = ns

    # ------------------------------------------------------------------------
    # 2. Tensor product of Pauli-X gates (X^⊗n)
    # Matrix Structure:
    # An anti-diagonal identity matrix that flips all qubit basis states
    # e.g. for n=2:
    # [[0, 0, 0, 1],
    #  [0, 0, 1, 0],
    #  [0, 1, 0, 0],
    #  [1, 0, 0, 0]]
    # ------------------------------------------------------------------------
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # noqa: E741
    X_n = X
    for _ in range(n_qubits - 1):
        X_n = np.kron(X_n, X)

    # ------------------------------------------------------------------------
    # 3. Multi-Controlled Z gate (MCZ)
    # Matrix Structure:
    # Diagonal matrix applying a -1 phase shift only to the |11...1> state
    # e.g.: diag(1, 1, ..., 1, -1)
    # ------------------------------------------------------------------------
    MCZ = np.eye(dim, dtype=complex)
    MCZ[-1, -1] = -1.0

    # ------------------------------------------------------------------------
    # 4. Reflection operator about the |0...0> state (R_0 = I - 2|0...0><0...0|)
    # Matrix Structure:
    # X^⊗n maps |0...0> to |1...1>, MCZ flips phase of |1...1>, X^⊗n maps back.
    # Resulting matrix has -1 only at index [0, 0] and +1 elsewhere on diagonal:
    # e.g.: diag(-1, 1, 1, ..., 1)
    # ------------------------------------------------------------------------
    R_0 = X_n @ MCZ @ X_n

    # ------------------------------------------------------------------------
    # 5. Assemble full Householder reflection operator (H = U @ R_0 @ U^dagger)
    # Transforms R_0 back to the subspace of |v>: H = I - 2|v><v|
    # ------------------------------------------------------------------------
    U_dag = U.conj().T
    H_gate = U @ R_0 @ U_dag

    if verbose:
        print(f"1. Input Vector |v> (Normalized, dim={dim}):\n{v}\n")
        print(f"2. State Prep Unitary U (Column 0 = |v>):\n{U}\n")
        print(f"3. X^⊗n (Bit Flip Operator, n={n_qubits}):\n{X_n}\n")
        print("4. Multi-Controlled Z (MCZ, Phase flip at |1...1>):\n" f"{MCZ}\n")
        print("5. R_0 = X^⊗n @ MCZ @ X^⊗n (Phase flip at |0...0>):\n" f"{R_0}\n")
        print(f"6. Final Assembled Householder Matrix H_gate:\n{H_gate}\n")

    return H_gate


# ==========================================
# Verification & Example Execution
# ==========================================
if __name__ == "__main__":
    n_qubits = 3  # 3-qubit system (8 dimensions)
    dim = 2**n_qubits

    # Generate and normalize an arbitrary complex quantum state
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v / np.linalg.norm(v)

    print("=========================================================")
    print(" ⚛️ Householder Reflection Matrix Construction & Verification")
    print("=========================================================\n")

    # 1. Gate-based matrix construction (with step-by-step verbosity)
    H_gate_based = create_householder_circuit_matrix(v, verbose=True)

    # 2. Mathematical definition: H = I - 2|v><v|
    H_math = np.eye(dim, dtype=complex) - 2 * np.outer(v, v.conj())

    # Verify numerical equivalence
    is_exact = np.allclose(H_gate_based, H_math)
    print("=========================================================")
    print(" 📊 Result Summary")
    print("=========================================================")
    print(f"Statevector Dimension: {dim} ({n_qubits} qubits)")
    print(f"Gate-assembled matrix matches I - 2|v><v| definition: {is_exact}")


# ==============================================================================
# 💡 EXAMPLE OUTPUT & MATRIX STRUCTURAL ANATOMY (For Reference)
# ==============================================================================
#
# 1. Pauli-X Tensor Product (X^⊗3):
#    Anti-diagonal 8x8 matrix that swaps state |000> <-> |111>, |001> <-> |110>, etc.
#    [[0 0 0 0 0 0 0 1]
#     [0 0 0 0 0 0 1 0]
#     ...
#     [1 0 0 0 0 0 0 0]]
#
# 2. Multi-Controlled Z (MCZ):
#    Diagonal matrix flipping phase of the last state |111>:
#    diag(1, 1, 1, 1, 1, 1, 1, -1)
#
# 3. |000> Reflection Operator (R_0 = X^⊗3 @ MCZ @ X^⊗3):
#    Diagonal matrix flipping phase of the first state |000>:
#    diag(-1, 1, 1, 1, 1, 1, 1, 1)
#
# 4. Final Householder Reflection (H_gate = U @ R_0 @ U^dagger):
#    Complex Unitary & Hermitian matrix satisfying H = I - 2|v><v|
#    satisfying H|v> = -|v> and H|v_perp> = |v_perp>.
# ==============================================================================

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Numpy Setting
np.set_printoptions(precision=4, suppress=True)

# 1. Pauli matrices and gates
I = np.array([[1, 0], [0, 1]])  # noqa: E741
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


def Ry(theta):
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


def kron_n(*matrices):
    """Generate a 2^n dimensional matrix by calculating the tensor product of multiple matrices"""
    res = matrices[0]
    for m in matrices[1:]:
        res = np.kron(res, m)
    return res


# 2. 8x8 Observable
# H = - Z0 Z1 I - I Z1 Z2 - X0 I I - I X1 I - I I X2
H = (
    -kron_n(Z, Z, I)
    - kron_n(I, Z, Z)
    - kron_n(X, I, I)
    - kron_n(I, X, I)
    - kron_n(I, I, X)
)

print("Hamiltonian Matrix (H):")
print(H)

# Hamiltonian Matrix (H):
# [[-2 -1 -1  0 -1  0  0  0]
#  [-1  0  0 -1  0 -1  0  0]
#  [-1  0  2 -1  0  0 -1  0]
#  [ 0 -1 -1  0  0  0  0 -1]
#  [-1  0  0  0  0 -1 -1  0]
#  [ 0 -1  0  0 -1  2  0 -1]
#  [ 0  0 -1  0 -1  0  0 -1]
#  [ 0  0  0 -1  0 -1 -1 -2]]

# 3. 8x8 Entanglement Layer (CZ gates)
# CZ(0,1) and CZ(1,2) mapped to 8 dimensions
CZ_01 = np.diag([1, 1, 1, 1, 1, 1, -1, -1])
CZ_12 = np.diag([1, 1, 1, -1, 1, 1, 1, -1])
Entangle_Layer = CZ_12 @ CZ_01

print("Entanglement Layer (CZ_12 @ CZ_01):")
print(Entangle_Layer)

# Entanglement Layer (CZ_12 @ CZ_01):
# [[ 1  0  0  0  0  0  0  0]
#  [ 0  1  0  0  0  0  0  0]
#  [ 0  0  1  0  0  0  0  0]
#  [ 0  0  0 -1  0  0  0  0]
#  [ 0  0  0  0  1  0  0  0]
#  [ 0  0  0  0  0  1  0  0]
#  [ 0  0  0  0  0  0 -1  0]
#  [ 0  0  0  0  0  0  0  1]]


def get_ansatz_matrix(params):
    """Takes 6 parameters and returns the entire 8x8 Ansatz Unitary matrix U."""
    R_layer1 = kron_n(Ry(params[0]), Ry(params[1]), Ry(params[2]))
    R_layer2 = kron_n(Ry(params[3]), Ry(params[4]), Ry(params[5]))

    # Matrix multiplication is applied from right to left, so R_layer2 @ Entangle @ R_layer1
    U = R_layer2 @ Entangle_Layer @ R_layer1
    return U


def get_ansatz_state(params):
    """Generates the matrix U and applies it to the initial state |000>."""
    state_000 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    U = get_ansatz_matrix(params)
    return U @ state_000


# [Additional] Lists to record optimization trajectory
history_energy = []
history_params = []


def objective_function(params):
    state = get_ansatz_state(params)
    expectation = state.T @ H @ state

    # Evaluate and record energy and parameters
    history_energy.append(expectation)
    history_params.append(params.copy())

    return expectation


if __name__ == "__main__":
    # 4. Classical optimization (COBYLA)
    initial_params = np.random.uniform(0, 2 * np.pi, 6)

    print("Optimizing...")
    result = minimize(
        objective_function, initial_params, method="COBYLA", options={"maxiter": 300}
    )
    optimal_params = result.x % (2 * np.pi)

    print("=== Optimization Result ===")
    print(f"Minimum Energy (Expectation Value): {result.fun:.4f}")
    print(f"Optimal Parameters: {optimal_params}")

    # [Verification] Optimal Ansatz Matrix U_opt (First 2 rows only)
    U_opt = get_ansatz_matrix(optimal_params)
    print("\n[Verification] Optimal Ansatz Matrix U_opt (First 2 rows only):")
    print(U_opt[:2, :])

    # Verify U_opt diagonalizes H (or extracts the minimum energy)
    # <000| U_opt^T * H * U_opt |000> is the (0,0) element of the transformed matrix
    H_transformed = U_opt.T @ H @ U_opt
    print(
        f"\n[Verification] <000| U_opt^T * H * U_opt |000> Verification Energy: {H_transformed[0, 0]:.4f}"
    )
    print(
        f"(Should match Scipy Minimize's return value {result.fun:.4f} for a perfect optimization)"
    )

    # -------------------------------------------
    # Calculate all eigenvalues and eigenvectors of the Hamiltonian matrix H (classical method)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Eigenvalues are sorted in ascending order, so the first value is the 'True Ground State Energy'
    true_ground_energy = eigenvalues[0]
    true_ground_state = eigenvectors[:, 0]

    print("=== Exact Diagonalization (ED) Result ===")
    print(f"True Ground Energy: {true_ground_energy:.4f}")
    print(f"True Ground State: {true_ground_state}")
    # -------------------------------------------

    # Visualization
    plt.figure(figsize=(14, 5))

    # 1. Energy Convergence Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_energy, linestyle="-", marker="", linewidth=2, color="tab:blue")
    plt.title("VQE Energy Convergence (COBYLA)")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Expectation Value <H>")
    plt.axhline(y=result.fun, color="r", linestyle="--", label="Minimum Found")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # 2. Parameter Evolution Plot
    plt.subplot(1, 2, 2)
    history_params_arr = np.array(history_params) % (2 * np.pi)  # 0 ~ 2pi 사이로 정규화
    for i in range(6):
        plt.plot(history_params_arr[:, i], label=f"$\\theta_{i}$")
    plt.title("Ansatz Parameters Evolution")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Parameter Value (Radians)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()


# -------------------------------------------

# === Optimization Result ===
# Minimum Energy (Expectation Value): -3.4844
# Optimal Parameters: [5.7944 4.7416 2.6532 1.5811 0.0293 4.7222]
#
# [Verification] Optimal Ansatz Matrix U_opt (First 2 rows only):
# [[-0.5211  0.3228 -0.5114  0.3046  0.3229 -0.186   0.3046 -0.1957]
#  [-0.3139 -0.5265 -0.3138 -0.506   0.201   0.3139  0.1802  0.3138]]
#
# [Verification] <000| U_opt^T * H * U_opt |000> Verification Energy: -3.4844
# (Should match Scipy Minimize's return value -3.4844 for a perfect optimization)
#
# === Exact Diagonalization (ED) Result ===
# True Ground Energy: -3.4940
# True Ground State: [-0.5325 -0.2955 -0.2045 -0.2955 -0.2955 -0.2045 -0.2955 -0.5325]

# -------------------------------------------

# === Optimization Result ===
# Minimum Energy (Expectation Value): -3.4844
# Optimal Parameters: [0.4875 4.7292 0.4892 1.5655 3.1246 1.5645]

# [Verification] Optimal Ansatz Matrix U_opt (First 2 rows only):
# [[-0.5184 -0.3055  0.524   0.316  -0.3064 -0.1887  0.317   0.1831]
#  [-0.3108  0.5152  0.3107 -0.5272 -0.1799  0.3118  0.1918 -0.3117]]

# [Verification] <000| U_opt^T * H * U_opt |000> Verification Energy: -3.4844
# (Should match Scipy Minimize's return value -3.4844 for a perfect optimization)
# === Exact Diagonalization (ED) Result ===
# True Ground Energy: -3.4940
# True Ground State: [-0.5325 -0.2955 -0.2045 -0.2955 -0.2955 -0.2045 -0.2955 -0.5325]

# -------------------------------------------

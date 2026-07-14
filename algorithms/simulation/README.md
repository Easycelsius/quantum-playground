# ⚛️ VQE: Hardware-Efficient Ansatz (HEA)

This directory contains simulations of the Variational Quantum Eigensolver (VQE) using a Hardware-Efficient Ansatz (HEA) constructed with single-qubit rotation gates and entangling gates.

---

## 📝 Mathematical Background (WIP)

> [!NOTE]
> Detailed mathematical background, including the Hamiltonian transformation, Dirac notation representation of the ansatz state, and the optimization formulations will be documented here soon. (Work In Progress)

---

## 🚀 How to Run

Ensure your virtual environment is active and requirements are installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the ansatz VQE simulation
python algorithms/simulation/ansatz_hea.py
```

Upon execution, it will perform COBYLA optimization to find the minimum energy eigenvalue of the 3-qubit Hamiltonian and display parameter convergence plots using Matplotlib.

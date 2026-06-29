# ⚛️ quantum-playground

> A personal repository for exploring quantum computing — from first principles to real algorithms.

---

## 🧭 What This Repository Is

`quantum-playground` is a structured, hands-on journey through quantum computing. The focus is on **understanding by building** — every concept here is implemented from scratch where possible, before turning to established frameworks.

This repository covers three layers:

| Layer | What lives here |
|---|---|
| **Circuits** | Quantum gates, entanglement, oracles, and reusable circuit blocks |
| **Algorithms** | End-to-end implementations of landmark quantum algorithms |
| **Simulators** | Custom-built simulators (statevector, density matrix, tensor network) |

Whether you're studying quantum computing yourself or just curious about the implementations, feel free to explore, run, and adapt anything here.

---

## 📁 Repository Structure

```
quantum-playground/
│
├── README.md                        
├── ROADMAP.md                       # Planned topics and implementation queue
├── requirements.txt                 # Python dependencies
│
├── circuits/                        # Quantum circuit building blocks
│   ├── basic/                       # Single/multi-qubit gates, Bell states, superposition
│   ├── arithmetic/                  # Quantum adders, comparators, modular arithmetic
│   └── oracles/                     # Boolean oracles, phase oracles (used in algorithms)
│
├── algorithms/                      # Full algorithm implementations
│   ├── search/                      # Grover's algorithm
│   ├── factoring/                   # Shor's algorithm
│   ├── simulation/                  # VQE, QAOA (variational algorithms)
│   ├── cryptography/                # BB84, QKD protocols
│   └── optimization/                # QUBO, quantum annealing approaches
│
├── simulators/                      # Custom-built quantum simulators
│   ├── statevector/                 # Pure-state simulator (NumPy-based)
│   ├── density_matrix/              # Mixed-state + noise channel simulator
│   └── tensor_network/              # Tensor network simulator (advanced)
│
├── noise_models/                    # Quantum noise and decoherence modeling
│   └── error_correction/            # QEC
│
├── notebooks/                       # Jupyter notebooks
│   ├── tutorials/                   # Concept-first walkthroughs with visuals
│   └── experiments/                 # Algorithm runs, result analysis, plots
│
├── utils/                           # Shared utilities
│   ├── visualization.py             # Circuit diagrams, Bloch sphere, probability plots
│   ├── benchmarks.py                # Runtime and fidelity benchmarking
│   └── helpers.py                   # State prep, measurement, conversion tools
│
└── tests/                           # Unit tests
    ├── test_circuits.py
    ├── test_algorithms.py
    └── test_simulators.py
```

Each subdirectory contains its own `README.md` explaining the theory, the implementation approach, and how to run the code.

---

## 🗺️ Implementation Roadmap

This project is divided into multiple learning phases, scaling from first-principles circuits to complex algorithms and custom simulation engines.

* **Phase 1 (Foundational)**: Bell states, teleportation, basic arithmetic, and a pure-NumPy statevector simulator.
* **Phase 2 (Intermediate)**: Grover's search algorithm, noise models, and a density matrix simulator.
* **Phase 3 (Advanced)**: Shor's factoring algorithm, VQE, and Quantum Error Correction (QEC).

For a detailed task queue, progress status, and implementation details, check out the dedicated **[ROADMAP.md](./ROADMAP.md)**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Easycelsius/quantum-playground.git
cd quantum-playground

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running an Example

```bash
# Run Bell state simulation (Phase 1 Example)
python circuits/basic/bell_states.py

# Launch a Jupyter notebook
jupyter notebook notebooks/tutorials/
```

### Running Tests

```bash
# Run all unit tests using pytest
pytest
```

### Code Quality & Formatting

We use **Ruff** for fast linting and formatting, and **pre-commit** hooks to automate checks before each commit.

#### Setting up pre-commit

To register the git hook scripts:
```bash
pre-commit install
```

Once registered, pre-commit will automatically run Ruff checks and formatters on files staged for commit.

#### Running Linting & Formatting Manually

```bash
# Run Ruff to find and automatically fix safe lint errors
ruff check . --fix

# Format files using Ruff formatter
ruff format .
```

---

## 🧱 Design Principles

**1. From Scratch First**
Every concept is implemented using only NumPy before reaching for a framework. This keeps the math visible and honest.

```
statevector/
├── statevector_scratch.py    # Pure NumPy — no quantum libraries
├── statevector_qiskit.py     # Qiskit version for comparison
└── statevector_cirq.py       # Cirq version for comparison
```

**2. Self-Contained Modules**
Each algorithm or circuit lives in its own folder with a `README.md`, implementation, and tests. You can read or run any part of this repo independently.

**3. Theory Meets Code**
Every `README.md` in a subdirectory explains the relevant math — Dirac notation, circuit diagrams, complexity analysis — before showing the implementation. Code without context is just syntax.

**4. Noise Is Not an Afterthought**
Real quantum hardware is noisy. Noise modeling and error correction are treated as first-class topics, not appendices.

---

## 🧰 Tech Stack

| Tool | Role |
|---|---|
| [NumPy](https://numpy.org/) | Linear algebra, statevector math |
| [Matplotlib](https://matplotlib.org/) | Visualization, probability histograms |
| [Jupyter](https://jupyter.org/) | Interactive exploration and tutorials |
| [pytest](https://pytest.org/) | Unit testing |

---

## 🤝 Contributing

This is a personal learning repository, but feedback, corrections, and suggestions are always welcome.

If you spot a bug in an implementation, a mistake in the math, or have a cleaner way to express something:

1. Open an [Issue](https://github.com/Easycelsius/quantum-playground/issues) with a clear description
2. Or submit a Pull Request — please include a short explanation of what changed and why

---

## 📝 Git Commit Convention

To keep the repository history clean and readable, we follow the **Conventional Commits** specification. Format your commit messages as follows:

```
<type>(<scope>): <subject>

[optional body]
```

### Commit Types

* **`feat`**: A new feature or algorithm implementation (e.g., `feat(search): add grover search algorithm`).
* **`fix`**: A bug fix (e.g., `fix(simulator): resolve matrix multiplication dimension mismatch`).
* **`docs`**: Documentation changes only (e.g., `docs(readme): add commit message convention`).
* **`style`**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.).
* **`refactor`**: A code change that neither fixes a bug nor adds a feature.
* **`test`**: Adding missing tests or correcting existing tests (e.g., `test(circuits): add unit test for bell state`).
* **`chore`**: Changes to the build process or auxiliary tools and libraries (e.g., `chore(deps): update pre-commit config`).

---

<div align="center">
  <sub>Built with curiosity, one qubit at a time.</sub>
</div>

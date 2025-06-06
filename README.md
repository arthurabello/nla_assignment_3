# nla_assignment_3

**Undergraduate Numerical Linear Algebra - Assignment 3 Solutions**

---
This assignment tackles important numerical methods for matrix computations, such as **[Hessenberg reduction](https://fr.wikipedia.org/wiki/Matrice_de_Hessenberg)** via Householder reflectors and the spectral properties of **orthogonal matrices**.

---

## 📋 Contents

- [Assignment PDF](./written_assignment_3.pdf) — Full assignment description and instructions.

- [Implementation Notebook](./assignment.ipynb) — Complete Python code with plots and tests.

- [Report Document](./assignment.typ) — Formal written report explaining methodology, results, and theoretical insights.
---

## 🔍 Overview of Work

### 1. Hessenberg Reduction  
- Implemented `to_hessenberg(A)` function for reduction of arbitrary square matrices to upper Hessenberg form using Householder reflectors.  
- Verified correctness via numerical residuals $‖A - QHQ^*‖$ and $‖Q^*Q - I‖$ on symmetric and non-symmetric matrices of various sizes.  
- Benchmarked execution times across matrix sizes from 10 to 10,000, observing expected computational complexity and speed-ups for symmetric matrices.

### 2. Orthogonal Matrices and Eigenvalue Algorithms  
- Analyzed eigenvalues of orthogonal matrices, showing all lie on the unit circle and implications for power iteration and inverse iteration convergence.  
- Explicitly computed eigenvalues of 2x2 orthogonal blocks analytically.  
- Generated random orthogonal matrices via QR factorization, reduced to Hessenberg form, and computed eigenvalues of bottom-right blocks.  
- Explored QR iteration with fixed shifts using these eigenvalues, assessing convergence behavior.

---

## 🚀 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/arthurabello/nla_assignment_3.git
    cd nla_assignment_3
    ```
2. Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3. Open and run the Jupyter notebook:
    ```bash
    jupyter notebook assignment.ipynb
    ```
4. Follow the notebook to reproduce results, plots, and benchmarks.

---

## 📊 Highlights & Results

- Efficient Householder reduction implementation with detailed complexity analysis.  
- Visual and numerical confirmation of orthogonality and factorization accuracy.  
- Insightful spectral analysis of orthogonal matrices, confirming theoretical properties and numerical behaviors.  
- Experimentation with QR iteration shifts revealing convergence patterns (or lack thereof).

---

## 📚 References

- **[L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, 1997.](https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf)**

---


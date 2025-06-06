#import "@preview/ctheorems:1.1.3": *
#import "@preview/plotst:0.2.0": *
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#codly(languages: codly-languages)

#show: codly-init.with()
#show: thmrules.with(qed-symbol: $square$)
#show link: underline
#show ref: underline

#set heading(numbering: "1.1.")
#set page(numbering: "1")
#set heading(numbering: "1.")
#set math.equation(
  numbering: "(1)",
  supplement: none,
)

#set par(first-line-indent: 1.5em,justify: true)
#show ref: it => {
  // provide custom reference for equations
  if it.element != none and it.element.func() == math.equation {
    // optional: wrap inside link, so whole label is linked
    link(it.target)[eq.~(#it)]
  } else {
    it
  }
}

#let theorem = thmbox("theorem", "Theorem", fill: rgb("#ffeeee")) //theorem color
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Example").with(numbering: "1.")
#let proof = thmproof("proof", "Proof")

//shortcuts

#let inv(arg, power) = $arg^(-power)$
#let herm(arg) = $arg^*$
#let transpose(arg) = $arg^T$
#let inner(var1, var2) = $angle.l var1, var2 angle.r$


#align(center, text(20pt)[
 * Assignment 3 - Numerical Linear Algebra*
])

#align(center, text(15pt)[
  Arthur Rabello Oliveira
  #footnote[#link("https://emap.fgv.br/")[Escola de Matemática Aplicada, Fundação Getúlio Vargas (FGV/EMAp)], email: #link("mailto:arthur.oliveira.1@fgv.edu.br")]

  #datetime.today().display("[day]/[month]/[year]")
])

#align(center)[
  *Abstract*\
We design and test a function `to_hessemberg(A)` that reduces an arbitrary square matrix to (upper) Hessenberg form with Householder reflectors, returns the reflector vectors, the compact Hessenberg matrix H, and the accumulated orthogonal factor Q, verifying numerically that $A=Q H Q^*$ and $Q^\*Q=I$ for symmetric and nonsymmetric inputs of orders $10 - 10000$. Timings confirm the expected $O("something")$ cost and reveal the $2 times$ speed-up attainable for symmetric matrices through trivial bandwidth savings. Leveraging this routine, we investigate the spectral structure of orthogonal matrices: we show that all eigenvalues lie on the unit circle, analyse the consequences for the power method and inverse iteration, and obtain a closed-form spectrum for generic $2 times 2$ orthogonals. Random $4 times 4$ orthogonal matrices generated via QR factorisation are then reduced to Hessenberg form; the eigenvalues of their trailing $2 times 2$ blocks are computed analytically and reused as fixed shifts in the QR iteration, where experiments demonstrate markedly faster convergence. Throughout, every algorithm is documented and supported by commented plots that corroborate the theoretical claims.
]


//  TABLE OF CONTENTS ----------------------------------
#outline()
#pagebreak()

= Introduction
<section_introduction>
One could calculate the eigenvalues of a square matrix using the following algorithm:

+ Compute the $n$-th degree polynomial $det(A - lambda I) = 0$,

+ Solve for $lambda$ (somehow).

On step 2, the eigenvalue problem would have been reduced to a polynomial root-finding problem, which is awful and extremely ill-conditioned. #link("https://github.com/arthurabello/nla_assignment_2/blob/main/assignment.pdf")[From the previous assignment] we know that in the denominator of the relative condition number $kappa(x)$ there's a $abs(x - n)$. So $kappa(x) -> oo$ when $x -> 0$. As an example, consider the polynomial

$
  p(x) = (x - 2)^9 = x^9 - 18x^8 + 144x^7 - 672x^6 + 2016x^5\ - 4032x^4 + 5376x^3 - 4608x^2 + 2304x - 512
$ <equation_example_polynomial_introduction>

#figure(
  image("images/introduction_example_plot_1.png", width: 65%), caption: [
    $p(x)$ via the coefficients in @equation_example_polynomial_introduction
  ]
) <figure_example_plot_1_introduction>

#figure(
  image("images/introduction_example_plot_2.png", width: 65%), caption: [
    $p(x)$ via $(x - 2)^9$
  ]
) <figure_example_plot_2_introduction>

@figure_example_plot_2_introduction shows a smooth curve, while @figure_example_plot_1_introduction shows a weird oscillation around $x = 0$ (And pretty much everywhere else if the reader is sufficiently persistent).

This is due to the round-off errors when $x approx 0$ and the big coefficients of the polynomial. In general, polynomial are very sensitive to perturbations in the coefficients, which is why rootfinding is a bad idea to find eigenvalues.

Here we discuss aspects of some iterative eigenvalue algorithms, such as power iteration, inverse iteration, and QR iteration.

= Hessemberg Reduction (Problem 1)
<section_hessemberg_reduction>
== Calculating the Householder Reflectors (a)
<section_calculating_householder_reflectors>

The following packages will be used in the next functions:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hessenberg, qr, eig
import time
from typing import List, Tuple
import pandas as pd
import math
from IPython.display import display, Markdown
from ast import literal_eval
```

The following function calculates the Householder reflectors that reduce a matrix to Hessenberg form. It returns the reflector vectors, the compact Hessenberg matrix $H$, and the accumulated orthogonal factor $Q$.

```python
def build_householder_unit_vector(
        target_vector: np.ndarray
) -> np.ndarray:
    
    """
    Builds a Householder unit vector

    Args:
        1. target_vector (np.ndarray): Column vector that we want to annihilate (size ≥ 1).

    Returns:
        np.ndarray:
            The normalised Householder vector (‖v‖₂ = 1) with a real first component.

    Raises:
        1. ValueError: If 'target_vector' has zero length.
    """

    if target_vector.size == 0:
        raise ValueError("The target vector is empty; no reflector needed.")

    vector_norm: float = np.linalg.norm(target_vector)

    if vector_norm == 0.0: #nothing to annihilate – return canonical basis vector
        householder_vector: np.ndarray = np.zeros_like(target_vector)
        householder_vector[0] = 1.0
        return householder_vector

    sign_correction: float = (
        1.0 if target_vector[0].real >= 0.0 else -1.0
    )
    copy_of_target_vector: np.ndarray = target_vector.copy()
    copy_of_target_vector[0] += sign_correction * vector_norm
    householder_vector: np.ndarray = (
        copy_of_target_vector / np.linalg.norm(copy_of_target_vector)
    )
    return householder_vector


def to_hessenberg(
        original_matrix: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    
    """
    Reduce 'original_matrix' to upper Hessenberg form by Householder reflections.

    Args
        1. original_matrix (np.ndarray): Real or complex square matrix of order 'matrix_order'.

    Returns
        Tuple consisting of:

        1. householder_reflectors_list (List[np.ndarray])
        2. hessenberg_matrix (np.ndarray)
        3. accumulated_orthogonal_matrix (np.ndarray)  s.t.
          original_matrix = Q · H · Qᴴ

    Raises
        1. ValueError: If 'original_matrix' is not square.
    """

    working_matrix: np.ndarray = np.asarray(original_matrix).copy()

    if working_matrix.shape[0] != working_matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    matrix_order: int = working_matrix.shape[0]
    accumulated_orthogonal_matrix: np.ndarray = np.eye(
        matrix_order, dtype=working_matrix.dtype
    )
    householder_reflectors_list: List[np.ndarray] = []

    for column_index in range(matrix_order - 2): #extract the part of column 'column_index' that we want to zero out
        target_column_segment: np.ndarray = working_matrix[
            column_index + 1 :, column_index
        ]

        householder_vector: np.ndarray = build_householder_unit_vector(
            target_column_segment
        )  #build Householder vector for this segment
        householder_reflectors_list.append(householder_vector)

        #expand it to the full matrix dimension
        expanded_householder_vector: np.ndarray = np.zeros(
            matrix_order, dtype=working_matrix.dtype
        )
        expanded_householder_vector[column_index + 1 :] = householder_vector


        working_matrix -= 2.0 * np.outer( 
            expanded_householder_vector,
            expanded_householder_vector.conj().T @ working_matrix,
        ) #apply reflector from BOTH sides
        working_matrix -= 2.0 * np.outer(
            working_matrix @ expanded_householder_vector,
            expanded_householder_vector.conj().T,
        )

        #accumulate Q
        accumulated_orthogonal_matrix -= 2.0 * np.outer(
            accumulated_orthogonal_matrix @ expanded_householder_vector,
            expanded_householder_vector.conj().T,
        )

    hessenberg_matrix: np.ndarray = working_matrix
    return (
        householder_reflectors_list,
        hessenberg_matrix,
        accumulated_orthogonal_matrix,
    )
```

We will evaluate this function in @section_evaluating_the_function.

== Evaluating the Function (b), (c), (d)
<section_evaluating_the_function>

We present another algorithm for evaluating the function `to_hessenberg(A)` for random matrices of various sizes, inputed by the user, which also gets to choose if symmetric matrices will be generated or not.

```python
#RANDOM MATRIX GENERATOR
def generate_random_matrix(n:int, distribution:str="normal",
                           symmetric:bool=False, seed:int|None=None):
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        A = rng.standard_normal((n, n))
    elif distribution == "uniform":
        A = rng.uniform(-1.0, 1.0, size=(n, n))
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")
    return (A + A.T) / 2.0 if symmetric else A


#REFLECTOR CALCULATOR
def _house_vec(x:np.ndarray) -> np.ndarray:

    """
    Builds a Householder reflector for a given column vector x.
    Args:
        x (np.ndarray): Column vector to be transformed.
    Returns:
        np.ndarray: Normalised Householder vector with a real first component.
    Raises:
        None
    """

    sigma = np.linalg.norm(x)
    if sigma == 0.0:
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        return e1
    sign = 1.0 if x[0].real >= 0.0 else -1.0
    v = x.copy()
    v[0] += sign * sigma
    return v / np.linalg.norm(v)

def hessenberg_reduction(A_in:np.ndarray, symmetric:bool=False, accumulate_q:bool=True):

    """
    Reduces a matrix to upper Hessenberg form using Householder reflections.
    Args:
        A_in (np.ndarray): Input matrix to be reduced.
        symmetric (bool): If True, treat the matrix as symmetric and reduce to tridiagonal form.
        accumulate_q (bool): If True, accumulate the orthogonal matrix Q.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The reduced matrix in Hessenberg form and the orthogonal matrix Q.
    Raises:
        None
    """

    A = A_in.copy()
    n = A.shape[0]
    Q = np.eye(n, dtype=A.dtype)

    if not symmetric:    #GENERAL caSe
        for k in range(n-2):
            v = _house_vec(A[k+1:, k])
            w = np.zeros(n, dtype=A.dtype)
            w[k+1:] = v
            A -= 2.0 * np.outer(w, w.conj().T @ A)
            A -= 2.0 * np.outer(A @ w, w.conj().T)
            if accumulate_q:
                Q -= 2.0 * np.outer(Q @ w, w.conj().T)
        return A, Q

    #SYMMETRIC TRIDIAGONAL CASE
    for k in range(n-2):
        x = A[k+1:, k]
        v = _house_vec(x)
        beta = 2.0

        w = A[k+1:, k+1:] @ v   #trailing submatrix rank-2 update (A ← A − v wᵀ − w vᵀ)
        tau = beta * 0.5 * (v @ w)
        w -= tau * v
        A[k+1:, k+1:] -= beta * np.outer(v, w) + beta * np.outer(w, v)

        new_val = -np.sign(x[0]) * np.linalg.norm(x)   #store the single sub-diagonal element, zero the rest
        A[k+1, k] = new_val
        A[k, k+1] = new_val
        A[k+2:, k] = 0.0
        A[k, k+2:] = 0.0

        if accumulate_q:  #accumulate Q if requested
            Q[:, k+1:] -= beta * np.outer(Q[:, k+1:] @ v, v)

    A = np.triu(A) + np.triu(A, 1).T  #force symmetry
    return A, Q


#VERIFYING PART
def verify_factorisation_once(n:int, dist:str, symmetric:bool, seed:int|None):

    """
    Verifies the factorisation of a random matrix of size n.
    Args:
        n (int): Size of the matrix.
        dist (str): Distribution type ('normal' or 'uniform').
        symmetric (bool): Whether the matrix is symmetric.
        seed (int | None): Random seed for reproducibility.
    Returns:
        None
    Raises:
        None
    """

    A = generate_random_matrix(n, dist, symmetric, seed)
    T, Q = hessenberg_reduction(A, symmetric=symmetric)
    res_fact = np.linalg.norm(A - Q @ T @ Q.T)
    res_orth = np.linalg.norm(Q.T @ Q - np.eye(n))
    colour = "green" if res_fact < 1e-11 else "red"
    typ = "symmetric" if symmetric else "general"
    display(Markdown(
        f"**{n}×{n} {typ}**  \n"
        f"<span style='color:{colour}'>‖A − Q T Qᵀ‖ = {res_fact:.2e}</span>  \n"
        f"‖QᵀQ − I‖ = {res_orth:.2e}"
    ))


def benchmark_hessenberg(size_list, dist:str, mode:str, seed:int|None, reps_small:int=5):

    """
    Benchmark the Hessenberg reduction for various matrix sizes and types.
    Args:
        size_list (list of int): List of matrix sizes to test.
        dist (str): Distribution type ('normal' or 'uniform').
        mode (str): Matrix type ('general', 'symmetric', or 'both').
        seed (int | None): Random seed for reproducibility.
        reps_small (int): Number of repetitions for small matrices.
    Returns:
        pd.DataFrame: DataFrame containing the benchmark results.
    Raises:
        None
    """

    records = []
    for n in size_list:
        for sym in ([False, True] if mode=="both" else [mode=="symmetric"]):
            A = generate_random_matrix(n, dist, sym, seed)

            t0 = time.perf_counter()
            hessenberg_reduction(A, symmetric=sym, accumulate_q=False)
            probe = time.perf_counter() - t0
            reps = reps_small if probe*reps_small >= 1.0 else math.ceil(1.0 / probe)

            times = []
            for _ in range(reps):
                start = time.perf_counter()
                hessenberg_reduction(A, symmetric=sym, accumulate_q=False)
                times.append(time.perf_counter() - start)

            records.append(dict(size=n,
                                type="symmetric" if sym else "general",
                                reps=reps,
                                avg=np.mean(times)))

    df = pd.DataFrame(records)
    display(df.style.format({"avg":"{:.3e}"}).hide(axis="index"))

    plt.figure(figsize=(7, 5))

    mark = {"general": "o", "symmetric": "s"}
    for label, sub in df.groupby("type"):
        # simple scatter/line plot – no regression curves
        plt.plot(
            sub["size"],           # x-axis: matrix order
            sub["avg"],            # y-axis: average runtime
            marker=mark[label],
            ls="-",
            label=label,
        )

    plt.xlabel("matrix size  (linear)")
    plt.ylabel("runtime [s]  (linear)")       # keep or change to log scale as you prefer
    plt.title("Hessenberg (general)  vs  Tridiagonal (symmetric)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()



# === INTERACTIVE PART ======================================================


def parse_size_spec(spec: str):
    """
    Accepts either   '[64,128,256,512]'   or   '64:1024:64'
    Returns a sorted list of unique integers ≥ 2
    """
    spec = spec.strip()
    interval = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*:\s*(\d+)\s*", spec)
    if interval:                         # range syntax
        lo, hi, step = map(int, interval.groups())
        if step <= 0 or lo < 2 or hi < lo:
            raise ValueError
        return list(range(lo, hi + 1, step))
    # otherwise fall back to literal - must be a list
    sizes = literal_eval(spec)
    if (not isinstance(sizes, (list, tuple)) or
            any((not isinstance(k, int)) or k < 2 for k in sizes)):
        raise ValueError
    return sorted(set(sizes))

try:
    raw = input(
        "\nMatrix sizes – list '[64,128,256]'  or  interval '64:1024:64' "
        "(default 64:1024:64): "
    )
    sizes = parse_size_spec(raw) if raw else list(range(64, 1025, 64))
except Exception:
    print("Bad specification → using default 64:1024:64.")
    sizes = list(range(64, 1025, 64))

dist = input("Distribution ('normal'/'uniform')  [normal]: ").strip().lower() or "normal"
mode_txt = input("Matrix type g=general, s=symmetric, b=both  [g]: ").strip().lower() or "g"
mode = "symmetric" if mode_txt == "s" else "both" if mode_txt == "b" else "general"
seed_txt = input("Random seed (None/int) [None]: ").strip()
seed_val = None if seed_txt.lower() in {"", "none"} else int(seed_txt)

# accuracy check
for n in sizes:
    for sym in ([False, True] if mode == "both" else [mode == "symmetric"]):
        verify_factorisation_once(n, dist, sym, seed_val)

benchmark_hessenberg(sizes, dist, mode, seed_val)  # timings
```

The reader should be aware that my poor #link("https://www.dell.com/support/manuals/pt-br/inspiron-15-5590-laptop/inspiron-5590-setup-and-specifications/specifications-of-inspiron-5590?guid=guid-7c9f07ce-626e-44ca-be3a-a1fb036413f9&lang=en-us")[Dell Inspiron 5590] has crashed precisely $5$ times while i was writing this (i might have tried with matrices of order $10^6 times 10^6$). Unfortunately the runtime was around $4$ minutes for a matrix $A approx 10^3 times 10^3$.

An expected output is:

```python
64×64 general
‖A − Q T Qᵀ‖ = 8.03e-14
‖QᵀQ − I‖ = 7.67e-15

64×64 symmetric
‖A − Q T Qᵀ‖ = 4.78e-14
‖QᵀQ − I‖ = 7.32e-15

128×128 general
‖A − Q T Qᵀ‖ = 1.80e-13
‖QᵀQ − I‖ = 1.24e-14

128×128 symmetric
‖A − Q T Qᵀ‖ = 1.13e-13
‖QᵀQ − I‖ = 1.25e-14

192×192 general
‖A − Q T Qᵀ‖ = 3.09e-13
‖QᵀQ − I‖ = 1.76e-14

192×192 symmetric
‖A − Q T Qᵀ‖ = 1.91e-13
‖QᵀQ − I‖ = 1.75e-14

256×256 general
‖A − Q T Qᵀ‖ = 4.53e-13
‖QᵀQ − I‖ = 2.27e-14

256×256 symmetric
‖A − Q T Qᵀ‖ = 2.97e-13
‖QᵀQ − I‖ = 2.38e-14

320×320 general
‖A − Q T Qᵀ‖ = 6.07e-13
‖QᵀQ − I‖ = 2.72e-14

320×320 symmetric
‖A − Q T Qᵀ‖ = 3.76e-13
‖QᵀQ − I‖ = 2.71e-14

384×384 general
‖A − Q T Qᵀ‖ = 7.78e-13
‖QᵀQ − I‖ = 3.11e-14

384×384 symmetric
‖A − Q T Qᵀ‖ = 4.71e-13
‖QᵀQ − I‖ = 3.16e-14

448×448 general
‖A − Q T Qᵀ‖ = 9.35e-13
‖QᵀQ − I‖ = 3.50e-14

448×448 symmetric
‖A − Q T Qᵀ‖ = 5.93e-13
‖QᵀQ − I‖ = 3.72e-14

512×512 general
‖A − Q T Qᵀ‖ = 1.16e-12
‖QᵀQ − I‖ = 4.13e-14

512×512 symmetric
‖A − Q T Qᵀ‖ = 7.14e-13
‖QᵀQ − I‖ = 4.13e-14

576×576 general
‖A − Q T Qᵀ‖ = 1.38e-12
‖QᵀQ − I‖ = 4.54e-14

576×576 symmetric
‖A − Q T Qᵀ‖ = 8.39e-13
‖QᵀQ − I‖ = 4.56e-14

640×640 general
‖A − Q T Qᵀ‖ = 1.58e-12
‖QᵀQ − I‖ = 4.93e-14

640×640 symmetric
‖A − Q T Qᵀ‖ = 9.77e-13
‖QᵀQ − I‖ = 5.07e-14

704×704 general
‖A − Q T Qᵀ‖ = 1.81e-12
‖QᵀQ − I‖ = 5.41e-14

704×704 symmetric
‖A − Q T Qᵀ‖ = 1.08e-12
‖QᵀQ − I‖ = 5.35e-14

768×768 general
‖A − Q T Qᵀ‖ = 2.05e-12
‖QᵀQ − I‖ = 5.92e-14

768×768 symmetric
‖A − Q T Qᵀ‖ = 1.25e-12
‖QᵀQ − I‖ = 5.98e-14

832×832 general
‖A − Q T Qᵀ‖ = 2.29e-12
‖QᵀQ − I‖ = 6.32e-14

832×832 symmetric
‖A − Q T Qᵀ‖ = 1.38e-12
‖QᵀQ − I‖ = 6.28e-14

896×896 general
‖A − Q T Qᵀ‖ = 2.53e-12
‖QᵀQ − I‖ = 6.71e-14

896×896 symmetric
‖A − Q T Qᵀ‖ = 1.50e-12
‖QᵀQ − I‖ = 6.65e-14

960×960 general
‖A − Q T Qᵀ‖ = 2.78e-12
‖QᵀQ − I‖ = 7.14e-14

960×960 symmetric
‖A − Q T Qᵀ‖ = 1.68e-12
‖QᵀQ − I‖ = 7.19e-14

1024×1024 general
‖A − Q T Qᵀ‖ = 3.09e-12
‖QᵀQ − I‖ = 7.71e-14

1024×1024 symmetric
‖A − Q T Qᵀ‖ = 1.84e-12
‖QᵀQ − I‖ = 7.53e-14


```
As $n$ grows, we observe that the residuals also grow, but still in machine precision. The difference between the symmetric and nonsymmetric cases are more pronounced in larger matrices.


#figure(
    image("images/plot_hessenberg_function_evaluation.png", width: 100%),
    caption: [
        Runtime of the Hessenberg reduction for ordinary and symmetric matrices
    ]
) <figure_plot_evaluation_hessenberg_function>

=== Complexity (c)
<section_complexity>

@figure_plot_evaluation_hessenberg_function shows the expected $O(n^3)$ complexity for the general case and usually half of it for the symmetric case. The latter is better discussed in @section_symmetric_case.

The weird behavior of the general case after $n approx 760$ is due purely to hardware restrictions, not mathematical ones. #link("https://numpy.org/devdocs/building/blas_lapack.html")[NumPy] delegates the heavy multiply-add to OpenBLAS / MKL. Those libraries switch to different blocking sizes and sometimes to multi-threaded kernels at dimension “milestones” (often multiples of $64$ or $128$). The change in algorithmic constant shows up as local bumps or dips. Residual noise comes from thread scheduling and page-fault variability in long runs.

To understand why the complexity is $O(n^3)$ in the general case, we can look at the algorithm. The outer loop runs $n - 2$ times, and inside it, we have two matrix-vector products and two outer products, which are all $O(n^2)$. Thus, the total complexity is $O(n^3)$.

=== The Symmetric Case (d)
<section_symmetric_case>

On the symmetric case we know that reflectors will be applied in only one side of the matrix, since $transpose(v) A = transpose(A v)$. That is precisely what the function `generate_random_matrix` does. Which cuts complexity from the expected $O(n^3)$ seen in the previous section to a $O(n^2)$ #footnote[See page 194 of #link("https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf")[Trefethen & Bau's Numerical Linear Algebra book]]. @trefethen

= Eigenvalues and Iterative Methods
<section_eigenvalues_and_iterative_methods>
== Power iteration
<section_power_iteration>

The power iteration consists on computing large powers of the sequence:

$
  x / norm(x) , (A x) / norm(A x) , (A^2 x) / norm(A^2 x), dots, A in CC^(m times m)
$

To see why this sequence converges(under good assumptions), let $A$ be diagonalizable. And write:

$
  x = sum_(i = 1)^m phi_i v_i
$

In a basis of eigenvectors $v_i$ with respective eigenvalues $lambda_i$. Then for $x in CC^m$ we have:

$
  A x = sum_(i = 1)^m lambda_i phi_i v_i
$

Or even better:

$
  A^n x = sum_(i = 1)^m lambda_i^n phi_i v_i
$

Let $v_j$ be the eigenvector associated to the biggest eigenvalue $lambda_j$, then we have:

$
  A^n x = 1 / lambda_j^n dot sum_(i = 1)^m lambda_i^n phi_i v_i = (lambda_1^n) / (lambda_j^n) phi_1 v_1 + dots + phi_j v_j + dots + lambda_m^n / (lambda_j^n) phi_m v_m
$

When $n -> oo$ all of the smaller $lambda_k / lambda_j$ will approach $0$, so we have:

$
  lim_(n -> oo) A^n x = phi_j v_j 
$

So the denominator on the original expression becomes

$
  norm(A^n x) = norm(phi_j v_j) = abs(phi_j) norm(v_j)
$

And the limit is:

$
  lim_(n -> oo) (A^n x) / norm(A^n x) = (phi_j v_j) / (abs(phi_j) norm(v_j)) 
$

Since $phi_j / abs(phi_j) = plus.minus 1$, the sequence converges to the eigenvector $v_j$ associated to the eigenvalue $lambda_j$.

== Inverse Iteration
<section_inverse_iteration>

We know that if $A in CC^(m times m), det(A) != 0$ has $Lambda subset RR := {lambda_j}$ as eigenvalues, then the eigenvalues of $inv(A, 1)$ are $inv(lambda_j, 1)$. Similarly the eigenvalues of $A + phi I$ are $lambda_j + phi$. So the eigenvalues of $inv((A - mu I), 1), mu in RR without Lambda$ are:

$
  hat(lambda_j) = 1 / (lambda_j - mu)
$

Where $lambda_j$ are the eigenvalues of $A$. So if $mu$ is close to an eigenvalue, then $hat(lambda)$ will be large. Power iteration seems interesting here, so the sequence:

$
  x / norm(x), (inv((A - mu I), 1) x) / norm(inv((A - mu I), 1) x), (inv((A - mu I), 2) x) / norm(inv((A - mu I), 2) x), dots
$

Converges to the eigenvector associated to the eigenvalue $hat(lambda)$.


= Orthogonal Matrices (Problem 2) (a)
<section_orthogonal_matrices>

Here we will discuss how orthogonal matrices behave when we appluy the iterations discussed in @section_power_iteration, and @section_inverse_iteration.

So let $Q in CC^(m times m)$ be an orthogonal matrix. We are interested in its eigenvalues $lambda$. We know that:

$
  Q x = lambda x <=> transpose(x) Q x = lambda transpose(x) x\

  <=> Q inner(x, x) = lambda inner(x, x)\
$

Since $Q$ preserves inner product, we have:

$
  Q inner(x, x) = lambda inner(x, x) <=> inner(x, x) = lambda inner(x, x)\

  <=> abs(lambda) = 1
$

So $lambda$ lies in the unit circle, i.e $lambda = e^(i phi), phi in RR$. We now discuss how this affects efficiency of some iterative methods

== Orthogonal Matrices and the Power Iteration
<section_orthogonal_matrices_and_power_iteration>

The power method is better discussed in @section_power_iteration. Here we will write straight forward the result:

$
  Q^n x = 1 / lambda_j^n dot sum_(i = 1)^m lambda_i^n phi_i v_i
$ <equation_orthogonal_power_iteration>

Where $lambda_i$ are the eigenvalues of $Q in CC^(m times m)$, $phi_i$ are the coefficients of the expansion of $x$ in the basis of eigenvectors $v_i$. Since we have that $abs(lambda_i) = 1$, we have:

The fact that $abs(lambda_i) = 1 => abs(lambda_i^n) = 1$ is sufficiently enough for one to be convinced that power iteration does not converge.

Let $lambda_k = e^(i psi_k)$, where $psi_k in RR$. Then expanding @equation_orthogonal_power_iteration:

$
    Q^n x = 1 / e^(i psi_j dot n) dot sum_(tau = 1)^m e^(i psi_tau n) phi_tau v_tau
$

When $n -> oo$ if $lambda_j = 1$ then we have:

$
  Q^n x = phi_j v_j + sum_(tau != j) e^(i psi_tau n) phi_tau v_tau
$

Since no eigenvalue dominates other eigenvalues in the orthogonal case, usually power iteration fails.

== Orthogonal Matrices and Inverse Iteration
<section_orthogonal_matrices_and_inverse_iteration>

Let $Q in CC^(m times m)$ be an orthogonal matrix. Using results from @section_inverse_iteration, we are interested in applying invers iteration to $Q$.

We know that the eigenvalues of $Q$ are on the unit circle, so if $mu$ is close to an eigenvalue $lambda_j$, $hat(lambda)_j$ will be huge (dominant), which makes power iteration converge to the eigenvector associated to $hat(lambda)_j$, which is the eigenvector associated to $lambda_j$. The fact that the eigenvalues are on the unit circle also contributes to the convergence of the method.

So we concude that inverse iteration works well for orthogonal matrices, _if $mu$ is close to an eigenvalue of $Q$_.

== The *$2 times 2$* Case (b)
<section_2x2_case>

We will calculate the eigenvalues of:

$
  A = mat(a, b; c, d)
$

With $a, b, c, d in RR$. The characteristic polynomial gives us:

$
  det(A - lambda I) = 0 <=> det mat(
    a - lambda, b;
    c, d - lambda
  ) = 0\

  <=> (a - lambda)(d - lambda) - b c = 0
  <=> lambda^2 + lambda(-a -d) + (a d - b c) = 0\
  <=> lambda = (a + d) plus.minus sqrt((a + d)^2 - 4(a d - b c)) / 2
$

So the eigenvalues are:

$
  lambda_1 = (a + d + sqrt((a + d)^2 - 4(a d - b c))) / 2\

  lambda_2 = (a + d - sqrt((a + d)^2 - 4(a d - b c))) / 2
$


== Random Orthogonal Matrices (c)
<section_random_orthogonal_matrices>

This code generates orthogonal matrices of order $4 times 4$ generated by the $Q R$ factorization of random matrices, and reduces the to Hessenberg form. The eigenvalues of the bottom-right $2 times 2$ block are analytically calculated using @section_2x2_case.

```python
def generate_orthogonal_matrix_qr(n=4, seed=None):

    """
    Generates a random orthogonal matrix using QR decomposition.
    Args:
        n (int): Size of the matrix (n x n).
        seed (int | None): Random seed for reproducibility.
    Returns:
        np.ndarray: An n x n orthogonal matrix.
    Raises:
        None
    """

    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q

def analytical_eigenvalues_2x2(a, b, c, d):

    """
    Calculates the eigenvalues of a 2x2 matrix analytically.
    Args:
        a (float): Element at position (0,0).
        b (float): Element at position (0,1).
        c (float): Element at position (1,0).
        d (float): Element at position (1,1).
    Returns:
        Tuple[float, float]: The two eigenvalues of the matrix.
    Raises:
    """

    trace = a + d
    det = a * d - b * c
    discriminant = trace**2 - 4 * det
    
    #complex if discriminant negative
    discriminant_root = np.sqrt(discriminant) if discriminant >= 0 else np.sqrt(complex(discriminant))
    
    lambda1 = (trace + discriminant_root) / 2
    lambda2 = (trace - discriminant_root) / 2
    
    return lambda1, lambda2

def analyze_orthogonal_and_hessenberg(n=4, n_matrices=30):

    """
    Analyzes orthogonal matrices and their Hessenberg forms.
    Args:
        n (int): Size of the matrices (n x n).
        n_matrices (int): Number of orthogonal matrices to generate and analyze.
    Returns:
        None
    Raises:
        None
    """

    for i in range(n_matrices):
        print(f"\n--- Orthogonal Matrix Q number {i+1} ---")
        Q = generate_orthogonal_matrix_qr(n=n)
        print("Matrix Q:")
        print(np.array_str(Q, precision=4, suppress_small=True))
        
        householder_list, H, Q_accum = to_hessenberg(Q)
        
        print("\nHessenberg Form H (of Q):")
        print(np.array_str(H, precision=4, suppress_small=True))
        
        block = Q[2:4, 2:4]
        a, b, c, d = block[0,0], block[0,1], block[1,0], block[1,1]
        analytical_eigenvalues = analytical_eigenvalues_2x2(a, b, c, d)
        
        print("\nBlock Q[3:4,3:4] (indices 2 and 3, 2x2):")
        
        print(np.array_str(block, precision=4, suppress_small=True))
        
        print("\nEigenvalues of the 2x2 block (analytically calculated):")
        for idx, val in enumerate(analytical_eigenvalues):
            print(f"  λ_{idx+1} = {val} (size = {abs(val):.4f})")
        
        print("-" * 40)

analyze_orthogonal_and_hessenberg()

```
We ran this code for $30$ matrices, the output was:

```
--- Orthogonal Matrix Q number 1 ---
Matrix Q:
[[-0.5629  0.6801  0.4635 -0.0764]
 [-0.6703 -0.6884  0.2231  0.1645]
 [-0.1497  0.2337 -0.3793  0.8827]
 [ 0.4598 -0.0949  0.7691  0.4336]]

Hessenberg Form H (of Q):
[[-0.5629 -0.6779 -0.4534 -0.1342]
 [ 0.8265 -0.4617 -0.3088 -0.0914]
 [ 0.     -0.5721  0.7865  0.2329]
 [-0.      0.      0.2839 -0.9588]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.3793  0.8827]
 [ 0.7691  0.4336]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.9458819605346136 (size = 0.9459)
  λ_2 = -0.8915812804514585 (size = 0.8916)
----------------------------------------

--- Orthogonal Matrix Q number 2 ---
Matrix Q:
[[-0.4016  0.3605  0.8354  0.1045]
 [-0.2846 -0.3518  0.1255 -0.8829]
 [-0.3325 -0.8166  0.136   0.4519]
 [ 0.8045 -0.282   0.5176 -0.0734]]

Hessenberg Form H (of Q):
[[-0.4016 -0.3235 -0.1952  0.8343]
 [ 0.9158 -0.1418 -0.0856  0.3658]
 [ 0.     -0.9355  0.0805 -0.3439]
 [ 0.      0.     -0.9737 -0.2278]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[ 0.136   0.4519]
 [ 0.5176 -0.0734]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.5261528266779573 (size = 0.5262)
  λ_2 = -0.4635182526918817 (size = 0.4635)
----------------------------------------

--- Orthogonal Matrix Q number 3 ---
Matrix Q:
[[-0.0452 -0.9852 -0.1571  0.0523]
 [ 0.4259  0.0412 -0.0809  0.9002]
 [ 0.1843  0.1346 -0.9569 -0.1793]
 [-0.8846  0.0982 -0.2303  0.3934]]

Hessenberg Form H (of Q):
[[-0.0452  0.4954 -0.8604 -0.1112]
 [-0.999  -0.0224  0.0389  0.005 ]
 [ 0.      0.8684  0.4918  0.0636]
 [-0.      0.      0.1282 -0.9917]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.9569 -0.1793]
 [-0.2303  0.3934]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.42331102317930497 (size = 0.4233)
  λ_2 = -0.9868680130188092 (size = 0.9869)
----------------------------------------

--- Orthogonal Matrix Q number 4 ---
Matrix Q:
[[-0.2568  0.8228 -0.2287  0.4525]
 [-0.2848 -0.018   0.9011  0.3265]
 [-0.6054 -0.5408 -0.3667  0.4544]
 [ 0.6975 -0.1738 -0.0346  0.6943]]

Hessenberg Form H (of Q):
[[-0.2568  0.2274  0.1895  0.92  ]
 [ 0.9665  0.0604  0.0503  0.2444]
 [-0.     -0.9719  0.0475  0.2305]
 [ 0.      0.     -0.9794  0.2017]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.3667  0.4544]
 [-0.0346  0.6943]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.67929095950588 (size = 0.6793)
  λ_2 = -0.3516952691053273 (size = 0.3517)
----------------------------------------

--- Orthogonal Matrix Q number 5 ---
Matrix Q:
[[-0.5025 -0.7649 -0.0345  0.4015]
 [ 0.3718  0.0654  0.6626  0.6469]
 [ 0.6819 -0.6372  0.0297 -0.358 ]
 [ 0.3798  0.0679 -0.7476  0.5406]]

Hessenberg Form H (of Q):
[[-0.5025  0.1798 -0.7903  0.3011]
 [-0.8646 -0.1045  0.4593 -0.175 ]
 [-0.     -0.9781 -0.1943  0.074 ]
 [-0.      0.      0.356   0.9345]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[ 0.0297 -0.358 ]
 [-0.7476  0.5406]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.8621172090206812 (size = 0.8621)
  λ_2 = -0.2918043055687003 (size = 0.2918)
----------------------------------------

--- Orthogonal Matrix Q number 6 ---
Matrix Q:
[[-0.9092  0.1745  0.2312 -0.2992]
 [ 0.1618 -0.3496 -0.2497 -0.8884]
 [-0.1454  0.4629 -0.8736  0.0369]
 [-0.3551 -0.7956 -0.3479  0.3462]]

Hessenberg Form H (of Q):
[[-0.9092 -0.2422 -0.3011 -0.1552]
 [-0.4164  0.5288  0.6573  0.3389]
 [ 0.      0.8134 -0.517  -0.2665]
 [-0.      0.      0.4582 -0.8888]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.8736  0.0369]
 [-0.3479  0.3462]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.3356278613353456 (size = 0.3356)
  λ_2 = -0.8630034423078552 (size = 0.8630)
----------------------------------------

--- Orthogonal Matrix Q number 7 ---
Matrix Q:
[[-0.4296  0.6844 -0.4085  0.4245]
 [ 0.2184 -0.4244 -0.028   0.8783]
 [ 0.17   -0.3028 -0.9122 -0.2177]
 [ 0.8596  0.5097 -0.0167  0.032 ]]

Hessenberg Form H (of Q):
[[-0.4296 -0.4928  0.7535  0.0703]
 [-0.903   0.2344 -0.3584 -0.0334]
 [-0.     -0.838  -0.5433 -0.0507]
 [ 0.      0.      0.0928 -0.9957]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.9122 -0.2177]
 [-0.0167  0.032 ]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.0358294023777706 (size = 0.0358)
  λ_2 = -0.9159948591320213 (size = 0.9160)
----------------------------------------

--- Orthogonal Matrix Q number 8 ---
Matrix Q:
[[-0.3402 -0.0284 -0.009   0.9399]
 [-0.7148 -0.6207  0.1662 -0.2759]
 [-0.5681  0.5973 -0.5323 -0.1927]
 [ 0.2249 -0.5071 -0.83    0.0581]]

Hessenberg Form H (of Q):
[[-0.3402  0.2518  0.8476  0.3201]
 [ 0.9403  0.0911  0.3067  0.1158]
 [ 0.      0.9635 -0.2505 -0.0946]
 [-0.     -0.      0.3533 -0.9355]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.5323 -0.1927]
 [-0.83    0.0581]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.25997158900622225 (size = 0.2600)
  λ_2 = -0.7341829044352151 (size = 0.7342)
----------------------------------------

--- Orthogonal Matrix Q number 9 ---
Matrix Q:
[[-0.2692  0.5878 -0.5428  0.5361]
 [-0.8282  0.0254  0.5486  0.1116]
 [-0.292   0.3638 -0.2875 -0.8365]
 [-0.3954 -0.7222 -0.5673  0.0189]]

Hessenberg Form H (of Q):
[[-0.2692 -0.561  -0.1619  0.7659]
 [ 0.9631 -0.1568 -0.0453  0.2141]
 [-0.     -0.8128  0.1205 -0.5699]
 [ 0.     -0.     -0.9784 -0.2069]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2875 -0.8365]
 [-0.5673  0.0189]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.5713848430035342 (size = 0.5714)
  λ_2 = -0.8399942790872519 (size = 0.8400)
----------------------------------------

--- Orthogonal Matrix Q number 10 ---
Matrix Q:
[[-0.4241  0.1349 -0.1773  0.8778]
 [-0.332  -0.8664 -0.3594 -0.0999]
 [ 0.8422 -0.2846 -0.2058  0.409 ]
 [-0.0237 -0.3876  0.8928  0.2284]]

Hessenberg Form H (of Q):
[[-0.4241 -0.2373  0.8677  0.1046]
 [ 0.9056 -0.1111  0.4063  0.049 ]
 [-0.      0.9651  0.2602  0.0313]
 [ 0.      0.      0.1196 -0.9928]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2058  0.409 ]
 [ 0.8928  0.2284]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.6533894082311485 (size = 0.6534)
  λ_2 = -0.6307995911677337 (size = 0.6308)
----------------------------------------

--- Orthogonal Matrix Q number 11 ---
Matrix Q:
[[-0.1675 -0.9786 -0.0461 -0.1099]
 [ 0.7871 -0.1742  0.5819  0.1072]
 [ 0.3816 -0.1052 -0.6644  0.6339]
 [-0.4547 -0.0293  0.4667  0.758 ]]

Hessenberg Form H (of Q):
[[-0.1675  0.7485 -0.1241 -0.6295]
 [-0.9859 -0.1271  0.0211  0.1069]
 [ 0.     -0.6508 -0.1468 -0.7449]
 [ 0.     -0.     -0.9811  0.1934]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.6644  0.6339]
 [ 0.4667  0.758 ]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.9421402114923986 (size = 0.9421)
  λ_2 = -0.8485799008471894 (size = 0.8486)
----------------------------------------

--- Orthogonal Matrix Q number 12 ---
Matrix Q:
[[-0.3541  0.5585 -0.0616  0.7476]
 [-0.5108 -0.6958 -0.4435  0.2414]
 [ 0.1705  0.3507 -0.885  -0.2542]
 [ 0.7646 -0.2844 -0.1274  0.5641]]

Hessenberg Form H (of Q):
[[-0.3541  0.295   0.7613  0.4561]
 [ 0.9352  0.1117  0.2883  0.1727]
 [-0.      0.949  -0.2706 -0.1621]
 [-0.     -0.      0.5139 -0.8579]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.885  -0.2542]
 [-0.1274  0.5641]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.5861402323977607 (size = 0.5861)
  λ_2 = -0.9070552039589896 (size = 0.9071)
----------------------------------------

--- Orthogonal Matrix Q number 13 ---
Matrix Q:
[[-0.7015  0.4915  0.3042 -0.417 ]
 [ 0.4623 -0.1626  0.1522 -0.8583]
 [ 0.2479  0.6778 -0.6824 -0.1159]
 [-0.4825 -0.5221 -0.647  -0.2757]]

Hessenberg Form H (of Q):
[[-0.7015 -0.7069 -0.0829 -0.0373]
 [-0.7127  0.6958  0.0816  0.0367]
 [ 0.      0.1276 -0.9045 -0.4069]
 [ 0.     -0.      0.4103 -0.912 ]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.6824 -0.1159]
 [-0.647  -0.2757]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = -0.1379150700589457 (size = 0.1379)
  λ_2 = -0.820164584090632 (size = 0.8202)
----------------------------------------

--- Orthogonal Matrix Q number 14 ---
Matrix Q:
[[-0.2164 -0.5735  0.7758 -0.1498]
 [-0.3581 -0.6722 -0.6278 -0.1609]
 [-0.1535  0.3054  0.0015 -0.9398]
 [-0.8952  0.3551  0.0633  0.2617]]

Hessenberg Form H (of Q):
[[-0.2164  0.2256 -0.3541  0.8814]
 [ 0.9763  0.05   -0.0785  0.1954]
 [-0.     -0.9729 -0.0862  0.2144]
 [-0.      0.     -0.9279 -0.3728]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[ 0.0015 -0.9398]
 [ 0.0633  0.2617]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = (0.1316169167334142+0.20628124638940884j) (size = 0.2447)
  λ_2 = (0.1316169167334142-0.20628124638940884j) (size = 0.2447)
----------------------------------------

--- Orthogonal Matrix Q number 15 ---
Matrix Q:
[[-0.467   0.1387  0.7316  0.4769]
 [ 0.0375 -0.9757  0.0774  0.2017]
 [-0.7384 -0.0272 -0.627   0.2467]
 [ 0.485   0.1677 -0.2562  0.8191]]

Hessenberg Form H (of Q):
[[-0.467   0.3435 -0.8022 -0.1428]
 [-0.8843 -0.1814  0.4236  0.0754]
 [-0.      0.9215  0.3824  0.068 ]
 [-0.     -0.      0.1752 -0.9845]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.627   0.2467]
 [-0.2562  0.8191]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.7740235941481243 (size = 0.7740)
  λ_2 = -0.5818796203145907 (size = 0.5819)
----------------------------------------

--- Orthogonal Matrix Q number 16 ---
Matrix Q:
[[-0.3849  0.0435 -0.7406  0.549 ]
 [ 0.2525 -0.952  -0.07    0.158 ]
 [ 0.7807  0.2969  0.0191  0.5495]
 [-0.4225 -0.06    0.668   0.6096]]

Hessenberg Form H (of Q):
[[-0.3849  0.866   0.1952  0.2527]
 [-0.923  -0.3611 -0.0814 -0.1054]
 [ 0.      0.346  -0.5736 -0.7425]
 [ 0.      0.     -0.7914  0.6114]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[0.0191 0.5495]
 [0.668  0.6096]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.9883202000131572 (size = 0.9883)
  λ_2 = -0.3596282457787697 (size = 0.3596)
----------------------------------------

--- Orthogonal Matrix Q number 17 ---
Matrix Q:
[[-0.1066  0.7212  0.0749 -0.6804]
 [-0.6764 -0.5121  0.3485 -0.3985]
 [-0.5738  0.1159 -0.8012  0.1245]
 [-0.4493  0.4519  0.4807  0.6024]]

Hessenberg Form H (of Q):
[[-0.1066 -0.2265 -0.8608 -0.4431]
 [ 0.9943 -0.0243 -0.0923 -0.0475]
 [ 0.     -0.9737  0.2025  0.1043]
 [ 0.      0.      0.4577 -0.8891]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.8012  0.1245]
 [ 0.4807  0.6024]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.6437814804055579 (size = 0.6438)
  λ_2 = -0.842579679233141 (size = 0.8426)
----------------------------------------

--- Orthogonal Matrix Q number 18 ---
Matrix Q:
[[-0.8255  0.1636  0.5347  0.0763]
 [-0.5177 -0.2699 -0.6467 -0.4908]
 [ 0.0194  0.9433 -0.2236 -0.2446]
 [-0.2238  0.103  -0.4958  0.8327]]

Hessenberg Form H (of Q):
[[-0.8255 -0.1619 -0.4367 -0.3188]
 [ 0.5644 -0.2368 -0.6388 -0.4663]
 [ 0.      0.958  -0.2317 -0.1691]
 [-0.      0.     -0.5896  0.8077]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2236 -0.2446]
 [-0.4958  0.8327]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.9372008193289816 (size = 0.9372)
  λ_2 = -0.32809878403400955 (size = 0.3281)
----------------------------------------

--- Orthogonal Matrix Q number 19 ---
Matrix Q:
[[-0.7885  0.0127  0.4922 -0.3686]
 [ 0.5982 -0.2155  0.6311 -0.4444]
 [ 0.0328  0.1282 -0.5621 -0.8164]
 [-0.1391 -0.968  -0.2085 -0.0141]]

Hessenberg Form H (of Q):
[[-0.7885 -0.122  -0.4769  0.3687]
 [-0.615   0.1564  0.6114 -0.4727]
 [ 0.      0.9801 -0.1569  0.1213]
 [ 0.      0.     -0.6116 -0.7911]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.5621 -0.8164]
 [-0.2085 -0.0141]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.2071704757269035 (size = 0.2072)
  λ_2 = -0.7833862909149965 (size = 0.7834)
----------------------------------------

--- Orthogonal Matrix Q number 20 ---
Matrix Q:
[[-0.3031  0.0429  0.9481  0.0861]
 [ 0.1571 -0.9342  0.1195 -0.2972]
 [-0.8533 -0.0239 -0.2292 -0.4678]
 [-0.3942 -0.3534 -0.1853  0.8279]]

Hessenberg Form H (of Q):
[[-0.3031  0.8775 -0.2647  0.261 ]
 [-0.953  -0.2791  0.0842 -0.083 ]
 [ 0.      0.3901  0.6557 -0.6464]
 [ 0.     -0.     -0.702  -0.7121]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2292 -0.4678]
 [-0.1853  0.8279]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.9043475415351031 (size = 0.9043)
  λ_2 = -0.30565758703551016 (size = 0.3057)
----------------------------------------

--- Orthogonal Matrix Q number 21 ---
Matrix Q:
[[-0.8149  0.3649 -0.4406 -0.0928]
 [ 0.0977 -0.6065 -0.7405  0.2726]
 [ 0.0387  0.372   0.0415  0.9265]
 [-0.57   -0.6005  0.5058  0.2423]]

Hessenberg Form H (of Q):
[[-0.8149 -0.1234  0.2317  0.5167]
 [-0.5796  0.1735 -0.3257 -0.7266]
 [ 0.     -0.9771 -0.0871 -0.1943]
 [-0.      0.      0.9125 -0.4091]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[0.0415 0.9265]
 [0.5058 0.2423]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.8337542266557703 (size = 0.8338)
  λ_2 = -0.5499977130761522 (size = 0.5500)
----------------------------------------

--- Orthogonal Matrix Q number 22 ---
Matrix Q:
[[-0.3663  0.4189 -0.6842 -0.4714]
 [ 0.2919 -0.672  -0.6656  0.142 ]
 [ 0.5024  0.6076 -0.2758  0.5498]
 [-0.7268 -0.061  -0.1131  0.6748]]

Hessenberg Form H (of Q):
[[-0.3663 -0.1301  0.2621  0.8833]
 [-0.9305  0.0512 -0.1032 -0.3477]
 [-0.     -0.9902 -0.0398 -0.134 ]
 [ 0.     -0.      0.9587 -0.2845]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2758  0.5498]
 [-0.1131  0.6748]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.6040601009013424 (size = 0.6041)
  λ_2 = -0.20508048336653284 (size = 0.2051)
----------------------------------------

--- Orthogonal Matrix Q number 23 ---
Matrix Q:
[[-0.4467 -0.7825 -0.2287  0.3687]
 [ 0.4222  0.182  -0.0157  0.8879]
 [ 0.6922 -0.3543 -0.5694 -0.2666]
 [-0.3783  0.4786 -0.7895  0.0678]]

Hessenberg Form H (of Q):
[[-0.4467  0.702  -0.0795 -0.5489]
 [-0.8947 -0.3505  0.0397  0.274 ]
 [ 0.     -0.62   -0.1125 -0.7765]
 [ 0.      0.     -0.9897  0.1433]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.5694 -0.2666]
 [-0.7895  0.0678]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.3077269347022792 (size = 0.3077)
  λ_2 = -0.8093108181365211 (size = 0.8093)
----------------------------------------

--- Orthogonal Matrix Q number 24 ---
Matrix Q:
[[-0.0413 -0.9591  0.2707  0.0721]
 [ 0.08   -0.274  -0.9584  0.0005]
 [-0.6777 -0.0389 -0.0459 -0.7328]
 [-0.7298  0.0604 -0.0778  0.6766]]

Hessenberg Form H (of Q):
[[-0.0413  0.3131  0.5108 -0.7996]
 [-0.9991 -0.0129 -0.0211  0.0331]
 [-0.      0.9496 -0.1687  0.2641]
 [ 0.      0.      0.8427  0.5384]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.0459 -0.7328]
 [-0.0778  0.6766]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.7483535886136121 (size = 0.7484)
  λ_2 = -0.1176716985715956 (size = 0.1177)
----------------------------------------

--- Orthogonal Matrix Q number 25 ---
Matrix Q:
[[-0.7483  0.5153 -0.0763  0.4107]
 [-0.4762 -0.7375 -0.4779 -0.0313]
 [-0.1713  0.3719 -0.3481 -0.8433]
 [ 0.4288  0.2287 -0.8029  0.3451]]

Hessenberg Form H (of Q):
[[-0.7483 -0.0847  0.4863  0.4431]
 [ 0.6633 -0.0956  0.5486  0.4998]
 [ 0.      0.9918  0.0944  0.086 ]
 [ 0.      0.      0.6735 -0.7392]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.3481 -0.8433]
 [-0.8029  0.3451]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.8914410770554546 (size = 0.8914)
  λ_2 = -0.8943450015206609 (size = 0.8943)
----------------------------------------

--- Orthogonal Matrix Q number 26 ---
Matrix Q:
[[-0.9036  0.127  -0.0161  0.4088]
 [-0.2097 -0.5366 -0.7493 -0.3265]
 [ 0.3449  0.3256 -0.6074  0.6373]
 [ 0.1433 -0.7681  0.2634  0.5658]]

Hessenberg Form H (of Q):
[[-0.9036  0.0616  0.2624  0.3329]
 [ 0.4284  0.13    0.5536  0.7022]
 [ 0.      0.9896 -0.0891 -0.113 ]
 [-0.     -0.      0.7853 -0.6191]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.6074  0.6373]
 [ 0.2634  0.5658]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.6947481590186378 (size = 0.6947)
  λ_2 = -0.7362843808778513 (size = 0.7363)
----------------------------------------

--- Orthogonal Matrix Q number 27 ---
Matrix Q:
[[-0.3783  0.2175  0.7347 -0.5194]
 [-0.486  -0.5305 -0.4607 -0.5198]
 [-0.7609  0.4199 -0.1943  0.455 ]
 [-0.2046 -0.7036  0.4585  0.5029]]

Hessenberg Form H (of Q):
[[-0.3783 -0.6032 -0.0497 -0.7004]
 [ 0.9257 -0.2465 -0.0203 -0.2862]
 [ 0.      0.7585 -0.0461 -0.65  ]
 [ 0.     -0.     -0.9975  0.0708]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.1943  0.455 ]
 [ 0.4585  0.5029]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.7288996874702811 (size = 0.7289)
  λ_2 = -0.42026844598624513 (size = 0.4203)
----------------------------------------

--- Orthogonal Matrix Q number 28 ---
Matrix Q:
[[-0.6177 -0.2663 -0.3252  0.6646]
 [ 0.7345 -0.3401  0.0741  0.5826]
 [ 0.0015 -0.8476 -0.2571 -0.4641]
 [ 0.2811  0.308  -0.907  -0.0592]]

Hessenberg Form H (of Q):
[[-0.6177  0.0118  0.5166  0.5928]
 [-0.7864 -0.0093 -0.4058 -0.4656]
 [ 0.     -0.9999  0.0099  0.0113]
 [ 0.     -0.      0.7539 -0.657 ]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.2571 -0.4641]
 [-0.907  -0.0592]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.4981407629675436 (size = 0.4981)
  λ_2 = -0.8144343078024388 (size = 0.8144)
----------------------------------------

--- Orthogonal Matrix Q number 29 ---
Matrix Q:
[[-0.889  -0.0288  0.3962 -0.2278]
 [ 0.1961 -0.4549 -0.0899 -0.864 ]
 [-0.4017 -0.36   -0.8217  0.1838]
 [ 0.0995 -0.814   0.3997  0.4095]]

Hessenberg Form H (of Q):
[[-0.889   0.4094  0.14   -0.1501]
 [-0.458  -0.7947 -0.2718  0.2914]
 [ 0.     -0.4483  0.6097 -0.6537]
 [ 0.     -0.     -0.7313 -0.6821]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.8217  0.1838]
 [ 0.3997  0.4095]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.46657705244665193 (size = 0.4666)
  λ_2 = -0.8787442629640516 (size = 0.8787)
----------------------------------------

--- Orthogonal Matrix Q number 30 ---
Matrix Q:
[[-0.2493  0.2801 -0.5896 -0.7154]
 [-0.5128 -0.2897  0.6545 -0.4742]
 [-0.2303 -0.8469 -0.4616  0.1292]
 [-0.7886  0.3471 -0.1044  0.4967]]

Hessenberg Form H (of Q):
[[-0.2493  0.5744 -0.1969 -0.7544]
 [ 0.9684  0.1479 -0.0507 -0.1942]
 [ 0.     -0.8051 -0.1498 -0.5739]
 [ 0.     -0.      0.9676 -0.2526]]

Block Q[3:4,3:4] (indices 2 and 3, 2x2):
[[-0.4616  0.1292]
 [-0.1044  0.4967]]

Eigenvalues of the 2x2 block (analytically calculated):
  λ_1 = 0.4824356418279282 (size = 0.4824)
  λ_2 = -0.4473584670796439 (size = 0.4474)
----------------------------------------
```

So we observe that in the $2 times 2$ blocks analyzed:

+ Orthogonality is not always preserved

+ The eigenvalues are usually real, with alternating sign and size around $1$.


== Shift With an Eigenvalue (d)
<section_shift_with_an_eigenvalue>

Now we use an eigenvalue of the $2 times 2$ block as a shift:

```python
def pretty(arr: np.ndarray, prec: int = 3) -> str:
    
    """
    Compact string for a 1-D NumPy array.

    Args:
        arr (np.ndarray): Input array to be formatted.
        prec (int): Precision for the string representation.
    Returns:
        str: Formatted string representation of the array.
    Raises:
        None
    """

    return np.array_str(arr, precision=prec, suppress_small=True)

def qr_iteration_with_fixed_shift(
    H: np.ndarray,
    mu: complex,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
    debug: bool = False,
):
    """
    Fixed-shift QR iteration that optionally shows the sub-diagonal before the first step and after the final step.

    Args:
        H (np.ndarray): initial matrix in Hessenberg form.
        mu (complex): fixed shift to be used.
        max_iter (int): maximum number of iterations.
        tol (float): tolerance for convergence.
        debug (bool): if True, print detailed information about each iteration.
    
    Returns:
        Hk (np.ndarray): matrix after iterations.
        converged (bool): whether it converged to almost upper triangular form.
        iterations (int): number of iterations performed.
    Raises:
        None
    """

    Hk = H.astype(np.complex128, copy=True)
    n = Hk.shape[0]

    if debug:
        init_sub = np.diag(Hk, k=-1)
        print("  before: subdiag=" + pretty(init_sub) +
              f",  ‖·‖₂={np.linalg.norm(init_sub):.3e}")

    for k in range(max_iter):
        Q, R = np.linalg.qr(Hk - mu * np.eye(n))
        Hk = R @ Q + mu * np.eye(n)

        sub = np.diag(Hk, k=-1)
        if debug:
            print(
                f"  iter {k:02d}: subdiag=" + pretty(sub) +
                f",  ‖·‖₂={np.linalg.norm(sub):.3e}"
            )

        if np.all(np.abs(sub) < tol):
            break  #tests convergence

    if debug: #final sub-diagonal
        final_sub = np.diag(Hk, k=-1)
        print("  after : subdiag=" + pretty(final_sub) +
              f",  ‖·‖₂={np.linalg.norm(final_sub):.3e}")

    converged = np.all(np.abs(np.diag(Hk, k=-1)) < tol)
    return Hk, converged, min(k + 1, max_iter)


def run_qr_iteration_with_shifts_and_debug(
    *,
    n: int = 4,
    n_matrices: int = 30,
    max_iter: int = 50,
    debug: bool = False,
):
    """
    Runs the QR iteration with fixed shifts on randomly generated orthogonal matrices,
    printing a summary for each matrix. Detailed logging appears only when debug = True.

    Args:
        n (int): Size of the matrices (n x n).
        n_matrices (int): Number of orthogonal matrices to generate and analyze.
        max_iter (int): Maximum number of iterations for the QR iteration.
        debug (bool): If True, print detailed information about each iteration.
        
    Returns:
        None
    Raises:
        None
    """

    for idx in range(1, n_matrices + 1):
        print(f"\n┌─ Matrix {idx:02d}/{n_matrices}  (size {n}x{n})")

        Q = generate_orthogonal_matrix_qr(n)
        _, H, _ = to_hessenberg(Q)

        a, b, c, d = H[-2:, -2:].ravel()
        ev1, ev2 = analytical_eigenvalues_2x2(a, b, c, d)
        mu = ev1 if abs(ev1 - H[-1, -1]) < abs(ev2 - H[-1, -1]) else ev2
        print(f"│  fixed shift μ = {mu:.6g} (|μ|={abs(mu):.4f})")

        Hk, ok, iters = qr_iteration_with_fixed_shift(
            H, mu, max_iter=max_iter, tol=1e-10, debug=debug
        )

        print(f"│  iterations    = {iters}/{max_iter}")
        print("│  sub‑diag magnitudes after last step:")
        print("│ ", pretty(np.abs(np.diag(Hk, k=-1))))
        print(f"└─ converged?    = {'yes' if ok else 'no'}")



run_qr_iteration_with_shifts_and_debug(n=4, n_matrices=30, max_iter=100, debug=True)
```

An expected output is:

```
┌─ Matrix 01/30  (size 4x4)
│  fixed shift μ = -0.936534 (|μ|=0.9365)
  before: subdiag=[ 0.959+0.j -0.993+0.j -0.393+0.j],  ‖·‖₂=1.436e+00
  iter 00: subdiag=[0.731+0.j 0.334+0.j 0.235+0.j],  ‖·‖₂=8.375e-01
  iter 01: subdiag=[ 0.723+0.j -0.045+0.j -0.231+0.j],  ‖·‖₂=7.601e-01
  iter 02: subdiag=[0.723+0.j 0.006+0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 03: subdiag=[ 0.723+0.j -0.001+0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 04: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 05: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 06: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 07: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 08: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 09: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 10: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 11: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 12: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 13: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 14: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 15: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 16: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 17: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 18: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 19: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 20: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 21: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 22: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 23: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 24: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 25: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 26: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 27: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 28: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 29: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 30: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 31: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 32: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 33: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 34: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 35: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 36: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 37: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 38: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 39: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 40: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 41: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 42: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 43: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 44: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 45: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 46: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 47: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 48: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 49: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 50: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 51: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 52: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 53: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 54: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 55: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 56: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 57: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 58: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 59: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 60: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 61: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 62: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 63: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 64: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 65: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 66: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 67: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 68: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 69: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 70: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 71: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 72: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 73: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 74: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 75: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 76: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 77: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 78: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 79: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 80: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 81: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 82: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 83: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 84: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 85: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 86: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 87: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 88: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 89: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 90: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 91: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 92: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 93: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 94: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 95: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 96: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 97: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  iter 98: subdiag=[0.723+0.j 0.   +0.j 0.231+0.j],  ‖·‖₂=7.586e-01
  iter 99: subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
  after : subdiag=[ 0.723+0.j -0.   +0.j -0.231+0.j],  ‖·‖₂=7.586e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.723 0.    0.231]
└─ converged?    = no

┌─ Matrix 02/30  (size 4x4)
│  fixed shift μ = 0.985706 (|μ|=0.9857)
  before: subdiag=[ 0.942+0.j -0.63 +0.j -0.464+0.j],  ‖·‖₂=1.224e+00
  iter 00: subdiag=[ 0.982+0.j -0.877+0.j -0.004+0.j],  ‖·‖₂=1.316e+00
  iter 01: subdiag=[ 0.993+0.j -0.997+0.j -0.   +0.j],  ‖·‖₂=1.407e+00
  iter 02: subdiag=[ 0.85 +0.j -0.977+0.j -0.   +0.j],  ‖·‖₂=1.295e+00
  iter 03: subdiag=[ 0.581+0.j -0.942+0.j -0.   +0.j],  ‖·‖₂=1.107e+00
  iter 04: subdiag=[ 0.346+0.j -0.925+0.j -0.   +0.j],  ‖·‖₂=9.879e-01
  iter 05: subdiag=[ 0.195+0.j -0.92 +0.j -0.   +0.j],  ‖·‖₂=9.402e-01
  iter 06: subdiag=[ 0.108+0.j -0.918+0.j -0.   +0.j],  ‖·‖₂=9.244e-01
  iter 07: subdiag=[ 0.059+0.j -0.918+0.j -0.   +0.j],  ‖·‖₂=9.194e-01
  iter 08: subdiag=[ 0.033+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.179e-01
  iter 09: subdiag=[ 0.018+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.175e-01
  iter 10: subdiag=[ 0.01 +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 11: subdiag=[ 0.005+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 12: subdiag=[ 0.003+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 13: subdiag=[ 0.002+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 14: subdiag=[ 0.001+0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 15: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 16: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 17: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 18: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 19: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 20: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 21: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 22: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 23: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 24: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 25: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 26: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 27: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 28: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 29: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 30: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 31: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 32: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 33: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 34: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 35: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 36: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 37: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 38: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 39: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 40: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 41: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 42: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 43: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 44: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 45: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 46: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 47: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 48: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 49: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 50: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 51: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 52: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 53: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 54: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 55: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 56: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 57: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 58: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 59: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 60: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 61: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 62: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 63: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 64: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 65: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 66: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 67: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 68: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 69: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 70: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 71: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 72: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 73: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 74: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 75: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 76: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 77: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 78: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 79: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 80: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 81: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 82: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 83: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 84: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 85: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 86: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 87: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 88: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 89: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 90: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 91: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 92: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 93: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 94: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 95: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 96: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 97: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 98: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  iter 99: subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
  after : subdiag=[ 0.   +0.j -0.917+0.j -0.   +0.j],  ‖·‖₂=9.173e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.917 0.   ]
└─ converged?    = no

┌─ Matrix 03/30  (size 4x4)
│  fixed shift μ = 0.999872 (|μ|=0.9999)
  before: subdiag=[ 0.889+0.j -0.786+0.j  0.008+0.j],  ‖·‖₂=1.186e+00
  iter 00: subdiag=[ 0.692+0.j -0.475+0.j  0.   +0.j],  ‖·‖₂=8.397e-01
  iter 01: subdiag=[ 0.182+0.j -0.444+0.j  0.   +0.j],  ‖·‖₂=4.803e-01
  iter 02: subdiag=[ 0.042+0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.448e-01
  iter 03: subdiag=[ 0.009+0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.428e-01
  iter 04: subdiag=[ 0.002+0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 05: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 06: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 07: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 08: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 09: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 10: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 11: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 12: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 13: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 14: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 15: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 16: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 17: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 18: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 19: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 20: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 21: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 22: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 23: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 24: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 25: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 26: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 27: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 28: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 29: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 30: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 31: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 32: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 33: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 34: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 35: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 36: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 37: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 38: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 39: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 40: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 41: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 42: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 43: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 44: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 45: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 46: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 47: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 48: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 49: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 50: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 51: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 52: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 53: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 54: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 55: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 56: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 57: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 58: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 59: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 60: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 61: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 62: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 63: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 64: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 65: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 66: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 67: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 68: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 69: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 70: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 71: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 72: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 73: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 74: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 75: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 76: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 77: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 78: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 79: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 80: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 81: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 82: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 83: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 84: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 85: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 86: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 87: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 88: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 89: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 90: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 91: subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
  iter 92: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 93: subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
  iter 94: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 95: subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
  iter 96: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 97: subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
  iter 98: subdiag=[ 0.   +0.j -0.443+0.j  0.   +0.j],  ‖·‖₂=4.427e-01
  iter 99: subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
  after : subdiag=[0.   +0.j 0.443+0.j 0.   +0.j],  ‖·‖₂=4.427e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.443 0.   ]
└─ converged?    = no

┌─ Matrix 04/30  (size 4x4)
│  fixed shift μ = -0.623567 (|μ|=0.6236)
  before: subdiag=[-0.823+0.j  0.998+0.j -0.726+0.j],  ‖·‖₂=1.483e+00
  iter 00: subdiag=[-1.   +0.j -0.866+0.j  0.426+0.j],  ‖·‖₂=1.390e+00
  iter 01: subdiag=[-0.973+0.j  0.433+0.j -0.346+0.j],  ‖·‖₂=1.120e+00
  iter 02: subdiag=[-0.964+0.j -0.161+0.j  0.334+0.j],  ‖·‖₂=1.033e+00
  iter 03: subdiag=[-0.963+0.j  0.057+0.j -0.333+0.j],  ‖·‖₂=1.020e+00
  iter 04: subdiag=[-0.962+0.j -0.02 +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 05: subdiag=[-0.962+0.j  0.007+0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 06: subdiag=[-0.962+0.j -0.002+0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 07: subdiag=[-0.962+0.j  0.001+0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 08: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 09: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 10: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 11: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 12: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 13: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 14: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 15: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 16: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 17: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 18: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 19: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 20: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 21: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 22: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 23: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 24: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 25: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 26: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 27: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 28: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 29: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 30: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 31: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 32: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 33: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 34: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 35: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 36: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 37: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 38: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 39: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 40: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 41: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 42: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 43: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 44: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 45: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 46: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 47: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 48: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 49: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 50: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 51: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 52: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 53: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 54: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 55: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 56: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 57: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 58: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 59: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 60: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 61: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 62: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 63: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 64: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 65: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 66: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 67: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 68: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 69: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 70: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 71: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 72: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 73: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 74: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 75: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 76: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 77: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 78: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 79: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 80: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 81: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 82: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 83: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 84: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 85: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 86: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 87: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 88: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 89: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 90: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 91: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 92: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 93: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 94: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 95: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 96: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 97: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  iter 98: subdiag=[-0.962+0.j -0.   +0.j  0.332+0.j],  ‖·‖₂=1.018e+00
  iter 99: subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
  after : subdiag=[-0.962+0.j  0.   +0.j -0.332+0.j],  ‖·‖₂=1.018e+00
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.962 0.    0.332]
└─ converged?    = no

┌─ Matrix 05/30  (size 4x4)
│  fixed shift μ = -0.649367-0.477769j (|μ|=0.8062)
  before: subdiag=[-0.78 +0.j -0.76 +0.j  0.617+0.j],  ‖·‖₂=1.252e+00
  iter 00: subdiag=[-0.705-0.j  0.441-0.j  0.209+0.j],  ‖·‖₂=8.579e-01
  iter 01: subdiag=[-0.627+0.j -0.316+0.j  0.04 +0.j],  ‖·‖₂=7.032e-01
  iter 02: subdiag=[-0.53 +0.j  0.232-0.j  0.008+0.j],  ‖·‖₂=5.789e-01
  iter 03: subdiag=[-0.432+0.j -0.173+0.j  0.001+0.j],  ‖·‖₂=4.649e-01
  iter 04: subdiag=[-0.342+0.j  0.13 -0.j  0.   +0.j],  ‖·‖₂=3.660e-01
  iter 05: subdiag=[-0.267+0.j -0.098+0.j  0.   +0.j],  ‖·‖₂=2.844e-01
  iter 06: subdiag=[-0.206+0.j  0.075-0.j  0.   -0.j],  ‖·‖₂=2.191e-01
  iter 07: subdiag=[-0.158+0.j -0.057+0.j  0.   +0.j],  ‖·‖₂=1.679e-01
  iter 08: subdiag=[-0.121+0.j  0.044-0.j  0.   -0.j],  ‖·‖₂=1.283e-01
  iter 09: subdiag=[-0.092+0.j -0.033+0.j  0.   -0.j],  ‖·‖₂=9.788e-02
  iter 10: subdiag=[-0.07 +0.j  0.026-0.j  0.   -0.j],  ‖·‖₂=7.459e-02
  iter 11: subdiag=[-0.053+0.j -0.02 +0.j  0.   -0.j],  ‖·‖₂=5.681e-02
  iter 12: subdiag=[-0.041+0.j  0.015-0.j  0.   -0.j],  ‖·‖₂=4.325e-02
  iter 13: subdiag=[-0.031+0.j -0.012+0.j  0.   -0.j],  ‖·‖₂=3.292e-02
  iter 14: subdiag=[-0.023+0.j  0.009-0.j  0.   -0.j],  ‖·‖₂=2.506e-02
  iter 15: subdiag=[-0.018+0.j -0.007+0.j  0.   -0.j],  ‖·‖₂=1.907e-02
  iter 16: subdiag=[-0.014+0.j  0.005-0.j  0.   -0.j],  ‖·‖₂=1.451e-02
  iter 17: subdiag=[-0.01 +0.j -0.004+0.j  0.   -0.j],  ‖·‖₂=1.105e-02
  iter 18: subdiag=[-0.008+0.j  0.003-0.j  0.   -0.j],  ‖·‖₂=8.407e-03
  iter 19: subdiag=[-0.006+0.j -0.002+0.j  0.   -0.j],  ‖·‖₂=6.398e-03
  iter 20: subdiag=[-0.005+0.j  0.002-0.j  0.   -0.j],  ‖·‖₂=4.870e-03
  iter 21: subdiag=[-0.003+0.j -0.001+0.j  0.   -0.j],  ‖·‖₂=3.706e-03
  iter 22: subdiag=[-0.003+0.j  0.001-0.j  0.   -0.j],  ‖·‖₂=2.821e-03
  iter 23: subdiag=[-0.002+0.j -0.001+0.j  0.   -0.j],  ‖·‖₂=2.147e-03
  iter 24: subdiag=[-0.002+0.j  0.001-0.j  0.   -0.j],  ‖·‖₂=1.634e-03
  iter 25: subdiag=[-0.001+0.j -0.   +0.j  0.   -0.j],  ‖·‖₂=1.244e-03
  iter 26: subdiag=[-0.001+0.j  0.   -0.j  0.   -0.j],  ‖·‖₂=9.466e-04
  iter 27: subdiag=[-0.001+0.j -0.   +0.j  0.   -0.j],  ‖·‖₂=7.205e-04
  iter 28: subdiag=[-0.001+0.j  0.   -0.j  0.   -0.j],  ‖·‖₂=5.484e-04
  iter 29: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.174e-04
  iter 30: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.177e-04
  iter 31: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.419e-04
  iter 32: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.841e-04
  iter 33: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.401e-04
  iter 34: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.067e-04
  iter 35: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=8.120e-05
  iter 36: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=6.182e-05
  iter 37: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.706e-05
  iter 38: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.582e-05
  iter 39: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.727e-05
  iter 40: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.076e-05
  iter 41: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.581e-05
  iter 42: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.203e-05
  iter 43: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=9.162e-06
  iter 44: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=6.975e-06
  iter 45: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=5.311e-06
  iter 46: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=4.044e-06
  iter 47: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=3.079e-06
  iter 48: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.344e-06
  iter 49: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.785e-06
  iter 50: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.359e-06
  iter 51: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.035e-06
  iter 52: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=7.880e-07
  iter 53: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=6.000e-07
  iter 54: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=4.569e-07
  iter 55: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=3.479e-07
  iter 56: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.650e-07
  iter 57: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.018e-07
  iter 58: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.537e-07
  iter 59: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.170e-07
  iter 60: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=8.913e-08
  iter 61: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=6.788e-08
  iter 62: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=5.170e-08
  iter 63: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=3.937e-08
  iter 64: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.999e-08
  iter 65: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.284e-08
  iter 66: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.740e-08
  iter 67: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.325e-08
  iter 68: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.009e-08
  iter 69: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=7.688e-09
  iter 70: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=5.857e-09
  iter 71: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.461e-09
  iter 72: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.398e-09
  iter 73: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.589e-09
  iter 74: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.972e-09
  iter 75: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.503e-09
  iter 76: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.145e-09
  iter 77: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=8.721e-10
  iter 78: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=6.644e-10
  iter 79: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=5.062e-10
  iter 80: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.857e-10
  iter 81: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.939e-10
  iter 82: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.239e-10
  iter 83: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.706e-10
  iter 84: subdiag=[-0.+0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.300e-10
  iter 85: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=9.907e-11
  after : subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=9.907e-11
│  iterations    = 86/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = yes

┌─ Matrix 06/30  (size 4x4)
│  fixed shift μ = -0.771151 (|μ|=0.7712)
  before: subdiag=[-0.695+0.j  0.921+0.j  0.395+0.j],  ‖·‖₂=1.220e+00
  iter 00: subdiag=[-0.414+0.j  0.919+0.j  0.099+0.j],  ‖·‖₂=1.012e+00
  iter 01: subdiag=[-0.236+0.j  0.912+0.j  0.023+0.j],  ‖·‖₂=9.419e-01
  iter 02: subdiag=[-0.131+0.j  0.909+0.j  0.005+0.j],  ‖·‖₂=9.186e-01
  iter 03: subdiag=[-0.072+0.j  0.908+0.j  0.001+0.j],  ‖·‖₂=9.112e-01
  iter 04: subdiag=[-0.04 +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.090e-01
  iter 05: subdiag=[-0.022+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.083e-01
  iter 06: subdiag=[-0.012+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.081e-01
  iter 07: subdiag=[-0.007+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 08: subdiag=[-0.004+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 09: subdiag=[-0.002+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 10: subdiag=[-0.001+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 11: subdiag=[-0.001+0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 12: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 13: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 14: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 15: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 16: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 17: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 18: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 19: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 20: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 21: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 22: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 23: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 24: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 25: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 26: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 27: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 28: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 29: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 30: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 31: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 32: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 33: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 34: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 35: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 36: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 37: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 38: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 39: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 40: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 41: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 42: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 43: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 44: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 45: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 46: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 47: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 48: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 49: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 50: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 51: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 52: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 53: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 54: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 55: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 56: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 57: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 58: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 59: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 60: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 61: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 62: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 63: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 64: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 65: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 66: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 67: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 68: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 69: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 70: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 71: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 72: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 73: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 74: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 75: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 76: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 77: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 78: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 79: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 80: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 81: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 82: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 83: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 84: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 85: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 86: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 87: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 88: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 89: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 90: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 91: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 92: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 93: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 94: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 95: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 96: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 97: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 98: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  iter 99: subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
  after : subdiag=[-0.   +0.j  0.908+0.j  0.   +0.j],  ‖·‖₂=9.080e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.908 0.   ]
└─ converged?    = no

┌─ Matrix 07/30  (size 4x4)
│  fixed shift μ = -0.884666 (|μ|=0.8847)
  before: subdiag=[ 1.   +0.j -0.815+0.j  0.838+0.j],  ‖·‖₂=1.538e+00
  iter 00: subdiag=[ 0.626+0.j  0.831+0.j -0.362+0.j],  ‖·‖₂=1.102e+00
  iter 01: subdiag=[ 0.56 +0.j -0.193+0.j  0.317+0.j],  ‖·‖₂=6.720e-01
  iter 02: subdiag=[ 0.558+0.j  0.035+0.j -0.316+0.j],  ‖·‖₂=6.419e-01
  iter 03: subdiag=[ 0.558+0.j -0.006+0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 04: subdiag=[ 0.558+0.j  0.001+0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 05: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 06: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 07: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 08: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 09: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 10: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 11: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 12: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 13: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 14: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 15: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 16: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 17: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 18: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 19: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 20: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 21: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 22: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 23: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 24: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 25: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 26: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 27: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 28: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 29: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 30: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 31: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 32: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 33: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 34: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 35: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 36: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 37: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 38: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 39: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 40: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 41: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 42: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 43: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 44: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 45: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 46: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 47: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 48: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 49: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 50: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 51: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 52: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 53: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 54: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 55: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 56: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 57: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 58: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 59: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 60: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 61: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 62: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 63: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 64: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 65: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 66: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 67: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 68: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 69: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 70: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 71: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 72: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 73: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 74: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 75: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 76: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 77: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 78: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 79: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 80: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 81: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 82: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 83: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 84: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 85: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 86: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 87: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 88: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 89: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 90: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 91: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 92: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 93: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 94: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 95: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 96: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 97: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  iter 98: subdiag=[ 0.558+0.j  0.   +0.j -0.316+0.j],  ‖·‖₂=6.409e-01
  iter 99: subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
  after : subdiag=[ 0.558+0.j -0.   +0.j  0.316+0.j],  ‖·‖₂=6.409e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.558 0.    0.316]
└─ converged?    = no

┌─ Matrix 08/30  (size 4x4)
│  fixed shift μ = -0.662036-0.664809j (|μ|=0.9382)
  before: subdiag=[ 0.992+0.j -0.474+0.j  0.71 +0.j],  ‖·‖₂=1.309e+00
  iter 00: subdiag=[0.782+0.j 0.413-0.j 0.08 -0.j],  ‖·‖₂=8.875e-01
  iter 01: subdiag=[ 0.462-0.j  0.541-0.j -0.005+0.j],  ‖·‖₂=7.119e-01
  iter 02: subdiag=[0.25 +0.j 0.713-0.j 0.   -0.j],  ‖·‖₂=7.559e-01
  iter 03: subdiag=[ 0.136+0.j  0.876-0.j -0.   +0.j],  ‖·‖₂=8.861e-01
  iter 04: subdiag=[0.078-0.j 0.95 -0.j 0.   -0.j],  ‖·‖₂=9.531e-01
  iter 05: subdiag=[ 0.048+0.j  0.89 -0.j -0.   +0.j],  ‖·‖₂=8.914e-01
  iter 06: subdiag=[0.032+0.j 0.732-0.j 0.   -0.j],  ‖·‖₂=7.331e-01
  iter 07: subdiag=[ 0.022+0.j  0.551-0.j -0.   +0.j],  ‖·‖₂=5.514e-01
  iter 08: subdiag=[0.015+0.j 0.393-0.j 0.   -0.j],  ‖·‖₂=3.938e-01
  iter 09: subdiag=[-0.011-0.j -0.274+0.j -0.   +0.j],  ‖·‖₂=2.738e-01
  iter 10: subdiag=[0.008+0.j 0.188-0.j 0.   -0.j],  ‖·‖₂=1.880e-01
  iter 11: subdiag=[-0.006-0.j -0.128+0.j -0.   +0.j],  ‖·‖₂=1.282e-01
  iter 12: subdiag=[0.004+0.j 0.087-0.j 0.   -0.j],  ‖·‖₂=8.725e-02
  iter 13: subdiag=[-0.003-0.j -0.059+0.j -0.   +0.j],  ‖·‖₂=5.928e-02
  iter 14: subdiag=[0.002+0.j 0.04 -0.j 0.   -0.j],  ‖·‖₂=4.025e-02
  iter 15: subdiag=[-0.002-0.j -0.027+0.j -0.   +0.j],  ‖·‖₂=2.732e-02
  iter 16: subdiag=[0.001+0.j 0.019-0.j 0.   -0.j],  ‖·‖₂=1.855e-02
  iter 17: subdiag=[-0.001-0.j -0.013+0.j -0.   +0.j],  ‖·‖₂=1.259e-02
  iter 18: subdiag=[0.001+0.j 0.009-0.j 0.   -0.j],  ‖·‖₂=8.545e-03
  iter 19: subdiag=[-0.   -0.j -0.006+0.j -0.   +0.j],  ‖·‖₂=5.800e-03
  iter 20: subdiag=[0.   +0.j 0.004-0.j 0.   -0.j],  ‖·‖₂=3.937e-03
  iter 21: subdiag=[-0.   -0.j -0.003+0.j -0.   +0.j],  ‖·‖₂=2.673e-03
  iter 22: subdiag=[0.   +0.j 0.002-0.j 0.   -0.j],  ‖·‖₂=1.814e-03
  iter 23: subdiag=[-0.   -0.j -0.001+0.j -0.   +0.j],  ‖·‖₂=1.232e-03
  iter 24: subdiag=[0.   +0.j 0.001-0.j 0.   -0.j],  ‖·‖₂=8.364e-04
  iter 25: subdiag=[-0.   -0.j -0.001+0.j -0.   +0.j],  ‖·‖₂=5.679e-04
  iter 26: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=3.857e-04
  iter 27: subdiag=[-0.-0.j -0.+0.j -0.+0.j],  ‖·‖₂=2.619e-04
  iter 28: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=1.779e-04
  iter 29: subdiag=[-0.-0.j -0.+0.j -0.-0.j],  ‖·‖₂=1.209e-04
  iter 30: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=8.213e-05
  iter 31: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=5.581e-05
  iter 32: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=3.794e-05
  iter 33: subdiag=[-0.-0.j -0.+0.j -0.-0.j],  ‖·‖₂=2.579e-05
  iter 34: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=1.754e-05
  iter 35: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.193e-05
  iter 36: subdiag=[0.+0.j 0.+0.j 0.-0.j],  ‖·‖₂=8.118e-06
  iter 37: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=5.526e-06
  iter 38: subdiag=[0.+0.j 0.+0.j 0.-0.j],  ‖·‖₂=3.763e-06
  iter 39: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.564e-06
  iter 40: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=1.748e-06
  iter 41: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.192e-06
  iter 42: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=8.135e-07
  iter 43: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=5.556e-07
  iter 44: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=3.797e-07
  iter 45: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=2.598e-07
  iter 46: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=1.779e-07
  iter 47: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.219e-07
  iter 48: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=8.368e-08
  iter 49: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=5.750e-08
  iter 50: subdiag=[0.+0.j 0.+0.j 0.-0.j],  ‖·‖₂=3.956e-08
  iter 51: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.726e-08
  iter 52: subdiag=[0.+0.j 0.+0.j 0.-0.j],  ‖·‖₂=1.881e-08
  iter 53: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.300e-08
  iter 54: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=9.004e-09
  iter 55: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=6.246e-09
  iter 56: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=4.341e-09
  iter 57: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=3.023e-09
  iter 58: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=2.110e-09
  iter 59: subdiag=[-0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.475e-09
  iter 60: subdiag=[0.+0.j 0.+0.j 0.+0.j],  ‖·‖₂=1.034e-09
  iter 61: subdiag=[-0.-0.j -0.+0.j -0.-0.j],  ‖·‖₂=7.257e-10
  iter 62: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=5.106e-10
  iter 63: subdiag=[-0.-0.j -0.+0.j -0.-0.j],  ‖·‖₂=3.599e-10
  iter 64: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=2.542e-10
  iter 65: subdiag=[-0.-0.j -0.+0.j -0.-0.j],  ‖·‖₂=1.799e-10
  iter 66: subdiag=[0.+0.j 0.-0.j 0.+0.j],  ‖·‖₂=1.276e-10
  iter 67: subdiag=[-0.-0.j -0.+0.j -0.+0.j],  ‖·‖₂=9.060e-11
  after : subdiag=[-0.-0.j -0.+0.j -0.+0.j],  ‖·‖₂=9.060e-11
│  iterations    = 68/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = yes

┌─ Matrix 09/30  (size 4x4)
│  fixed shift μ = 0.995354 (|μ|=0.9954)
  before: subdiag=[ 0.805+0.j  0.682+0.j -0.243+0.j],  ‖·‖₂=1.083e+00
  iter 00: subdiag=[ 0.618+0.j -0.241+0.j  0.043+0.j],  ‖·‖₂=6.650e-01
  iter 01: subdiag=[ 0.614+0.j  0.006+0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 02: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 03: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 04: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 05: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 06: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 07: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 08: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 09: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 10: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 11: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 12: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 13: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 14: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 15: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 16: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 17: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 18: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 19: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 20: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 21: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 22: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 23: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 24: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 25: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 26: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 27: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 28: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 29: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 30: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 31: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 32: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 33: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 34: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 35: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 36: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 37: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 38: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 39: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 40: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 41: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 42: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 43: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 44: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 45: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 46: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 47: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 48: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 49: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 50: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 51: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 52: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 53: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 54: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 55: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 56: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 57: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 58: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 59: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 60: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 61: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 62: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 63: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 64: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 65: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 66: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 67: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 68: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 69: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 70: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 71: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 72: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 73: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 74: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 75: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 76: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 77: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 78: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 79: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 80: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 81: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 82: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 83: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 84: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 85: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 86: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 87: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 88: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 89: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 90: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 91: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 92: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 93: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 94: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 95: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 96: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 97: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  iter 98: subdiag=[ 0.614+0.j -0.   +0.j  0.043+0.j],  ‖·‖₂=6.158e-01
  iter 99: subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
  after : subdiag=[ 0.614+0.j  0.   +0.j -0.043+0.j],  ‖·‖₂=6.158e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.614 0.    0.043]
└─ converged?    = no

┌─ Matrix 10/30  (size 4x4)
│  fixed shift μ = -0.926518 (|μ|=0.9265)
  before: subdiag=[ 0.914+0.j  0.658+0.j -0.888+0.j],  ‖·‖₂=1.434e+00
  iter 00: subdiag=[ 0.572+0.j -0.901+0.j  0.244+0.j],  ‖·‖₂=1.095e+00
  iter 01: subdiag=[ 0.491+0.j  0.144+0.j -0.205+0.j],  ‖·‖₂=5.513e-01
  iter 02: subdiag=[ 0.49 +0.j -0.016+0.j  0.204+0.j],  ‖·‖₂=5.311e-01
  iter 03: subdiag=[ 0.49 +0.j  0.002+0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 04: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 05: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 06: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 07: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 08: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 09: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 10: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 11: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 12: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 13: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 14: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 15: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 16: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 17: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 18: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 19: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 20: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 21: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 22: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 23: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 24: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 25: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 26: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 27: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 28: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 29: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 30: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 31: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 32: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 33: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 34: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 35: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 36: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 37: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 38: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 39: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 40: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 41: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 42: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 43: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 44: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 45: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 46: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 47: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 48: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 49: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 50: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 51: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 52: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 53: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 54: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 55: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 56: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 57: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 58: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 59: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 60: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 61: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 62: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 63: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 64: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 65: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 66: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 67: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 68: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 69: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 70: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 71: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 72: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 73: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 74: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 75: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 76: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 77: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 78: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 79: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 80: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 81: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 82: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 83: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 84: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 85: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 86: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 87: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 88: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 89: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 90: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 91: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 92: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 93: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 94: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 95: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 96: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 97: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  iter 98: subdiag=[ 0.49 +0.j -0.   +0.j  0.204+0.j],  ‖·‖₂=5.308e-01
  iter 99: subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
  after : subdiag=[ 0.49 +0.j  0.   +0.j -0.204+0.j],  ‖·‖₂=5.308e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.49  0.    0.204]
└─ converged?    = no

┌─ Matrix 11/30  (size 4x4)
│  fixed shift μ = -0.553226-0.269054j (|μ|=0.6152)
  before: subdiag=[-0.998+0.j  0.926+0.j  0.596+0.j],  ‖·‖₂=1.486e+00
  iter 00: subdiag=[-0.938+0.j -0.68 +0.j  0.382+0.j],  ‖·‖₂=1.219e+00
  iter 01: subdiag=[-0.81 +0.j  0.462-0.j  0.206-0.j],  ‖·‖₂=9.552e-01
  iter 02: subdiag=[-0.661+0.j -0.318+0.j  0.103+0.j],  ‖·‖₂=7.404e-01
  iter 03: subdiag=[-0.517-0.j  0.221-0.j  0.051+0.j],  ‖·‖₂=5.644e-01
  iter 04: subdiag=[-0.394-0.j -0.156+0.j  0.025+0.j],  ‖·‖₂=4.240e-01
  iter 05: subdiag=[-0.295-0.j  0.11 -0.j  0.012+0.j],  ‖·‖₂=3.154e-01
  iter 06: subdiag=[-0.22 -0.j -0.078+0.j  0.006-0.j],  ‖·‖₂=2.332e-01
  iter 07: subdiag=[-0.163+0.j  0.055-0.j  0.003+0.j],  ‖·‖₂=1.718e-01
  iter 08: subdiag=[-0.12 -0.j -0.039+0.j  0.001+0.j],  ‖·‖₂=1.263e-01
  iter 09: subdiag=[-0.089-0.j  0.028-0.j  0.001-0.j],  ‖·‖₂=9.283e-02
  iter 10: subdiag=[-0.065-0.j -0.02 +0.j  0.   -0.j],  ‖·‖₂=6.818e-02
  iter 11: subdiag=[-0.048+0.j  0.014-0.j  0.   -0.j],  ‖·‖₂=5.006e-02
  iter 12: subdiag=[-0.035+0.j -0.01 +0.j  0.   -0.j],  ‖·‖₂=3.676e-02
  iter 13: subdiag=[-0.026+0.j  0.007-0.j  0.   +0.j],  ‖·‖₂=2.699e-02
  iter 14: subdiag=[-0.019+0.j -0.005+0.j  0.   +0.j],  ‖·‖₂=1.982e-02
  iter 15: subdiag=[-0.014+0.j  0.004-0.j  0.   +0.j],  ‖·‖₂=1.456e-02
  iter 16: subdiag=[-0.01 +0.j -0.003+0.j  0.   +0.j],  ‖·‖₂=1.069e-02
  iter 17: subdiag=[-0.008+0.j  0.002-0.j  0.   +0.j],  ‖·‖₂=7.855e-03
  iter 18: subdiag=[-0.006+0.j -0.001+0.j  0.   -0.j],  ‖·‖₂=5.771e-03
  iter 19: subdiag=[-0.004+0.j  0.001-0.j  0.   -0.j],  ‖·‖₂=4.240e-03
  iter 20: subdiag=[-0.003-0.j -0.001-0.j  0.   -0.j],  ‖·‖₂=3.116e-03
  iter 21: subdiag=[-0.002-0.j  0.   +0.j  0.   -0.j],  ‖·‖₂=2.290e-03
  iter 22: subdiag=[-0.002-0.j -0.   -0.j  0.   +0.j],  ‖·‖₂=1.683e-03
  iter 23: subdiag=[-0.001-0.j  0.   -0.j  0.   +0.j],  ‖·‖₂=1.237e-03
  iter 24: subdiag=[-0.001-0.j -0.   +0.j  0.   -0.j],  ‖·‖₂=9.094e-04
  iter 25: subdiag=[-0.001-0.j  0.   +0.j  0.   -0.j],  ‖·‖₂=6.685e-04
  iter 26: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.915e-04
  iter 27: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.614e-04
  iter 28: subdiag=[-0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.657e-04
  iter 29: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=1.954e-04
  iter 30: subdiag=[-0.+0.j -0.+0.j  0.+0.j],  ‖·‖₂=1.437e-04
  iter 31: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=1.057e-04
  iter 32: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=7.772e-05
  iter 33: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=5.716e-05
  iter 34: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=4.204e-05
  iter 35: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=3.093e-05
  iter 36: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=2.275e-05
  iter 37: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=1.673e-05
  iter 38: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=1.231e-05
  iter 39: subdiag=[-0.-0.j  0.+0.j  0.+0.j],  ‖·‖₂=9.056e-06
  iter 40: subdiag=[-0.-0.j -0.-0.j  0.+0.j],  ‖·‖₂=6.662e-06
  iter 41: subdiag=[-0.-0.j  0.+0.j  0.+0.j],  ‖·‖₂=4.902e-06
  iter 42: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=3.606e-06
  iter 43: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=2.653e-06
  iter 44: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=1.952e-06
  iter 45: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=1.436e-06
  iter 46: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=1.057e-06
  iter 47: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=7.776e-07
  iter 48: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=5.722e-07
  iter 49: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=4.210e-07
  iter 50: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=3.098e-07
  iter 51: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=2.280e-07
  iter 52: subdiag=[-0.-0.j -0.+0.j  0.+0.j],  ‖·‖₂=1.677e-07
  iter 53: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=1.234e-07
  iter 54: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=9.083e-08
  iter 55: subdiag=[-0.-0.j  0.-0.j  0.+0.j],  ‖·‖₂=6.684e-08
  iter 56: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.919e-08
  iter 57: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.620e-08
  iter 58: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.664e-08
  iter 59: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.960e-08
  iter 60: subdiag=[-0.-0.j -0.-0.j  0.-0.j],  ‖·‖₂=1.443e-08
  iter 61: subdiag=[-0.-0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.062e-08
  iter 62: subdiag=[-0.-0.j -0.-0.j  0.-0.j],  ‖·‖₂=7.812e-09
  iter 63: subdiag=[-0.-0.j  0.+0.j  0.-0.j],  ‖·‖₂=5.749e-09
  iter 64: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=4.231e-09
  iter 65: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=3.114e-09
  iter 66: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=2.292e-09
  iter 67: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.686e-09
  iter 68: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.241e-09
  iter 69: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=9.134e-10
  iter 70: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=6.722e-10
  iter 71: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=4.947e-10
  iter 72: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=3.641e-10
  iter 73: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=2.680e-10
  iter 74: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.972e-10
  iter 75: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=1.451e-10
  iter 76: subdiag=[-0.-0.j -0.+0.j  0.-0.j],  ‖·‖₂=1.068e-10
  iter 77: subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=7.861e-11
  after : subdiag=[-0.-0.j  0.-0.j  0.-0.j],  ‖·‖₂=7.861e-11
│  iterations    = 78/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = yes

┌─ Matrix 12/30  (size 4x4)
│  fixed shift μ = 0.918038 (|μ|=0.9180)
  before: subdiag=[-0.634+0.j  0.891+0.j  0.631+0.j],  ‖·‖₂=1.263e+00
  iter 00: subdiag=[-0.868+0.j -0.701+0.j -0.147+0.j],  ‖·‖₂=1.126e+00
  iter 01: subdiag=[-0.823+0.j  0.075+0.j  0.133+0.j],  ‖·‖₂=8.371e-01
  iter 02: subdiag=[-0.823+0.j -0.007+0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 03: subdiag=[-0.823+0.j  0.001+0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 04: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 05: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 06: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 07: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 08: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 09: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 10: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 11: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 12: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 13: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 14: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 15: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 16: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 17: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 18: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 19: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 20: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 21: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 22: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 23: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 24: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 25: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 26: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 27: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 28: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 29: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 30: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 31: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 32: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 33: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 34: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 35: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 36: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 37: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 38: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 39: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 40: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 41: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 42: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 43: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 44: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 45: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 46: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 47: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 48: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 49: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 50: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 51: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 52: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 53: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 54: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 55: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 56: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 57: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 58: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 59: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 60: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 61: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 62: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 63: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 64: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 65: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 66: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 67: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 68: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 69: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 70: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 71: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 72: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 73: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 74: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 75: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 76: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 77: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 78: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 79: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 80: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 81: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 82: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 83: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 84: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 85: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 86: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 87: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 88: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 89: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 90: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 91: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 92: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 93: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 94: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 95: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 96: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 97: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  iter 98: subdiag=[-0.823+0.j -0.   +0.j -0.133+0.j],  ‖·‖₂=8.333e-01
  iter 99: subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
  after : subdiag=[-0.823+0.j  0.   +0.j  0.133+0.j],  ‖·‖₂=8.333e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.823 0.    0.133]
└─ converged?    = no

┌─ Matrix 13/30  (size 4x4)
│  fixed shift μ = -0.0259786-0.624268j (|μ|=0.6248)
  before: subdiag=[ 0.996+0.j -0.921+0.j  0.999+0.j],  ‖·‖₂=1.685e+00
  iter 00: subdiag=[ 0.78 +0.j -0.999+0.j  0.781+0.j],  ‖·‖₂=1.488e+00
  iter 01: subdiag=[-0.6  +0.j -0.718+0.j -0.6  -0.j],  ‖·‖₂=1.111e+00
  iter 02: subdiag=[0.56 +0.j 0.319-0.j 0.559+0.j],  ‖·‖₂=8.526e-01
  iter 03: subdiag=[-0.553+0.j -0.124-0.j -0.551+0.j],  ‖·‖₂=7.910e-01
  iter 04: subdiag=[0.552+0.j 0.047-0.j 0.549+0.j],  ‖·‖₂=7.802e-01
  iter 05: subdiag=[-0.551-0.j  0.018-0.j -0.548+0.j],  ‖·‖₂=7.772e-01
  iter 06: subdiag=[0.549+0.j 0.007-0.j 0.547+0.j],  ‖·‖₂=7.750e-01
  iter 07: subdiag=[-0.548-0.j  0.003+0.j -0.545+0.j],  ‖·‖₂=7.726e-01
  iter 08: subdiag=[0.546+0.j 0.001+0.j 0.543-0.j],  ‖·‖₂=7.698e-01
  iter 09: subdiag=[-0.543-0.j  0.   +0.j -0.541+0.j],  ‖·‖₂=7.666e-01
  iter 10: subdiag=[0.54 +0.j 0.   +0.j 0.539-0.j],  ‖·‖₂=7.631e-01
  iter 11: subdiag=[-0.537-0.j  0.   +0.j -0.536+0.j],  ‖·‖₂=7.591e-01
  iter 12: subdiag=[0.534-0.j 0.   +0.j 0.534+0.j],  ‖·‖₂=7.548e-01
  iter 13: subdiag=[-0.53 +0.j  0.   +0.j -0.531-0.j],  ‖·‖₂=7.502e-01
  iter 14: subdiag=[0.526-0.j 0.   +0.j 0.528+0.j],  ‖·‖₂=7.452e-01
  iter 15: subdiag=[-0.521+0.j  0.   +0.j -0.525+0.j],  ‖·‖₂=7.399e-01
  iter 16: subdiag=[0.517-0.j 0.   +0.j 0.522+0.j],  ‖·‖₂=7.343e-01
  iter 17: subdiag=[-0.512-0.j  0.   +0.j -0.518-0.j],  ‖·‖₂=7.283e-01
  iter 18: subdiag=[0.506+0.j 0.   +0.j 0.515+0.j],  ‖·‖₂=7.221e-01
  iter 19: subdiag=[-0.501-0.j  0.   +0.j -0.511-0.j],  ‖·‖₂=7.157e-01
  iter 20: subdiag=[0.495+0.j 0.   +0.j 0.507+0.j],  ‖·‖₂=7.089e-01
  iter 21: subdiag=[-0.489-0.j  0.   +0.j -0.503+0.j],  ‖·‖₂=7.020e-01
  iter 22: subdiag=[0.483+0.j 0.   +0.j 0.499-0.j],  ‖·‖₂=6.948e-01
  iter 23: subdiag=[-0.477-0.j  0.   +0.j -0.495+0.j],  ‖·‖₂=6.874e-01
  iter 24: subdiag=[0.47 +0.j 0.   +0.j 0.491-0.j],  ‖·‖₂=6.798e-01
  iter 25: subdiag=[-0.464-0.j  0.   +0.j -0.486+0.j],  ‖·‖₂=6.720e-01
  iter 26: subdiag=[0.457+0.j 0.   +0.j 0.482-0.j],  ‖·‖₂=6.641e-01
  iter 27: subdiag=[-0.45 -0.j  0.   +0.j -0.477+0.j],  ‖·‖₂=6.560e-01
  iter 28: subdiag=[0.443+0.j 0.   +0.j 0.472+0.j],  ‖·‖₂=6.478e-01
  iter 29: subdiag=[-0.436-0.j  0.   +0.j -0.468+0.j],  ‖·‖₂=6.395e-01
  iter 30: subdiag=[0.429+0.j 0.   +0.j 0.463+0.j],  ‖·‖₂=6.311e-01
  iter 31: subdiag=[-0.422-0.j  0.   +0.j -0.458+0.j],  ‖·‖₂=6.226e-01
  iter 32: subdiag=[0.415+0.j 0.   +0.j 0.453-0.j],  ‖·‖₂=6.140e-01
  iter 33: subdiag=[-0.408-0.j  0.   +0.j -0.448+0.j],  ‖·‖₂=6.054e-01
  iter 34: subdiag=[0.4  +0.j 0.   +0.j 0.443+0.j],  ‖·‖₂=5.967e-01
  iter 35: subdiag=[-0.393-0.j  0.   +0.j -0.437+0.j],  ‖·‖₂=5.880e-01
  iter 36: subdiag=[0.386+0.j 0.   +0.j 0.432-0.j],  ‖·‖₂=5.792e-01
  iter 37: subdiag=[-0.378-0.j  0.   +0.j -0.427+0.j],  ‖·‖₂=5.704e-01
  iter 38: subdiag=[0.371+0.j 0.   +0.j 0.422+0.j],  ‖·‖₂=5.616e-01
  iter 39: subdiag=[-0.364-0.j  0.   +0.j -0.416+0.j],  ‖·‖₂=5.529e-01
  iter 40: subdiag=[0.357+0.j 0.   +0.j 0.411+0.j],  ‖·‖₂=5.441e-01
  iter 41: subdiag=[-0.349-0.j  0.   +0.j -0.406+0.j],  ‖·‖₂=5.354e-01
  iter 42: subdiag=[0.342+0.j 0.   +0.j 0.4  -0.j],  ‖·‖₂=5.266e-01
  iter 43: subdiag=[-0.335-0.j  0.   +0.j -0.395+0.j],  ‖·‖₂=5.180e-01
  iter 44: subdiag=[0.328+0.j 0.   +0.j 0.389-0.j],  ‖·‖₂=5.093e-01
  iter 45: subdiag=[-0.321-0.j  0.   +0.j -0.384+0.j],  ‖·‖₂=5.008e-01
  iter 46: subdiag=[0.314+0.j 0.   +0.j 0.379-0.j],  ‖·‖₂=4.922e-01
  iter 47: subdiag=[-0.308-0.j  0.   +0.j -0.373+0.j],  ‖·‖₂=4.838e-01
  iter 48: subdiag=[0.301+0.j 0.   +0.j 0.368-0.j],  ‖·‖₂=4.754e-01
  iter 49: subdiag=[-0.294-0.j  0.   +0.j -0.363+0.j],  ‖·‖₂=4.671e-01
  iter 50: subdiag=[0.288+0.j 0.   +0.j 0.357-0.j],  ‖·‖₂=4.588e-01
  iter 51: subdiag=[-0.281-0.j  0.   +0.j -0.352+0.j],  ‖·‖₂=4.507e-01
  iter 52: subdiag=[0.275+0.j 0.   +0.j 0.347-0.j],  ‖·‖₂=4.426e-01
  iter 53: subdiag=[-0.269-0.j  0.   +0.j -0.342+0.j],  ‖·‖₂=4.346e-01
  iter 54: subdiag=[0.263+0.j 0.   +0.j 0.336+0.j],  ‖·‖₂=4.267e-01
  iter 55: subdiag=[-0.257-0.j  0.   +0.j -0.331+0.j],  ‖·‖₂=4.189e-01
  iter 56: subdiag=[0.251+0.j 0.   +0.j 0.326+0.j],  ‖·‖₂=4.112e-01
  iter 57: subdiag=[-0.245-0.j  0.   +0.j -0.321-0.j],  ‖·‖₂=4.036e-01
  iter 58: subdiag=[0.239+0.j 0.   +0.j 0.316+0.j],  ‖·‖₂=3.961e-01
  iter 59: subdiag=[-0.233-0.j  0.   +0.j -0.311-0.j],  ‖·‖₂=3.887e-01
  iter 60: subdiag=[0.228+0.j 0.   +0.j 0.306+0.j],  ‖·‖₂=3.814e-01
  iter 61: subdiag=[-0.222-0.j  0.   +0.j -0.301-0.j],  ‖·‖₂=3.742e-01
  iter 62: subdiag=[0.217+0.j 0.   +0.j 0.296+0.j],  ‖·‖₂=3.671e-01
  iter 63: subdiag=[-0.212-0.j  0.   +0.j -0.291-0.j],  ‖·‖₂=3.601e-01
  iter 64: subdiag=[0.207+0.j 0.   +0.j 0.286+0.j],  ‖·‖₂=3.532e-01
  iter 65: subdiag=[-0.202-0.j  0.   +0.j -0.282-0.j],  ‖·‖₂=3.464e-01
  iter 66: subdiag=[0.197+0.j 0.   +0.j 0.277+0.j],  ‖·‖₂=3.398e-01
  iter 67: subdiag=[-0.192-0.j  0.   +0.j -0.272-0.j],  ‖·‖₂=3.332e-01
  iter 68: subdiag=[0.187+0.j 0.   +0.j 0.268+0.j],  ‖·‖₂=3.267e-01
  iter 69: subdiag=[-0.183-0.j  0.   +0.j -0.263+0.j],  ‖·‖₂=3.204e-01
  iter 70: subdiag=[0.178+0.j 0.   +0.j 0.259+0.j],  ‖·‖₂=3.141e-01
  iter 71: subdiag=[-0.174-0.j  0.   +0.j -0.254-0.j],  ‖·‖₂=3.080e-01
  iter 72: subdiag=[0.169+0.j 0.   +0.j 0.25 +0.j],  ‖·‖₂=3.020e-01
  iter 73: subdiag=[-0.165-0.j  0.   +0.j -0.246-0.j],  ‖·‖₂=2.960e-01
  iter 74: subdiag=[0.161+0.j 0.   +0.j 0.241+0.j],  ‖·‖₂=2.902e-01
  iter 75: subdiag=[-0.157-0.j  0.   +0.j -0.237-0.j],  ‖·‖₂=2.845e-01
  iter 76: subdiag=[0.153+0.j 0.   +0.j 0.233+0.j],  ‖·‖₂=2.789e-01
  iter 77: subdiag=[-0.149-0.j  0.   +0.j -0.229+0.j],  ‖·‖₂=2.733e-01
  iter 78: subdiag=[0.146+0.j 0.   +0.j 0.225-0.j],  ‖·‖₂=2.679e-01
  iter 79: subdiag=[-0.142-0.j  0.   +0.j -0.221+0.j],  ‖·‖₂=2.626e-01
  iter 80: subdiag=[0.138+0.j 0.   +0.j 0.217+0.j],  ‖·‖₂=2.574e-01
  iter 81: subdiag=[-0.135-0.j  0.   +0.j -0.213+0.j],  ‖·‖₂=2.522e-01
  iter 82: subdiag=[0.131+0.j 0.   +0.j 0.209+0.j],  ‖·‖₂=2.472e-01
  iter 83: subdiag=[-0.128-0.j  0.   +0.j -0.206-0.j],  ‖·‖₂=2.423e-01
  iter 84: subdiag=[0.125+0.j 0.   +0.j 0.202+0.j],  ‖·‖₂=2.374e-01
  iter 85: subdiag=[-0.122-0.j  0.   +0.j -0.198-0.j],  ‖·‖₂=2.327e-01
  iter 86: subdiag=[0.119+0.j 0.   +0.j 0.195+0.j],  ‖·‖₂=2.280e-01
  iter 87: subdiag=[-0.116-0.j  0.   +0.j -0.191-0.j],  ‖·‖₂=2.234e-01
  iter 88: subdiag=[0.113+0.j 0.   +0.j 0.188+0.j],  ‖·‖₂=2.189e-01
  iter 89: subdiag=[-0.11 -0.j  0.   +0.j -0.184+0.j],  ‖·‖₂=2.145e-01
  iter 90: subdiag=[0.107+0.j 0.   +0.j 0.181-0.j],  ‖·‖₂=2.102e-01
  iter 91: subdiag=[-0.104-0.j  0.   +0.j -0.178+0.j],  ‖·‖₂=2.060e-01
  iter 92: subdiag=[0.101+0.j 0.   +0.j 0.174-0.j],  ‖·‖₂=2.018e-01
  iter 93: subdiag=[-0.099-0.j  0.   +0.j -0.171+0.j],  ‖·‖₂=1.978e-01
  iter 94: subdiag=[0.096+0.j 0.   +0.j 0.168+0.j],  ‖·‖₂=1.938e-01
  iter 95: subdiag=[-0.094-0.j  0.   +0.j -0.165-0.j],  ‖·‖₂=1.899e-01
  iter 96: subdiag=[0.091+0.j 0.   +0.j 0.162+0.j],  ‖·‖₂=1.860e-01
  iter 97: subdiag=[-0.089-0.j  0.   +0.j -0.159-0.j],  ‖·‖₂=1.823e-01
  iter 98: subdiag=[0.087+0.j 0.   +0.j 0.156+0.j],  ‖·‖₂=1.786e-01
  iter 99: subdiag=[-0.085-0.j  0.   +0.j -0.153+0.j],  ‖·‖₂=1.750e-01
  after : subdiag=[-0.085-0.j  0.   +0.j -0.153+0.j],  ‖·‖₂=1.750e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.085 0.    0.153]
└─ converged?    = no

┌─ Matrix 14/30  (size 4x4)
│  fixed shift μ = 0.842538 (|μ|=0.8425)
  before: subdiag=[-0.944+0.j  0.964+0.j  0.696+0.j],  ‖·‖₂=1.518e+00
  iter 00: subdiag=[-0.662+0.j -0.644+0.j -0.413+0.j],  ‖·‖₂=1.012e+00
  iter 01: subdiag=[-0.628+0.j  0.166+0.j  0.387+0.j],  ‖·‖₂=7.558e-01
  iter 02: subdiag=[-0.626+0.j -0.038+0.j -0.386+0.j],  ‖·‖₂=7.360e-01
  iter 03: subdiag=[-0.626+0.j  0.009+0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 04: subdiag=[-0.626+0.j -0.002+0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 05: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 06: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 07: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 08: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 09: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 10: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 11: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 12: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 13: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 14: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 15: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 16: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 17: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 18: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 19: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 20: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 21: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 22: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 23: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 24: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 25: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 26: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 27: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 28: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 29: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 30: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 31: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 32: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 33: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 34: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 35: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 36: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 37: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 38: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 39: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 40: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 41: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 42: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 43: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 44: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 45: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 46: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 47: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 48: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 49: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 50: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 51: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 52: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 53: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 54: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 55: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 56: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 57: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 58: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 59: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 60: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 61: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 62: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 63: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 64: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 65: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 66: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 67: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 68: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 69: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 70: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 71: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 72: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 73: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 74: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 75: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 76: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 77: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 78: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 79: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 80: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 81: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 82: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 83: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 84: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 85: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 86: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 87: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 88: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 89: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 90: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 91: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 92: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 93: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 94: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 95: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 96: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 97: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  iter 98: subdiag=[-0.626+0.j -0.   +0.j -0.386+0.j],  ‖·‖₂=7.349e-01
  iter 99: subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
  after : subdiag=[-0.626+0.j  0.   +0.j  0.386+0.j],  ‖·‖₂=7.349e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.626 0.    0.386]
└─ converged?    = no

┌─ Matrix 15/30  (size 4x4)
│  fixed shift μ = 0.941615 (|μ|=0.9416)
  before: subdiag=[-0.745+0.j  0.93 +0.j -0.49 +0.j],  ‖·‖₂=1.288e+00
  iter 00: subdiag=[-0.99 +0.j  0.927+0.j -0.031+0.j],  ‖·‖₂=1.356e+00
  iter 01: subdiag=[-0.767+0.j  0.739+0.j -0.002+0.j],  ‖·‖₂=1.065e+00
  iter 02: subdiag=[-0.338+0.j  0.69 +0.j -0.   +0.j],  ‖·‖₂=7.680e-01
  iter 03: subdiag=[-0.127+0.j  0.682+0.j -0.   +0.j],  ‖·‖₂=6.941e-01
  iter 04: subdiag=[-0.047+0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.829e-01
  iter 05: subdiag=[-0.017+0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.813e-01
  iter 06: subdiag=[-0.006+0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 07: subdiag=[-0.002+0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 08: subdiag=[-0.001+0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 09: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 10: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 11: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 12: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 13: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 14: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 15: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 16: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 17: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 18: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 19: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 20: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 21: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 22: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 23: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 24: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 25: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 26: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 27: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 28: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 29: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 30: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 31: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 32: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 33: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 34: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 35: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 36: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 37: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 38: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 39: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 40: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 41: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 42: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 43: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 44: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 45: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 46: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 47: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 48: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 49: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 50: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 51: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 52: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 53: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 54: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 55: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 56: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 57: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 58: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 59: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 60: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 61: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 62: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 63: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 64: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 65: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 66: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 67: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 68: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 69: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 70: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 71: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 72: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 73: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 74: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 75: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 76: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 77: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 78: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 79: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 80: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 81: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 82: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 83: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 84: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 85: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 86: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 87: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 88: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 89: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 90: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 91: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 92: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 93: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 94: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 95: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 96: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 97: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 98: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  iter 99: subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
  after : subdiag=[-0.   +0.j  0.681+0.j -0.   +0.j],  ‖·‖₂=6.811e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.681 0.   ]
└─ converged?    = no

┌─ Matrix 16/30  (size 4x4)
│  fixed shift μ = 0.533835 (|μ|=0.5338)
  before: subdiag=[ 0.92 +0.j  0.998+0.j -0.89 +0.j],  ‖·‖₂=1.624e+00
  iter 00: subdiag=[ 0.987+0.j -0.962+0.j  0.616+0.j],  ‖·‖₂=1.510e+00
  iter 01: subdiag=[ 0.908+0.j  0.654+0.j -0.485+0.j],  ‖·‖₂=1.219e+00
  iter 02: subdiag=[ 0.877+0.j -0.318+0.j  0.453+0.j],  ‖·‖₂=1.037e+00
  iter 03: subdiag=[ 0.87 +0.j  0.139+0.j -0.447+0.j],  ‖·‖₂=9.880e-01
  iter 04: subdiag=[ 0.869+0.j -0.06 +0.j  0.446+0.j],  ‖·‖₂=9.783e-01
  iter 05: subdiag=[ 0.869+0.j  0.025+0.j -0.445+0.j],  ‖·‖₂=9.766e-01
  iter 06: subdiag=[ 0.869+0.j -0.011+0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 07: subdiag=[ 0.869+0.j  0.005+0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 08: subdiag=[ 0.869+0.j -0.002+0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 09: subdiag=[ 0.869+0.j  0.001+0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 10: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 11: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 12: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 13: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 14: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 15: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 16: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 17: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 18: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 19: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 20: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 21: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 22: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 23: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 24: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 25: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 26: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 27: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 28: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 29: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 30: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 31: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 32: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 33: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 34: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 35: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 36: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 37: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 38: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 39: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 40: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 41: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 42: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 43: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 44: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 45: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 46: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 47: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 48: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 49: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 50: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 51: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 52: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 53: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 54: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 55: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 56: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 57: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 58: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 59: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 60: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 61: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 62: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 63: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 64: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 65: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 66: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 67: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 68: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 69: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 70: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 71: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 72: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 73: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 74: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 75: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 76: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 77: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 78: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 79: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 80: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 81: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 82: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 83: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 84: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 85: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 86: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 87: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 88: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 89: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 90: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 91: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 92: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 93: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 94: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 95: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 96: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 97: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  iter 98: subdiag=[ 0.869+0.j -0.   +0.j  0.445+0.j],  ‖·‖₂=9.762e-01
  iter 99: subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
  after : subdiag=[ 0.869+0.j  0.   +0.j -0.445+0.j],  ‖·‖₂=9.762e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.869 0.    0.445]
└─ converged?    = no

┌─ Matrix 17/30  (size 4x4)
│  fixed shift μ = 0.0598199-0.0955454j (|μ|=0.1127)
  before: subdiag=[0.866+0.j 1.   +0.j 0.993+0.j],  ‖·‖₂=1.654e+00
  iter 00: subdiag=[-0.888+0.j -0.998-0.j  0.979+0.j],  ‖·‖₂=1.657e+00
  iter 01: subdiag=[0.904-0.j 0.995+0.j 0.951+0.j],  ‖·‖₂=1.647e+00
  iter 02: subdiag=[-0.913+0.j -0.992-0.j  0.912+0.j],  ‖·‖₂=1.627e+00
  iter 03: subdiag=[0.913-0.j 0.987+0.j 0.862+0.j],  ‖·‖₂=1.597e+00
  iter 04: subdiag=[-0.906+0.j -0.98 -0.j  0.808+0.j],  ‖·‖₂=1.560e+00
  iter 05: subdiag=[0.893-0.j 0.968+0.j 0.751+0.j],  ‖·‖₂=1.516e+00
  iter 06: subdiag=[-0.876+0.j -0.95 +0.j  0.696-0.j],  ‖·‖₂=1.468e+00
  iter 07: subdiag=[0.857-0.j 0.924-0.j 0.643-0.j],  ‖·‖₂=1.415e+00
  iter 08: subdiag=[-0.838+0.j -0.89 +0.j  0.595-0.j],  ‖·‖₂=1.359e+00
  iter 09: subdiag=[0.819-0.j 0.847+0.j 0.552-0.j],  ‖·‖₂=1.301e+00
  iter 10: subdiag=[ 0.801-0.j -0.798-0.j  0.513-0.j],  ‖·‖₂=1.241e+00
  iter 11: subdiag=[0.786-0.j 0.743+0.j 0.478-0.j],  ‖·‖₂=1.182e+00
  iter 12: subdiag=[ 0.772-0.j -0.686+0.j  0.447-0.j],  ‖·‖₂=1.125e+00
  iter 13: subdiag=[0.759-0.j 0.628+0.j 0.419-0.j],  ‖·‖₂=1.071e+00
  iter 14: subdiag=[ 0.749-0.j -0.571+0.j  0.394-0.j],  ‖·‖₂=1.020e+00
  iter 15: subdiag=[0.739-0.j 0.516-0.j 0.371-0.j],  ‖·‖₂=9.747e-01
  iter 16: subdiag=[ 0.731-0.j -0.464+0.j  0.35 -0.j],  ‖·‖₂=9.337e-01
  iter 17: subdiag=[0.724-0.j 0.416-0.j 0.33 -0.j],  ‖·‖₂=8.975e-01
  iter 18: subdiag=[ 0.717-0.j -0.371+0.j  0.312-0.j],  ‖·‖₂=8.656e-01
  iter 19: subdiag=[0.711-0.j 0.331-0.j 0.295-0.j],  ‖·‖₂=8.378e-01
  iter 20: subdiag=[ 0.705-0.j -0.294+0.j  0.279-0.j],  ‖·‖₂=8.136e-01
  iter 21: subdiag=[0.7  -0.j 0.261-0.j 0.264-0.j],  ‖·‖₂=7.925e-01
  iter 22: subdiag=[ 0.695-0.j -0.232+0.j  0.249-0.j],  ‖·‖₂=7.741e-01
  iter 23: subdiag=[0.69 -0.j 0.206-0.j 0.236-0.j],  ‖·‖₂=7.580e-01
  iter 24: subdiag=[ 0.686-0.j -0.182+0.j  0.223-0.j],  ‖·‖₂=7.437e-01
  iter 25: subdiag=[0.681-0.j 0.162-0.j 0.211-0.j],  ‖·‖₂=7.311e-01
  iter 26: subdiag=[ 0.677-0.j -0.143+0.j  0.2  -0.j],  ‖·‖₂=7.197e-01
  iter 27: subdiag=[0.672-0.j 0.127-0.j 0.189-0.j],  ‖·‖₂=7.095e-01
  iter 28: subdiag=[ 0.668-0.j -0.112+0.j  0.178-0.j],  ‖·‖₂=7.001e-01
  iter 29: subdiag=[0.663-0.j 0.099-0.j 0.168-0.j],  ‖·‖₂=6.915e-01
  iter 30: subdiag=[ 0.659-0.j -0.088+0.j  0.159-0.j],  ‖·‖₂=6.835e-01
  iter 31: subdiag=[0.655-0.j 0.078-0.j 0.15 -0.j],  ‖·‖₂=6.761e-01
  iter 32: subdiag=[ 0.65 -0.j -0.069+0.j  0.142-0.j],  ‖·‖₂=6.690e-01
  iter 33: subdiag=[0.646-0.j 0.061-0.j 0.134-0.j],  ‖·‖₂=6.624e-01
  iter 34: subdiag=[ 0.641-0.j -0.054+0.j  0.127-0.j],  ‖·‖₂=6.560e-01
  iter 35: subdiag=[0.637-0.j 0.048-0.j 0.12 -0.j],  ‖·‖₂=6.499e-01
  iter 36: subdiag=[ 0.633-0.j -0.043+0.j  0.113-0.j],  ‖·‖₂=6.440e-01
  iter 37: subdiag=[-0.628+0.j  0.038-0.j  0.107-0.j],  ‖·‖₂=6.382e-01
  iter 38: subdiag=[ 0.624-0.j -0.033+0.j  0.101-0.j],  ‖·‖₂=6.327e-01
  iter 39: subdiag=[-0.619+0.j  0.03 -0.j  0.095-0.j],  ‖·‖₂=6.272e-01
  iter 40: subdiag=[ 0.615-0.j -0.026+0.j  0.09 -0.j],  ‖·‖₂=6.219e-01
  iter 41: subdiag=[-0.61 +0.j  0.023-0.j  0.084-0.j],  ‖·‖₂=6.166e-01
  iter 42: subdiag=[ 0.606-0.j -0.021+0.j  0.08 -0.j],  ‖·‖₂=6.115e-01
  iter 43: subdiag=[-0.601-0.j  0.018-0.j  0.075-0.j],  ‖·‖₂=6.064e-01
  iter 44: subdiag=[ 0.597+0.j -0.016+0.j  0.071-0.j],  ‖·‖₂=6.014e-01
  iter 45: subdiag=[-0.592-0.j  0.014-0.j  0.067-0.j],  ‖·‖₂=5.964e-01
  iter 46: subdiag=[ 0.588+0.j -0.013+0.j  0.063-0.j],  ‖·‖₂=5.915e-01
  iter 47: subdiag=[-0.584-0.j  0.011-0.j  0.06 -0.j],  ‖·‖₂=5.866e-01
  iter 48: subdiag=[ 0.579+0.j -0.01 +0.j  0.056-0.j],  ‖·‖₂=5.818e-01
  iter 49: subdiag=[-0.575-0.j  0.009-0.j  0.053-0.j],  ‖·‖₂=5.770e-01
  iter 50: subdiag=[ 0.57 +0.j -0.008+0.j  0.05 -0.j],  ‖·‖₂=5.723e-01
  iter 51: subdiag=[-0.566+0.j  0.007-0.j  0.047-0.j],  ‖·‖₂=5.675e-01
  iter 52: subdiag=[ 0.561+0.j -0.006+0.j  0.044-0.j],  ‖·‖₂=5.628e-01
  iter 53: subdiag=[-0.557+0.j  0.005-0.j  0.042-0.j],  ‖·‖₂=5.581e-01
  iter 54: subdiag=[ 0.552-0.j -0.005+0.j  0.04 -0.j],  ‖·‖₂=5.535e-01
  iter 55: subdiag=[-0.548+0.j  0.004-0.j  0.037-0.j],  ‖·‖₂=5.488e-01
  iter 56: subdiag=[ 0.543-0.j -0.004+0.j  0.035-0.j],  ‖·‖₂=5.442e-01
  iter 57: subdiag=[-0.539+0.j  0.003-0.j  0.033-0.j],  ‖·‖₂=5.396e-01
  iter 58: subdiag=[ 0.534+0.j -0.003+0.j  0.031-0.j],  ‖·‖₂=5.351e-01
  iter 59: subdiag=[-0.53 -0.j  0.003-0.j  0.029-0.j],  ‖·‖₂=5.305e-01
  iter 60: subdiag=[ 0.525+0.j -0.002+0.j  0.028-0.j],  ‖·‖₂=5.260e-01
  iter 61: subdiag=[-0.521-0.j  0.002-0.j  0.026-0.j],  ‖·‖₂=5.215e-01
  iter 62: subdiag=[ 0.516+0.j -0.002+0.j  0.025-0.j],  ‖·‖₂=5.170e-01
  iter 63: subdiag=[-0.512-0.j  0.002-0.j  0.023-0.j],  ‖·‖₂=5.125e-01
  iter 64: subdiag=[ 0.508+0.j -0.001+0.j  0.022-0.j],  ‖·‖₂=5.080e-01
  iter 65: subdiag=[-0.503+0.j  0.001-0.j  0.021-0.j],  ‖·‖₂=5.036e-01
  iter 66: subdiag=[ 0.499-0.j -0.001+0.j  0.02 -0.j],  ‖·‖₂=4.992e-01
  iter 67: subdiag=[-0.494+0.j  0.001-0.j  0.018-0.j],  ‖·‖₂=4.948e-01
  iter 68: subdiag=[ 0.49 -0.j -0.001+0.j  0.017-0.j],  ‖·‖₂=4.904e-01
  iter 69: subdiag=[-0.486+0.j  0.001-0.j  0.016-0.j],  ‖·‖₂=4.860e-01
  iter 70: subdiag=[ 0.481-0.j -0.001+0.j  0.015-0.j],  ‖·‖₂=4.817e-01
  iter 71: subdiag=[-0.477+0.j  0.001-0.j  0.015-0.j],  ‖·‖₂=4.774e-01
  iter 72: subdiag=[ 0.473-0.j -0.001+0.j  0.014-0.j],  ‖·‖₂=4.731e-01
  iter 73: subdiag=[-0.469+0.j  0.   -0.j  0.013-0.j],  ‖·‖₂=4.688e-01
  iter 74: subdiag=[ 0.464-0.j -0.   +0.j  0.012-0.j],  ‖·‖₂=4.645e-01
  iter 75: subdiag=[-0.46 +0.j  0.   -0.j  0.012-0.j],  ‖·‖₂=4.603e-01
  iter 76: subdiag=[ 0.456-0.j -0.   +0.j  0.011-0.j],  ‖·‖₂=4.561e-01
  iter 77: subdiag=[-0.452-0.j  0.   -0.j  0.01 -0.j],  ‖·‖₂=4.519e-01
  iter 78: subdiag=[ 0.448+0.j -0.   +0.j  0.01 -0.j],  ‖·‖₂=4.477e-01
  iter 79: subdiag=[-0.443-0.j  0.   -0.j  0.009-0.j],  ‖·‖₂=4.436e-01
  iter 80: subdiag=[ 0.439+0.j -0.   +0.j  0.009-0.j],  ‖·‖₂=4.394e-01
  iter 81: subdiag=[-0.435-0.j  0.   -0.j  0.008-0.j],  ‖·‖₂=4.353e-01
  iter 82: subdiag=[ 0.431+0.j -0.   +0.j  0.008-0.j],  ‖·‖₂=4.313e-01
  iter 83: subdiag=[-0.427-0.j  0.   -0.j  0.007-0.j],  ‖·‖₂=4.272e-01
  iter 84: subdiag=[ 0.423+0.j -0.   +0.j  0.007-0.j],  ‖·‖₂=4.232e-01
  iter 85: subdiag=[-0.419-0.j  0.   -0.j  0.006-0.j],  ‖·‖₂=4.192e-01
  iter 86: subdiag=[ 0.415+0.j -0.   +0.j  0.006-0.j],  ‖·‖₂=4.152e-01
  iter 87: subdiag=[-0.411-0.j  0.   -0.j  0.006-0.j],  ‖·‖₂=4.112e-01
  iter 88: subdiag=[ 0.407+0.j -0.   +0.j  0.005-0.j],  ‖·‖₂=4.073e-01
  iter 89: subdiag=[-0.403-0.j  0.   -0.j  0.005-0.j],  ‖·‖₂=4.034e-01
  iter 90: subdiag=[ 0.399+0.j -0.   +0.j  0.005-0.j],  ‖·‖₂=3.995e-01
  iter 91: subdiag=[-0.396-0.j  0.   -0.j  0.005-0.j],  ‖·‖₂=3.957e-01
  iter 92: subdiag=[ 0.392+0.j -0.   +0.j  0.004-0.j],  ‖·‖₂=3.918e-01
  iter 93: subdiag=[-0.388-0.j  0.   -0.j  0.004-0.j],  ‖·‖₂=3.880e-01
  iter 94: subdiag=[ 0.384+0.j -0.   +0.j  0.004-0.j],  ‖·‖₂=3.842e-01
  iter 95: subdiag=[-0.38 -0.j  0.   -0.j  0.004-0.j],  ‖·‖₂=3.805e-01
  iter 96: subdiag=[ 0.377+0.j -0.   +0.j  0.003-0.j],  ‖·‖₂=3.768e-01
  iter 97: subdiag=[-0.373+0.j  0.   -0.j  0.003-0.j],  ‖·‖₂=3.731e-01
  iter 98: subdiag=[ 0.369+0.j -0.   +0.j  0.003-0.j],  ‖·‖₂=3.694e-01
  iter 99: subdiag=[-0.366-0.j  0.   -0.j  0.003-0.j],  ‖·‖₂=3.658e-01
  after : subdiag=[-0.366-0.j  0.   -0.j  0.003-0.j],  ‖·‖₂=3.658e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.366 0.    0.003]
└─ converged?    = no

┌─ Matrix 18/30  (size 4x4)
│  fixed shift μ = 0.0412469-0.965376j (|μ|=0.9663)
  before: subdiag=[-0.986+0.j -0.358+0.j -0.999+0.j],  ‖·‖₂=1.449e+00
  iter 00: subdiag=[-0.281-0.j  0.975-0.j -0.203-0.j],  ‖·‖₂=1.035e+00
  iter 01: subdiag=[-0.184-0.j  0.285-0.j  0.078+0.j],  ‖·‖₂=3.481e-01
  iter 02: subdiag=[ 0.183+0.j -0.037+0.j -0.043-0.j],  ‖·‖₂=1.915e-01
  iter 03: subdiag=[-0.184-0.j  0.005-0.j  0.023+0.j],  ‖·‖₂=1.853e-01
  iter 04: subdiag=[ 0.185+0.j -0.001+0.j -0.012-0.j],  ‖·‖₂=1.850e-01
  iter 05: subdiag=[ 0.185+0.j -0.   +0.j  0.007+0.j],  ‖·‖₂=1.854e-01
  iter 06: subdiag=[ 0.186+0.j -0.   +0.j -0.004-0.j],  ‖·‖₂=1.860e-01
  iter 07: subdiag=[ 0.187+0.j -0.   +0.j  0.002+0.j],  ‖·‖₂=1.866e-01
  iter 08: subdiag=[ 0.187+0.j -0.   +0.j -0.001-0.j],  ‖·‖₂=1.872e-01
  iter 09: subdiag=[ 0.188+0.j -0.   +0.j  0.001+0.j],  ‖·‖₂=1.878e-01
  iter 10: subdiag=[ 0.188+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.884e-01
  iter 11: subdiag=[ 0.189+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.889e-01
  iter 12: subdiag=[ 0.189+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.894e-01
  iter 13: subdiag=[ 0.19+0.j -0.  +0.j  0.  +0.j],  ‖·‖₂=1.898e-01
  iter 14: subdiag=[ 0.19+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.902e-01
  iter 15: subdiag=[ 0.191+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.906e-01
  iter 16: subdiag=[ 0.191+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.909e-01
  iter 17: subdiag=[ 0.191+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.912e-01
  iter 18: subdiag=[ 0.191+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.915e-01
  iter 19: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.917e-01
  iter 20: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.919e-01
  iter 21: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.920e-01
  iter 22: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.921e-01
  iter 23: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.922e-01
  iter 24: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.922e-01
  iter 25: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.922e-01
  iter 26: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.921e-01
  iter 27: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.920e-01
  iter 28: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.919e-01
  iter 29: subdiag=[ 0.192+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.918e-01
  iter 30: subdiag=[ 0.192+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.915e-01
  iter 31: subdiag=[ 0.191+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.913e-01
  iter 32: subdiag=[ 0.191+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.910e-01
  iter 33: subdiag=[ 0.191+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.907e-01
  iter 34: subdiag=[ 0.19+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.903e-01
  iter 35: subdiag=[ 0.19+0.j -0.  +0.j  0.  +0.j],  ‖·‖₂=1.900e-01
  iter 36: subdiag=[ 0.19+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.895e-01
  iter 37: subdiag=[ 0.189+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.891e-01
  iter 38: subdiag=[ 0.189+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.886e-01
  iter 39: subdiag=[ 0.188+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.880e-01
  iter 40: subdiag=[ 0.187+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.875e-01
  iter 41: subdiag=[ 0.187+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.869e-01
  iter 42: subdiag=[ 0.186+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.862e-01
  iter 43: subdiag=[ 0.186+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.855e-01
  iter 44: subdiag=[ 0.185+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.848e-01
  iter 45: subdiag=[ 0.184+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.841e-01
  iter 46: subdiag=[ 0.183+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.833e-01
  iter 47: subdiag=[ 0.183+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.825e-01
  iter 48: subdiag=[ 0.182+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.817e-01
  iter 49: subdiag=[-0.181-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.809e-01
  iter 50: subdiag=[ 0.18+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.800e-01
  iter 51: subdiag=[-0.179-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.791e-01
  iter 52: subdiag=[ 0.178+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.781e-01
  iter 53: subdiag=[-0.177-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.772e-01
  iter 54: subdiag=[ 0.176+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.762e-01
  iter 55: subdiag=[-0.175-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.752e-01
  iter 56: subdiag=[ 0.174+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.741e-01
  iter 57: subdiag=[-0.173-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.731e-01
  iter 58: subdiag=[ 0.172+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.720e-01
  iter 59: subdiag=[-0.171-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.709e-01
  iter 60: subdiag=[ 0.17+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.698e-01
  iter 61: subdiag=[-0.169-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.687e-01
  iter 62: subdiag=[ 0.167+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.675e-01
  iter 63: subdiag=[-0.166-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.663e-01
  iter 64: subdiag=[ 0.165+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.651e-01
  iter 65: subdiag=[-0.164-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.639e-01
  iter 66: subdiag=[ 0.163+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.627e-01
  iter 67: subdiag=[-0.161-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.615e-01
  iter 68: subdiag=[ 0.16+0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.602e-01
  iter 69: subdiag=[-0.159-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.590e-01
  iter 70: subdiag=[ 0.158+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.577e-01
  iter 71: subdiag=[-0.156-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.564e-01
  iter 72: subdiag=[ 0.155-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.551e-01
  iter 73: subdiag=[-0.154-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.538e-01
  iter 74: subdiag=[ 0.152-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.525e-01
  iter 75: subdiag=[-0.151-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.512e-01
  iter 76: subdiag=[ 0.15-0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.498e-01
  iter 77: subdiag=[-0.148-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.485e-01
  iter 78: subdiag=[ 0.147+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.472e-01
  iter 79: subdiag=[-0.146+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.458e-01
  iter 80: subdiag=[ 0.144-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.445e-01
  iter 81: subdiag=[-0.143-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.431e-01
  iter 82: subdiag=[ 0.142+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.418e-01
  iter 83: subdiag=[-0.14-0.j -0.  +0.j  0.  +0.j],  ‖·‖₂=1.404e-01
  iter 84: subdiag=[ 0.139+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.390e-01
  iter 85: subdiag=[-0.138-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.377e-01
  iter 86: subdiag=[ 0.136-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.363e-01
  iter 87: subdiag=[-0.135+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.349e-01
  iter 88: subdiag=[ 0.134-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.336e-01
  iter 89: subdiag=[-0.132+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.322e-01
  iter 90: subdiag=[ 0.131+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.309e-01
  iter 91: subdiag=[-0.129-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.295e-01
  iter 92: subdiag=[ 0.128-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.281e-01
  iter 93: subdiag=[-0.127+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.268e-01
  iter 94: subdiag=[ 0.125+0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.254e-01
  iter 95: subdiag=[-0.124+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.241e-01
  iter 96: subdiag=[ 0.123-0.j -0.   +0.j -0.   -0.j],  ‖·‖₂=1.227e-01
  iter 97: subdiag=[-0.121+0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.214e-01
  iter 98: subdiag=[ 0.12-0.j -0.  +0.j -0.  -0.j],  ‖·‖₂=1.201e-01
  iter 99: subdiag=[-0.119-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.187e-01
  after : subdiag=[-0.119-0.j -0.   +0.j  0.   +0.j],  ‖·‖₂=1.187e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.119 0.    0.   ]
└─ converged?    = no

┌─ Matrix 19/30  (size 4x4)
│  fixed shift μ = -0.993087 (|μ|=0.9931)
  before: subdiag=[-0.745+0.j  0.986+0.j  0.139+0.j],  ‖·‖₂=1.244e+00
  iter 00: subdiag=[-0.592+0.j  0.997+0.j  0.001+0.j],  ‖·‖₂=1.159e+00
  iter 01: subdiag=[-0.445+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.094e+00
  iter 02: subdiag=[-0.324+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.051e+00
  iter 03: subdiag=[-0.231+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.026e+00
  iter 04: subdiag=[-0.163+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.013e+00
  iter 05: subdiag=[-0.114+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.006e+00
  iter 06: subdiag=[-0.08+0.j  1.  +0.j  0.  +0.j],  ‖·‖₂=1.003e+00
  iter 07: subdiag=[-0.056+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.001e+00
  iter 08: subdiag=[-0.039+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.000e+00
  iter 09: subdiag=[-0.027+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=1.000e+00
  iter 10: subdiag=[-0.019+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.999e-01
  iter 11: subdiag=[-0.013+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.998e-01
  iter 12: subdiag=[-0.009+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 13: subdiag=[-0.006+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 14: subdiag=[-0.005+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 15: subdiag=[-0.003+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 16: subdiag=[-0.002+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 17: subdiag=[-0.002+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 18: subdiag=[-0.001+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 19: subdiag=[-0.001+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 20: subdiag=[-0.001+0.j  1.   +0.j  0.   +0.j],  ‖·‖₂=9.997e-01
  iter 21: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 22: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 23: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 24: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 25: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 26: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 27: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 28: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 29: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 30: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 31: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 32: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 33: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 34: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 35: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 36: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 37: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 38: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 39: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 40: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 41: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 42: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 43: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 44: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 45: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 46: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 47: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 48: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 49: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 50: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 51: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 52: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 53: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 54: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 55: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 56: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 57: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 58: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 59: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 60: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 61: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 62: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 63: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 64: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 65: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 66: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 67: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 68: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 69: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 70: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 71: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 72: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 73: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 74: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 75: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 76: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 77: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 78: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 79: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 80: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 81: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 82: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 83: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 84: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 85: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 86: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 87: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 88: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 89: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 90: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 91: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 92: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 93: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 94: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 95: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 96: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 97: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 98: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  iter 99: subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
  after : subdiag=[-0.+0.j  1.+0.j  0.+0.j],  ‖·‖₂=9.997e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0. 1. 0.]
└─ converged?    = no

┌─ Matrix 20/30  (size 4x4)
│  fixed shift μ = 0.943718 (|μ|=0.9437)
  before: subdiag=[-0.429+0.j -0.972+0.j  0.419+0.j],  ‖·‖₂=1.142e+00
  iter 00: subdiag=[-0.982+0.j -0.64 +0.j  0.047+0.j],  ‖·‖₂=1.173e+00
  iter 01: subdiag=[-0.51 +0.j -0.44 +0.j  0.006+0.j],  ‖·‖₂=6.740e-01
  iter 02: subdiag=[-0.12 +0.j -0.426+0.j  0.001+0.j],  ‖·‖₂=4.427e-01
  iter 03: subdiag=[-0.027+0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.262e-01
  iter 04: subdiag=[-0.006+0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 05: subdiag=[-0.001+0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 06: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 07: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 08: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 09: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 10: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 11: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 12: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 13: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 14: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 15: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 16: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 17: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 18: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 19: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 20: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 21: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 22: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 23: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 24: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 25: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 26: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 27: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 28: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 29: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 30: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 31: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 32: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 33: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 34: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 35: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 36: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 37: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 38: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 39: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 40: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 41: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 42: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 43: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 44: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 45: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 46: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 47: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 48: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 49: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 50: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 51: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 52: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 53: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 54: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 55: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 56: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 57: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 58: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 59: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 60: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 61: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 62: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 63: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 64: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 65: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 66: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 67: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 68: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 69: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 70: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 71: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 72: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 73: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 74: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 75: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 76: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 77: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 78: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 79: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 80: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 81: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 82: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 83: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 84: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 85: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 86: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 87: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 88: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 89: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 90: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 91: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 92: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 93: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 94: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 95: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 96: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 97: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 98: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  iter 99: subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
  after : subdiag=[-0.   +0.j -0.425+0.j  0.   +0.j],  ‖·‖₂=4.253e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.425 0.   ]
└─ converged?    = no

┌─ Matrix 21/30  (size 4x4)
│  fixed shift μ = -0.645528 (|μ|=0.6455)
  before: subdiag=[-0.998+0.j  0.999+0.j  0.737+0.j],  ‖·‖₂=1.593e+00
  iter 00: subdiag=[-0.902+0.j  0.984+0.j  0.314+0.j],  ‖·‖₂=1.371e+00
  iter 01: subdiag=[-0.666+0.j  0.955+0.j  0.114+0.j],  ‖·‖₂=1.169e+00
  iter 02: subdiag=[-0.431+0.j  0.936+0.j  0.041+0.j],  ‖·‖₂=1.032e+00
  iter 03: subdiag=[-0.262+0.j  0.928+0.j  0.015+0.j],  ‖·‖₂=9.647e-01
  iter 04: subdiag=[-0.155+0.j  0.925+0.j  0.006+0.j],  ‖·‖₂=9.383e-01
  iter 05: subdiag=[-0.091+0.j  0.924+0.j  0.002+0.j],  ‖·‖₂=9.288e-01
  iter 06: subdiag=[-0.053+0.j  0.924+0.j  0.001+0.j],  ‖·‖₂=9.255e-01
  iter 07: subdiag=[-0.031+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.244e-01
  iter 08: subdiag=[-0.018+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.240e-01
  iter 09: subdiag=[-0.011+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.239e-01
  iter 10: subdiag=[-0.006+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.239e-01
  iter 11: subdiag=[-0.004+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 12: subdiag=[-0.002+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 13: subdiag=[-0.001+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 14: subdiag=[-0.001+0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 15: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 16: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 17: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 18: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 19: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 20: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 21: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 22: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 23: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 24: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 25: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 26: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 27: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 28: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 29: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 30: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 31: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 32: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 33: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 34: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 35: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 36: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 37: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 38: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 39: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 40: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 41: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 42: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 43: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 44: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 45: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 46: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 47: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 48: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 49: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 50: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 51: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 52: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 53: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 54: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 55: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 56: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 57: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 58: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 59: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 60: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 61: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 62: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 63: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 64: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 65: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 66: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 67: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 68: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 69: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 70: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 71: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 72: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 73: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 74: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 75: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 76: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 77: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 78: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 79: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 80: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 81: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 82: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 83: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 84: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 85: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 86: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 87: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 88: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 89: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 90: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 91: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 92: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 93: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 94: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 95: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 96: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 97: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 98: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  iter 99: subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
  after : subdiag=[-0.   +0.j  0.924+0.j  0.   +0.j],  ‖·‖₂=9.238e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.924 0.   ]
└─ converged?    = no

┌─ Matrix 22/30  (size 4x4)
│  fixed shift μ = 0.833114 (|μ|=0.8331)
  before: subdiag=[-0.476+0.j  0.996+0.j  0.602+0.j],  ‖·‖₂=1.258e+00
  iter 00: subdiag=[-0.329+0.j -0.495+0.j -0.426+0.j],  ‖·‖₂=7.312e-01
  iter 01: subdiag=[-0.318+0.j  0.123+0.j  0.413+0.j],  ‖·‖₂=5.355e-01
  iter 02: subdiag=[-0.318+0.j -0.029+0.j -0.412+0.j],  ‖·‖₂=5.211e-01
  iter 03: subdiag=[-0.318+0.j  0.007+0.j  0.412+0.j],  ‖·‖₂=5.203e-01
  iter 04: subdiag=[-0.318+0.j -0.002+0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 05: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 06: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 07: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 08: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 09: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 10: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 11: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 12: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 13: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 14: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 15: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 16: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 17: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 18: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 19: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 20: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 21: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 22: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 23: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 24: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 25: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 26: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 27: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 28: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 29: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 30: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 31: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 32: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 33: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 34: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 35: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 36: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 37: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 38: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 39: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 40: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 41: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 42: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 43: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 44: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 45: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 46: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 47: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 48: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 49: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 50: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 51: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 52: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 53: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 54: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 55: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 56: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 57: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 58: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 59: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 60: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 61: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 62: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 63: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 64: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 65: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 66: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 67: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 68: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 69: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 70: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 71: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 72: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 73: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 74: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 75: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 76: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 77: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 78: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 79: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 80: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 81: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 82: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 83: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 84: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 85: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 86: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 87: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 88: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 89: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 90: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 91: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 92: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 93: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 94: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 95: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 96: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 97: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  iter 98: subdiag=[-0.318+0.j -0.   +0.j -0.412+0.j],  ‖·‖₂=5.202e-01
  iter 99: subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
  after : subdiag=[-0.318+0.j  0.   +0.j  0.412+0.j],  ‖·‖₂=5.202e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.318 0.    0.412]
└─ converged?    = no

┌─ Matrix 23/30  (size 4x4)
│  fixed shift μ = 0.475111-0.570494j (|μ|=0.7424)
  before: subdiag=[0.907+0.j 0.834+0.j 0.79 +0.j],  ‖·‖₂=1.464e+00
  iter 00: subdiag=[ 0.727-0.j -0.911-0.j  0.447+0.j],  ‖·‖₂=1.248e+00
  iter 01: subdiag=[ 0.484-0.j -0.857-0.j -0.193-0.j],  ‖·‖₂=1.003e+00
  iter 02: subdiag=[-0.345+0.j  0.585+0.j  0.093+0.j],  ‖·‖₂=6.857e-01
  iter 03: subdiag=[ 0.253-0.j -0.345-0.j -0.048+0.j],  ‖·‖₂=4.310e-01
  iter 04: subdiag=[-0.185+0.j  0.195+0.j  0.026-0.j],  ‖·‖₂=2.705e-01
  iter 05: subdiag=[ 0.135-0.j -0.109-0.j -0.014-0.j],  ‖·‖₂=1.741e-01
  iter 06: subdiag=[-0.097+0.j  0.061+0.j  0.007+0.j],  ‖·‖₂=1.152e-01
  iter 07: subdiag=[ 0.07 -0.j -0.034-0.j -0.004-0.j],  ‖·‖₂=7.812e-02
  iter 08: subdiag=[-0.05 +0.j  0.019+0.j  0.002-0.j],  ‖·‖₂=5.393e-02
  iter 09: subdiag=[ 0.036-0.j -0.011-0.j -0.001+0.j],  ‖·‖₂=3.772e-02
  iter 10: subdiag=[-0.026+0.j  0.006+0.j  0.001+0.j],  ‖·‖₂=2.661e-02
  iter 11: subdiag=[ 0.019-0.j -0.003-0.j -0.   -0.j],  ‖·‖₂=1.888e-02
  iter 12: subdiag=[-0.013+0.j  0.002+0.j  0.   +0.j],  ‖·‖₂=1.345e-02
  iter 13: subdiag=[ 0.01 -0.j -0.001-0.j -0.   -0.j],  ‖·‖₂=9.598e-03
  iter 14: subdiag=[-0.007+0.j  0.001+0.j  0.   +0.j],  ‖·‖₂=6.862e-03
  iter 15: subdiag=[ 0.005-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=4.910e-03
  iter 16: subdiag=[-0.004+0.j  0.   +0.j  0.   +0.j],  ‖·‖₂=3.515e-03
  iter 17: subdiag=[ 0.003-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=2.517e-03
  iter 18: subdiag=[-0.002+0.j  0.   +0.j  0.   +0.j],  ‖·‖₂=1.803e-03
  iter 19: subdiag=[ 0.001-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=1.292e-03
  iter 20: subdiag=[-0.001+0.j  0.   +0.j  0.   +0.j],  ‖·‖₂=9.255e-04
  iter 21: subdiag=[ 0.001-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=6.631e-04
  iter 22: subdiag=[-0.+0.j  0.+0.j  0.+0.j],  ‖·‖₂=4.752e-04
  iter 23: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=3.405e-04
  iter 24: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=2.440e-04
  iter 25: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.748e-04
  iter 26: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.253e-04
  iter 27: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=8.975e-05
  iter 28: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=6.431e-05
  iter 29: subdiag=[ 0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=4.608e-05
  iter 30: subdiag=[-0.+0.j  0.+0.j  0.+0.j],  ‖·‖₂=3.302e-05
  iter 31: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.366e-05
  iter 32: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.695e-05
  iter 33: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.215e-05
  iter 34: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=8.705e-06
  iter 35: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=6.238e-06
  iter 36: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=4.470e-06
  iter 37: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=3.203e-06
  iter 38: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=2.295e-06
  iter 39: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.645e-06
  iter 40: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.178e-06
  iter 41: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=8.444e-07
  iter 42: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=6.050e-07
  iter 43: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=4.335e-07
  iter 44: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=3.107e-07
  iter 45: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.226e-07
  iter 46: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.595e-07
  iter 47: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.143e-07
  iter 48: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=8.190e-08
  iter 49: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=5.868e-08
  iter 50: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=4.205e-08
  iter 51: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=3.013e-08
  iter 52: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=2.159e-08
  iter 53: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.547e-08
  iter 54: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.109e-08
  iter 55: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=7.944e-09
  iter 56: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=5.692e-09
  iter 57: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=4.079e-09
  iter 58: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=2.923e-09
  iter 59: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.094e-09
  iter 60: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.501e-09
  iter 61: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.075e-09
  iter 62: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=7.705e-10
  iter 63: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=5.521e-10
  iter 64: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=3.956e-10
  iter 65: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=2.835e-10
  iter 66: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=2.031e-10
  iter 67: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=1.455e-10
  iter 68: subdiag=[-0.+0.j  0.+0.j  0.-0.j],  ‖·‖₂=1.043e-10
  iter 69: subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=7.473e-11
  after : subdiag=[ 0.-0.j -0.-0.j -0.+0.j],  ‖·‖₂=7.473e-11
│  iterations    = 70/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = yes

┌─ Matrix 24/30  (size 4x4)
│  fixed shift μ = 0.921576-0.227859j (|μ|=0.9493)
  before: subdiag=[-0.916+0.j  0.433+0.j -0.245+0.j],  ‖·‖₂=1.042e+00
  iter 00: subdiag=[-0.888+0.j -0.146-0.j -0.052-0.j],  ‖·‖₂=9.016e-01
  iter 01: subdiag=[-0.756-0.j  0.066+0.j -0.006-0.j],  ‖·‖₂=7.589e-01
  iter 02: subdiag=[-0.595+0.j -0.031-0.j -0.001+0.j],  ‖·‖₂=5.955e-01
  iter 03: subdiag=[-0.446-0.j  0.015+0.j -0.   +0.j],  ‖·‖₂=4.459e-01
  iter 04: subdiag=[-0.325-0.j -0.007-0.j -0.   +0.j],  ‖·‖₂=3.251e-01
  iter 05: subdiag=[-0.234-0.j  0.003+0.j -0.   -0.j],  ‖·‖₂=2.336e-01
  iter 06: subdiag=[-0.167-0.j -0.002-0.j -0.   -0.j],  ‖·‖₂=1.666e-01
  iter 07: subdiag=[-0.118-0.j  0.001+0.j -0.   -0.j],  ‖·‖₂=1.184e-01
  iter 08: subdiag=[-0.084-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=8.395e-02
  iter 09: subdiag=[-0.059-0.j  0.   +0.j -0.   -0.j],  ‖·‖₂=5.947e-02
  iter 10: subdiag=[-0.042-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=4.211e-02
  iter 11: subdiag=[-0.03-0.j  0.  +0.j -0.  +0.j],  ‖·‖₂=2.981e-02
  iter 12: subdiag=[-0.021-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=2.110e-02
  iter 13: subdiag=[-0.015+0.j  0.   +0.j -0.   +0.j],  ‖·‖₂=1.493e-02
  iter 14: subdiag=[-0.011-0.j -0.   -0.j -0.   +0.j],  ‖·‖₂=1.057e-02
  iter 15: subdiag=[-0.007-0.j  0.   +0.j -0.   -0.j],  ‖·‖₂=7.480e-03
  iter 16: subdiag=[-0.005-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=5.294e-03
  iter 17: subdiag=[-0.004-0.j  0.   +0.j -0.   -0.j],  ‖·‖₂=3.747e-03
  iter 18: subdiag=[-0.003-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=2.652e-03
  iter 19: subdiag=[-0.002-0.j  0.   +0.j -0.   -0.j],  ‖·‖₂=1.877e-03
  iter 20: subdiag=[-0.001-0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=1.328e-03
  iter 21: subdiag=[-0.001+0.j  0.   +0.j -0.   -0.j],  ‖·‖₂=9.400e-04
  iter 22: subdiag=[-0.001+0.j -0.   -0.j -0.   -0.j],  ‖·‖₂=6.652e-04
  iter 23: subdiag=[-0.+0.j  0.+0.j -0.+0.j],  ‖·‖₂=4.708e-04
  iter 24: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=3.332e-04
  iter 25: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.358e-04
  iter 26: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.669e-04
  iter 27: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.181e-04
  iter 28: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=8.359e-05
  iter 29: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=5.916e-05
  iter 30: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=4.187e-05
  iter 31: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.963e-05
  iter 32: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=2.097e-05
  iter 33: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.484e-05
  iter 34: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.050e-05
  iter 35: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=7.434e-06
  iter 36: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=5.261e-06
  iter 37: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=3.724e-06
  iter 38: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=2.635e-06
  iter 39: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.865e-06
  iter 40: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.320e-06
  iter 41: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=9.342e-07
  iter 42: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=6.611e-07
  iter 43: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=4.679e-07
  iter 44: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=3.311e-07
  iter 45: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.344e-07
  iter 46: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.659e-07
  iter 47: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.174e-07
  iter 48: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=8.308e-08
  iter 49: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=5.880e-08
  iter 50: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=4.161e-08
  iter 51: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.945e-08
  iter 52: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=2.084e-08
  iter 53: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.475e-08
  iter 54: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.044e-08
  iter 55: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=7.388e-09
  iter 56: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=5.229e-09
  iter 57: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=3.701e-09
  iter 58: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=2.619e-09
  iter 59: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.854e-09
  iter 60: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.312e-09
  iter 61: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=9.284e-10
  iter 62: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=6.570e-10
  iter 63: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=4.650e-10
  iter 64: subdiag=[-0.-0.j -0.-0.j -0.-0.j],  ‖·‖₂=3.291e-10
  iter 65: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.329e-10
  iter 66: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=1.648e-10
  iter 67: subdiag=[-0.+0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.167e-10
  iter 68: subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=8.256e-11
  after : subdiag=[-0.+0.j -0.-0.j -0.-0.j],  ‖·‖₂=8.256e-11
│  iterations    = 69/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = yes

┌─ Matrix 25/30  (size 4x4)
│  fixed shift μ = -0.65739-0.393338j (|μ|=0.7661)
  before: subdiag=[ 0.282+0.j  0.81 +0.j -0.56 +0.j],  ‖·‖₂=1.024e+00
  iter 00: subdiag=[-0.423+0.j  0.699+0.j  0.406+0.j],  ‖·‖₂=9.128e-01
  iter 01: subdiag=[ 0.525-0.j  0.476+0.j -0.313+0.j],  ‖·‖₂=7.751e-01
  iter 02: subdiag=[ 0.389-0.j -0.366-0.j  0.245+0.j],  ‖·‖₂=5.879e-01
  iter 03: subdiag=[-0.207-0.j  0.318+0.j -0.189+0.j],  ‖·‖₂=4.242e-01
  iter 04: subdiag=[ 0.1  -0.j -0.287-0.j  0.143+0.j],  ‖·‖₂=3.359e-01
  iter 05: subdiag=[-0.047+0.j  0.259+0.j -0.108-0.j],  ‖·‖₂=2.846e-01
  iter 06: subdiag=[ 0.022+0.j -0.233-0.j  0.081+0.j],  ‖·‖₂=2.481e-01
  iter 07: subdiag=[-0.011-0.j  0.21 +0.j -0.061-0.j],  ‖·‖₂=2.187e-01
  iter 08: subdiag=[ 0.005-0.j -0.188+0.j  0.045+0.j],  ‖·‖₂=1.937e-01
  iter 09: subdiag=[-0.002-0.j  0.169+0.j -0.034-0.j],  ‖·‖₂=1.721e-01
  iter 10: subdiag=[ 0.001+0.j -0.151-0.j  0.025+0.j],  ‖·‖₂=1.531e-01
  iter 11: subdiag=[-0.001-0.j  0.135-0.j -0.019+0.j],  ‖·‖₂=1.363e-01
  iter 12: subdiag=[ 0.   +0.j -0.121-0.j  0.014+0.j],  ‖·‖₂=1.215e-01
  iter 13: subdiag=[-0.   +0.j  0.108+0.j -0.011+0.j],  ‖·‖₂=1.083e-01
  iter 14: subdiag=[ 0.   +0.j -0.096+0.j  0.008+0.j],  ‖·‖₂=9.653e-02
  iter 15: subdiag=[-0.   -0.j  0.086+0.j -0.006-0.j],  ‖·‖₂=8.606e-02
  iter 16: subdiag=[ 0.   +0.j -0.077-0.j  0.004+0.j],  ‖·‖₂=7.673e-02
  iter 17: subdiag=[-0.   -0.j  0.068+0.j -0.003-0.j],  ‖·‖₂=6.842e-02
  iter 18: subdiag=[ 0.   +0.j -0.061-0.j  0.003+0.j],  ‖·‖₂=6.100e-02
  iter 19: subdiag=[-0.   +0.j  0.054+0.j -0.002+0.j],  ‖·‖₂=5.438e-02
  iter 20: subdiag=[ 0.   -0.j -0.048-0.j  0.001+0.j],  ‖·‖₂=4.848e-02
  iter 21: subdiag=[-0.   -0.j  0.043-0.j -0.001+0.j],  ‖·‖₂=4.322e-02
  iter 22: subdiag=[ 0.   +0.j -0.039-0.j  0.001+0.j],  ‖·‖₂=3.853e-02
  iter 23: subdiag=[-0.   -0.j  0.034+0.j -0.001+0.j],  ‖·‖₂=3.434e-02
  iter 24: subdiag=[ 0.   +0.j -0.031-0.j  0.   +0.j],  ‖·‖₂=3.061e-02
  iter 25: subdiag=[-0.   -0.j  0.027+0.j -0.   +0.j],  ‖·‖₂=2.729e-02
  iter 26: subdiag=[ 0.   +0.j -0.024-0.j  0.   +0.j],  ‖·‖₂=2.432e-02
  iter 27: subdiag=[-0.   -0.j  0.022+0.j -0.   +0.j],  ‖·‖₂=2.168e-02
  iter 28: subdiag=[ 0.   -0.j -0.019-0.j  0.   -0.j],  ‖·‖₂=1.932e-02
  iter 29: subdiag=[-0.   +0.j  0.017-0.j -0.   +0.j],  ‖·‖₂=1.722e-02
  iter 30: subdiag=[ 0.   -0.j -0.015-0.j  0.   -0.j],  ‖·‖₂=1.535e-02
  iter 31: subdiag=[-0.   +0.j  0.014+0.j -0.   +0.j],  ‖·‖₂=1.368e-02
  iter 32: subdiag=[ 0.   +0.j -0.012-0.j  0.   +0.j],  ‖·‖₂=1.220e-02
  iter 33: subdiag=[-0.   -0.j  0.011+0.j -0.   +0.j],  ‖·‖₂=1.087e-02
  iter 34: subdiag=[ 0.  +0.j -0.01-0.j  0.  +0.j],  ‖·‖₂=9.688e-03
  iter 35: subdiag=[-0.   -0.j  0.009+0.j -0.   -0.j],  ‖·‖₂=8.635e-03
  iter 36: subdiag=[ 0.   +0.j -0.008-0.j  0.   +0.j],  ‖·‖₂=7.696e-03
  iter 37: subdiag=[-0.   -0.j  0.007+0.j -0.   -0.j],  ‖·‖₂=6.860e-03
  iter 38: subdiag=[ 0.   +0.j -0.006-0.j  0.   +0.j],  ‖·‖₂=6.114e-03
  iter 39: subdiag=[-0.   -0.j  0.005+0.j -0.   -0.j],  ‖·‖₂=5.449e-03
  iter 40: subdiag=[ 0.   +0.j -0.005-0.j  0.   +0.j],  ‖·‖₂=4.857e-03
  iter 41: subdiag=[-0.   -0.j  0.004+0.j -0.   +0.j],  ‖·‖₂=4.329e-03
  iter 42: subdiag=[ 0.   +0.j -0.004+0.j  0.   -0.j],  ‖·‖₂=3.858e-03
  iter 43: subdiag=[-0.   -0.j  0.003-0.j -0.   +0.j],  ‖·‖₂=3.439e-03
  iter 44: subdiag=[ 0.   +0.j -0.003+0.j  0.   -0.j],  ‖·‖₂=3.065e-03
  iter 45: subdiag=[-0.   -0.j  0.003+0.j -0.   +0.j],  ‖·‖₂=2.731e-03
  iter 46: subdiag=[ 0.   +0.j -0.002-0.j  0.   -0.j],  ‖·‖₂=2.434e-03
  iter 47: subdiag=[-0.   -0.j  0.002+0.j -0.   +0.j],  ‖·‖₂=2.170e-03
  iter 48: subdiag=[ 0.   +0.j -0.002-0.j  0.   -0.j],  ‖·‖₂=1.934e-03
  iter 49: subdiag=[-0.   -0.j  0.002+0.j -0.   +0.j],  ‖·‖₂=1.724e-03
  iter 50: subdiag=[ 0.   +0.j -0.002-0.j  0.   -0.j],  ‖·‖₂=1.536e-03
  iter 51: subdiag=[-0.   -0.j  0.001+0.j -0.   +0.j],  ‖·‖₂=1.369e-03
  iter 52: subdiag=[ 0.   +0.j -0.001-0.j  0.   -0.j],  ‖·‖₂=1.220e-03
  iter 53: subdiag=[-0.   -0.j  0.001-0.j -0.   +0.j],  ‖·‖₂=1.088e-03
  iter 54: subdiag=[ 0.   +0.j -0.001-0.j  0.   -0.j],  ‖·‖₂=9.694e-04
  iter 55: subdiag=[-0.   -0.j  0.001+0.j -0.   +0.j],  ‖·‖₂=8.640e-04
  iter 56: subdiag=[ 0.   +0.j -0.001-0.j  0.   -0.j],  ‖·‖₂=7.701e-04
  iter 57: subdiag=[-0.   -0.j  0.001+0.j -0.   +0.j],  ‖·‖₂=6.863e-04
  iter 58: subdiag=[ 0.   +0.j -0.001-0.j  0.   -0.j],  ‖·‖₂=6.117e-04
  iter 59: subdiag=[-0.   -0.j  0.001+0.j -0.   +0.j],  ‖·‖₂=5.452e-04
  iter 60: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=4.859e-04
  iter 61: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=4.331e-04
  iter 62: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=3.860e-04
  iter 63: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=3.440e-04
  iter 64: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=3.066e-04
  iter 65: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.733e-04
  iter 66: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=2.436e-04
  iter 67: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.171e-04
  iter 68: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=1.935e-04
  iter 69: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.725e-04
  iter 70: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=1.537e-04
  iter 71: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.370e-04
  iter 72: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=1.221e-04
  iter 73: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=1.088e-04
  iter 74: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=9.699e-05
  iter 75: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=8.644e-05
  iter 76: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=7.704e-05
  iter 77: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=6.867e-05
  iter 78: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=6.120e-05
  iter 79: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=5.455e-05
  iter 80: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=4.862e-05
  iter 81: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=4.333e-05
  iter 82: subdiag=[ 0.+0.j -0.-0.j  0.-0.j],  ‖·‖₂=3.862e-05
  iter 83: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=3.442e-05
  iter 84: subdiag=[ 0.+0.j -0.-0.j  0.-0.j],  ‖·‖₂=3.068e-05
  iter 85: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=2.734e-05
  iter 86: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=2.437e-05
  iter 87: subdiag=[-0.-0.j  0.+0.j -0.-0.j],  ‖·‖₂=2.172e-05
  iter 88: subdiag=[ 0.+0.j -0.-0.j  0.+0.j],  ‖·‖₂=1.936e-05
  iter 89: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=1.725e-05
  iter 90: subdiag=[ 0.+0.j -0.-0.j  0.-0.j],  ‖·‖₂=1.538e-05
  iter 91: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=1.371e-05
  iter 92: subdiag=[ 0.+0.j -0.-0.j  0.-0.j],  ‖·‖₂=1.222e-05
  iter 93: subdiag=[-0.-0.j  0.+0.j -0.+0.j],  ‖·‖₂=1.089e-05
  iter 94: subdiag=[ 0.+0.j -0.-0.j  0.-0.j],  ‖·‖₂=9.704e-06
  iter 95: subdiag=[-0.-0.j  0.-0.j -0.+0.j],  ‖·‖₂=8.649e-06
  iter 96: subdiag=[ 0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=7.708e-06
  iter 97: subdiag=[-0.-0.j  0.-0.j -0.+0.j],  ‖·‖₂=6.870e-06
  iter 98: subdiag=[ 0.+0.j -0.+0.j  0.-0.j],  ‖·‖₂=6.123e-06
  iter 99: subdiag=[-0.-0.j  0.-0.j -0.+0.j],  ‖·‖₂=5.458e-06
  after : subdiag=[-0.-0.j  0.-0.j -0.+0.j],  ‖·‖₂=5.458e-06
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0. 0. 0.]
└─ converged?    = no

┌─ Matrix 26/30  (size 4x4)
│  fixed shift μ = 0.755595 (|μ|=0.7556)
  before: subdiag=[-0.994+0.j -0.988+0.j -0.76 +0.j],  ‖·‖₂=1.594e+00
  iter 00: subdiag=[-0.882+0.j -0.998+0.j -0.19 +0.j],  ‖·‖₂=1.345e+00
  iter 01: subdiag=[-0.701+0.j -0.997+0.j -0.039+0.j],  ‖·‖₂=1.219e+00
  iter 02: subdiag=[-0.503+0.j -0.988+0.j -0.008+0.j],  ‖·‖₂=1.109e+00
  iter 03: subdiag=[-0.338+0.j -0.982+0.j -0.002+0.j],  ‖·‖₂=1.039e+00
  iter 04: subdiag=[-0.221+0.j -0.98 +0.j -0.   +0.j],  ‖·‖₂=1.004e+00
  iter 05: subdiag=[-0.142+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.887e-01
  iter 06: subdiag=[-0.091+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.821e-01
  iter 07: subdiag=[-0.058+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.794e-01
  iter 08: subdiag=[-0.037+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.783e-01
  iter 09: subdiag=[-0.024+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.779e-01
  iter 10: subdiag=[-0.015+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.777e-01
  iter 11: subdiag=[-0.01 +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.776e-01
  iter 12: subdiag=[-0.006+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.776e-01
  iter 13: subdiag=[-0.004+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.776e-01
  iter 14: subdiag=[-0.002+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.776e-01
  iter 15: subdiag=[-0.002+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 16: subdiag=[-0.001+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 17: subdiag=[-0.001+0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 18: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 19: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 20: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 21: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 22: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 23: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 24: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 25: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 26: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 27: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 28: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 29: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 30: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 31: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 32: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 33: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 34: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 35: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 36: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 37: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 38: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 39: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 40: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 41: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 42: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 43: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 44: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 45: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 46: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 47: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 48: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 49: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 50: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 51: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 52: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 53: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 54: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 55: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 56: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 57: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 58: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 59: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 60: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 61: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 62: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 63: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 64: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 65: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 66: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 67: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 68: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 69: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 70: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 71: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 72: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 73: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 74: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 75: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 76: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 77: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 78: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 79: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 80: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 81: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 82: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 83: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 84: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 85: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 86: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 87: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 88: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 89: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 90: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 91: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 92: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 93: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 94: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 95: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 96: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 97: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 98: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  iter 99: subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
  after : subdiag=[-0.   +0.j -0.978+0.j -0.   +0.j],  ‖·‖₂=9.775e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.978 0.   ]
└─ converged?    = no

┌─ Matrix 27/30  (size 4x4)
│  fixed shift μ = -0.995116 (|μ|=0.9951)
  before: subdiag=[-0.999+0.j  0.99 +0.j -0.085+0.j],  ‖·‖₂=1.409e+00
  iter 00: subdiag=[-0.804+0.j  0.902+0.j -0.   +0.j],  ‖·‖₂=1.208e+00
  iter 01: subdiag=[-0.466+0.j  0.855+0.j -0.   +0.j],  ‖·‖₂=9.735e-01
  iter 02: subdiag=[-0.234+0.j  0.841+0.j -0.   +0.j],  ‖·‖₂=8.733e-01
  iter 03: subdiag=[-0.113+0.j  0.838+0.j -0.   +0.j],  ‖·‖₂=8.458e-01
  iter 04: subdiag=[-0.054+0.j  0.838+0.j -0.   +0.j],  ‖·‖₂=8.392e-01
  iter 05: subdiag=[-0.026+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.377e-01
  iter 06: subdiag=[-0.012+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.374e-01
  iter 07: subdiag=[-0.006+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 08: subdiag=[-0.003+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 09: subdiag=[-0.001+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 10: subdiag=[-0.001+0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 11: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 12: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 13: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 14: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 15: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 16: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 17: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 18: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 19: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 20: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 21: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 22: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 23: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 24: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 25: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 26: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 27: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 28: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 29: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 30: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 31: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 32: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 33: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 34: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 35: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 36: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 37: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 38: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 39: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 40: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 41: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 42: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 43: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 44: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 45: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 46: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 47: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 48: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 49: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 50: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 51: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 52: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 53: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 54: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 55: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 56: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 57: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 58: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 59: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 60: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 61: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 62: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 63: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 64: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 65: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 66: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 67: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 68: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 69: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 70: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 71: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 72: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 73: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 74: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 75: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 76: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 77: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 78: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 79: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 80: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 81: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 82: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 83: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 84: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 85: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 86: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 87: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 88: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 89: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 90: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 91: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 92: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 93: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 94: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 95: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 96: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 97: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 98: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  iter 99: subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
  after : subdiag=[-0.   +0.j  0.837+0.j -0.   +0.j],  ‖·‖₂=8.373e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.837 0.   ]
└─ converged?    = no

┌─ Matrix 28/30  (size 4x4)
│  fixed shift μ = -0.976473 (|μ|=0.9765)
  before: subdiag=[-0.799+0.j  0.936+0.j  0.149+0.j],  ‖·‖₂=1.239e+00
  iter 00: subdiag=[-0.683+0.j -0.093+0.j -0.118+0.j],  ‖·‖₂=6.991e-01
  iter 01: subdiag=[-0.682+0.j  0.006+0.j  0.118+0.j],  ‖·‖₂=6.923e-01
  iter 02: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 03: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 04: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 05: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 06: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 07: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 08: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 09: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 10: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 11: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 12: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 13: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 14: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 15: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 16: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 17: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 18: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 19: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 20: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 21: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 22: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 23: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 24: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 25: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 26: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 27: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 28: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 29: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 30: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 31: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 32: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 33: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 34: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 35: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 36: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 37: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 38: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 39: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 40: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 41: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 42: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 43: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 44: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 45: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 46: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 47: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 48: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 49: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 50: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 51: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 52: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 53: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 54: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 55: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 56: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 57: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 58: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 59: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 60: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 61: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 62: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 63: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 64: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 65: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 66: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 67: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 68: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 69: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 70: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 71: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 72: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 73: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 74: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 75: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 76: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 77: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 78: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 79: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 80: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 81: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 82: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 83: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 84: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 85: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 86: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 87: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 88: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 89: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 90: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 91: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 92: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 93: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 94: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 95: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 96: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 97: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  iter 98: subdiag=[-0.682+0.j -0.   +0.j -0.118+0.j],  ‖·‖₂=6.922e-01
  iter 99: subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
  after : subdiag=[-0.682+0.j  0.   +0.j  0.118+0.j],  ‖·‖₂=6.922e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.682 0.    0.118]
└─ converged?    = no

┌─ Matrix 29/30  (size 4x4)
│  fixed shift μ = 0.790166 (|μ|=0.7902)
  before: subdiag=[ 0.811+0.j -0.867+0.j  0.948+0.j],  ‖·‖₂=1.519e+00
  iter 00: subdiag=[ 0.726+0.j -0.795+0.j  0.199+0.j],  ‖·‖₂=1.095e+00
  iter 01: subdiag=[ 0.664+0.j -0.832+0.j  0.027+0.j],  ‖·‖₂=1.064e+00
  iter 02: subdiag=[ 0.596+0.j -0.861+0.j  0.004+0.j],  ‖·‖₂=1.047e+00
  iter 03: subdiag=[ 0.525+0.j -0.883+0.j  0.   +0.j],  ‖·‖₂=1.028e+00
  iter 04: subdiag=[ 0.456+0.j -0.899+0.j  0.   +0.j],  ‖·‖₂=1.008e+00
  iter 05: subdiag=[ 0.391+0.j -0.909+0.j  0.   +0.j],  ‖·‖₂=9.899e-01
  iter 06: subdiag=[ 0.332+0.j -0.917+0.j  0.   +0.j],  ‖·‖₂=9.751e-01
  iter 07: subdiag=[ 0.28 +0.j -0.922+0.j  0.   +0.j],  ‖·‖₂=9.635e-01
  iter 08: subdiag=[ 0.234+0.j -0.925+0.j  0.   +0.j],  ‖·‖₂=9.547e-01
  iter 09: subdiag=[ 0.196+0.j -0.928+0.j  0.   +0.j],  ‖·‖₂=9.483e-01
  iter 10: subdiag=[ 0.163+0.j -0.93 +0.j  0.   +0.j],  ‖·‖₂=9.437e-01
  iter 11: subdiag=[ 0.135+0.j -0.931+0.j  0.   +0.j],  ‖·‖₂=9.405e-01
  iter 12: subdiag=[ 0.112+0.j -0.931+0.j  0.   +0.j],  ‖·‖₂=9.382e-01
  iter 13: subdiag=[ 0.093+0.j -0.932+0.j  0.   +0.j],  ‖·‖₂=9.366e-01
  iter 14: subdiag=[ 0.077+0.j -0.932+0.j  0.   +0.j],  ‖·‖₂=9.355e-01
  iter 15: subdiag=[ 0.064+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.348e-01
  iter 16: subdiag=[ 0.053+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.342e-01
  iter 17: subdiag=[ 0.044+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.339e-01
  iter 18: subdiag=[ 0.036+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.336e-01
  iter 19: subdiag=[ 0.03 +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.335e-01
  iter 20: subdiag=[ 0.025+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.334e-01
  iter 21: subdiag=[ 0.02 +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.333e-01
  iter 22: subdiag=[ 0.017+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.332e-01
  iter 23: subdiag=[ 0.014+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.332e-01
  iter 24: subdiag=[ 0.012+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.332e-01
  iter 25: subdiag=[ 0.01 +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 26: subdiag=[ 0.008+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 27: subdiag=[ 0.007+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 28: subdiag=[ 0.005+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 29: subdiag=[ 0.004+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 30: subdiag=[ 0.004+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 31: subdiag=[ 0.003+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 32: subdiag=[ 0.003+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 33: subdiag=[ 0.002+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 34: subdiag=[ 0.002+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 35: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 36: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 37: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 38: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 39: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 40: subdiag=[ 0.001+0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 41: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 42: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 43: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 44: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 45: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 46: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 47: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 48: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 49: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 50: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 51: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 52: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 53: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 54: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 55: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 56: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 57: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 58: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 59: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 60: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 61: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 62: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 63: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 64: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 65: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 66: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 67: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 68: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 69: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 70: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 71: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 72: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 73: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 74: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 75: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 76: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 77: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 78: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 79: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 80: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 81: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 82: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 83: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 84: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 85: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 86: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 87: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 88: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 89: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 90: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 91: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 92: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 93: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 94: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 95: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 96: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 97: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 98: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  iter 99: subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
  after : subdiag=[ 0.   +0.j -0.933+0.j  0.   +0.j],  ‖·‖₂=9.331e-01
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.933 0.   ]
└─ converged?    = no

┌─ Matrix 30/30  (size 4x4)
│  fixed shift μ = -0.0383725-0.203948j (|μ|=0.2075)
  before: subdiag=[-0.744+0.j -0.999+0.j -0.997+0.j],  ‖·‖₂=1.596e+00
  iter 00: subdiag=[-0.752-0.j -0.99 +0.j -0.977+0.j],  ‖·‖₂=1.581e+00
  iter 01: subdiag=[0.757+0.j 0.957+0.j 0.928-0.j],  ‖·‖₂=1.533e+00
  iter 02: subdiag=[-0.758-0.j -0.894+0.j -0.87 +0.j],  ‖·‖₂=1.460e+00
  iter 03: subdiag=[0.76 +0.j 0.803-0.j 0.814-0.j],  ‖·‖₂=1.372e+00
  iter 04: subdiag=[-0.761-0.j -0.694-0.j -0.766+0.j],  ‖·‖₂=1.284e+00
  iter 05: subdiag=[0.761+0.j 0.582-0.j 0.729-0.j],  ‖·‖₂=1.204e+00
  iter 06: subdiag=[-0.758-0.j -0.477+0.j -0.699+0.j],  ‖·‖₂=1.136e+00
  iter 07: subdiag=[-0.751-0.j -0.385-0.j  0.674-0.j],  ‖·‖₂=1.080e+00
  iter 08: subdiag=[-0.738-0.j -0.309+0.j -0.653+0.j],  ‖·‖₂=1.033e+00
  iter 09: subdiag=[-0.719-0.j -0.247+0.j  0.635-0.j],  ‖·‖₂=9.907e-01
  iter 10: subdiag=[ 0.694+0.j -0.197-0.j -0.618+0.j],  ‖·‖₂=9.504e-01
  iter 11: subdiag=[-0.665-0.j -0.158-0.j  0.602-0.j],  ‖·‖₂=9.105e-01
  iter 12: subdiag=[ 0.631+0.j -0.126-0.j -0.586+0.j],  ‖·‖₂=8.704e-01
  iter 13: subdiag=[-0.594-0.j -0.102-0.j  0.571-0.j],  ‖·‖₂=8.301e-01
  iter 14: subdiag=[ 0.555+0.j -0.082-0.j -0.556+0.j],  ‖·‖₂=7.899e-01
  iter 15: subdiag=[-0.516+0.j -0.066-0.j  0.541+0.j],  ‖·‖₂=7.503e-01
  iter 16: subdiag=[ 0.476+0.j -0.054-0.j -0.526+0.j],  ‖·‖₂=7.118e-01
  iter 17: subdiag=[-0.438+0.j -0.044-0.j  0.511+0.j],  ‖·‖₂=6.747e-01
  iter 18: subdiag=[ 0.401-0.j -0.036-0.j -0.496-0.j],  ‖·‖₂=6.393e-01
  iter 19: subdiag=[-0.366-0.j -0.029+0.j  0.482+0.j],  ‖·‖₂=6.059e-01
  iter 20: subdiag=[ 0.333+0.j -0.024+0.j -0.467-0.j],  ‖·‖₂=5.745e-01
  iter 21: subdiag=[-0.303+0.j -0.019+0.j  0.453+0.j],  ‖·‖₂=5.451e-01
  iter 22: subdiag=[ 0.275+0.j -0.016+0.j -0.439+0.j],  ‖·‖₂=5.176e-01
  iter 23: subdiag=[-0.248+0.j -0.013+0.j  0.425-0.j],  ‖·‖₂=4.921e-01
  iter 24: subdiag=[ 0.224-0.j -0.011+0.j -0.411+0.j],  ‖·‖₂=4.682e-01
  iter 25: subdiag=[-0.203+0.j -0.009-0.j  0.397-0.j],  ‖·‖₂=4.460e-01
  iter 26: subdiag=[ 0.183+0.j -0.007-0.j -0.384+0.j],  ‖·‖₂=4.253e-01
  iter 27: subdiag=[-0.165-0.j -0.006-0.j  0.371-0.j],  ‖·‖₂=4.060e-01
  iter 28: subdiag=[ 0.148+0.j -0.005-0.j -0.358+0.j],  ‖·‖₂=3.878e-01
  iter 29: subdiag=[-0.134+0.j -0.004-0.j  0.346-0.j],  ‖·‖₂=3.708e-01
  iter 30: subdiag=[ 0.12 -0.j -0.003-0.j -0.334+0.j],  ‖·‖₂=3.548e-01
  iter 31: subdiag=[-0.108-0.j -0.003-0.j  0.322-0.j],  ‖·‖₂=3.397e-01
  iter 32: subdiag=[ 0.097+0.j -0.002-0.j -0.311+0.j],  ‖·‖₂=3.255e-01
  iter 33: subdiag=[-0.088+0.j -0.002-0.j  0.299-0.j],  ‖·‖₂=3.120e-01
  iter 34: subdiag=[ 0.079+0.j -0.002-0.j -0.289+0.j],  ‖·‖₂=2.992e-01
  iter 35: subdiag=[-0.071-0.j -0.001-0.j  0.278-0.j],  ‖·‖₂=2.870e-01
  iter 36: subdiag=[ 0.064-0.j -0.001+0.j -0.268+0.j],  ‖·‖₂=2.754e-01
  iter 37: subdiag=[-0.057+0.j -0.001+0.j  0.258-0.j],  ‖·‖₂=2.644e-01
  iter 38: subdiag=[ 0.051-0.j -0.001+0.j -0.249+0.j],  ‖·‖₂=2.538e-01
  iter 39: subdiag=[-0.046+0.j -0.001+0.j  0.239-0.j],  ‖·‖₂=2.437e-01
  iter 40: subdiag=[ 0.042-0.j -0.   +0.j -0.23 +0.j],  ‖·‖₂=2.341e-01
  iter 41: subdiag=[-0.037+0.j -0.   -0.j  0.222-0.j],  ‖·‖₂=2.249e-01
  iter 42: subdiag=[ 0.034+0.j -0.   -0.j -0.213+0.j],  ‖·‖₂=2.160e-01
  iter 43: subdiag=[-0.03 +0.j -0.   -0.j  0.205-0.j],  ‖·‖₂=2.075e-01
  iter 44: subdiag=[ 0.027-0.j -0.   -0.j -0.198+0.j],  ‖·‖₂=1.994e-01
  iter 45: subdiag=[-0.024+0.j -0.   -0.j  0.19 -0.j],  ‖·‖₂=1.916e-01
  iter 46: subdiag=[ 0.022-0.j -0.   -0.j -0.183+0.j],  ‖·‖₂=1.841e-01
  iter 47: subdiag=[-0.02 +0.j -0.   -0.j  0.176-0.j],  ‖·‖₂=1.769e-01
  iter 48: subdiag=[ 0.018-0.j -0.   -0.j -0.169+0.j],  ‖·‖₂=1.700e-01
  iter 49: subdiag=[-0.016+0.j -0.   -0.j  0.163-0.j],  ‖·‖₂=1.634e-01
  iter 50: subdiag=[ 0.014-0.j -0.   -0.j -0.156+0.j],  ‖·‖₂=1.570e-01
  iter 51: subdiag=[-0.013+0.j -0.   -0.j  0.15 -0.j],  ‖·‖₂=1.509e-01
  iter 52: subdiag=[ 0.012-0.j -0.   -0.j -0.145+0.j],  ‖·‖₂=1.450e-01
  iter 53: subdiag=[-0.01 +0.j -0.   -0.j  0.139+0.j],  ‖·‖₂=1.393e-01
  iter 54: subdiag=[ 0.009-0.j -0.   +0.j -0.134-0.j],  ‖·‖₂=1.339e-01
  iter 55: subdiag=[-0.008-0.j -0.   +0.j  0.128+0.j],  ‖·‖₂=1.286e-01
  iter 56: subdiag=[ 0.008-0.j -0.   +0.j -0.123-0.j],  ‖·‖₂=1.236e-01
  iter 57: subdiag=[-0.007+0.j -0.   +0.j  0.119+0.j],  ‖·‖₂=1.188e-01
  iter 58: subdiag=[ 0.006-0.j -0.   +0.j -0.114-0.j],  ‖·‖₂=1.141e-01
  iter 59: subdiag=[-0.005+0.j -0.   +0.j  0.11 +0.j],  ‖·‖₂=1.097e-01
  iter 60: subdiag=[ 0.005-0.j -0.   +0.j -0.105-0.j],  ‖·‖₂=1.054e-01
  iter 61: subdiag=[-0.004+0.j -0.   +0.j  0.101+0.j],  ‖·‖₂=1.013e-01
  iter 62: subdiag=[ 0.004-0.j -0.   -0.j -0.097-0.j],  ‖·‖₂=9.730e-02
  iter 63: subdiag=[-0.004+0.j -0.   -0.j  0.093+0.j],  ‖·‖₂=9.349e-02
  iter 64: subdiag=[ 0.003-0.j -0.   -0.j -0.09 -0.j],  ‖·‖₂=8.983e-02
  iter 65: subdiag=[-0.003+0.j -0.   -0.j  0.086+0.j],  ‖·‖₂=8.631e-02
  iter 66: subdiag=[ 0.003-0.j -0.   -0.j -0.083-0.j],  ‖·‖₂=8.292e-02
  iter 67: subdiag=[-0.002+0.j -0.   -0.j  0.08 +0.j],  ‖·‖₂=7.967e-02
  iter 68: subdiag=[ 0.002-0.j -0.   -0.j -0.077-0.j],  ‖·‖₂=7.655e-02
  iter 69: subdiag=[-0.002+0.j -0.   -0.j  0.074+0.j],  ‖·‖₂=7.355e-02
  iter 70: subdiag=[ 0.002-0.j -0.   -0.j -0.071-0.j],  ‖·‖₂=7.066e-02
  iter 71: subdiag=[-0.002+0.j -0.   -0.j  0.068+0.j],  ‖·‖₂=6.789e-02
  iter 72: subdiag=[ 0.001-0.j -0.   -0.j -0.065-0.j],  ‖·‖₂=6.522e-02
  iter 73: subdiag=[-0.001+0.j -0.   -0.j  0.063+0.j],  ‖·‖₂=6.266e-02
  iter 74: subdiag=[ 0.001-0.j -0.   -0.j -0.06 -0.j],  ‖·‖₂=6.020e-02
  iter 75: subdiag=[-0.001+0.j -0.   -0.j  0.058+0.j],  ‖·‖₂=5.783e-02
  iter 76: subdiag=[ 0.001-0.j -0.   -0.j -0.056-0.j],  ‖·‖₂=5.556e-02
  iter 77: subdiag=[-0.001+0.j -0.   -0.j  0.053+0.j],  ‖·‖₂=5.338e-02
  iter 78: subdiag=[ 0.001-0.j -0.   -0.j -0.051-0.j],  ‖·‖₂=5.128e-02
  iter 79: subdiag=[-0.001+0.j -0.   -0.j  0.049+0.j],  ‖·‖₂=4.926e-02
  iter 80: subdiag=[ 0.001-0.j -0.   -0.j -0.047-0.j],  ‖·‖₂=4.733e-02
  iter 81: subdiag=[-0.001+0.j -0.   -0.j  0.045+0.j],  ‖·‖₂=4.547e-02
  iter 82: subdiag=[ 0.   -0.j -0.   -0.j -0.044-0.j],  ‖·‖₂=4.368e-02
  iter 83: subdiag=[-0.   +0.j -0.   -0.j  0.042+0.j],  ‖·‖₂=4.196e-02
  iter 84: subdiag=[ 0.  -0.j -0.  -0.j -0.04-0.j],  ‖·‖₂=4.031e-02
  iter 85: subdiag=[-0.   +0.j -0.   -0.j  0.039+0.j],  ‖·‖₂=3.872e-02
  iter 86: subdiag=[ 0.   -0.j -0.   -0.j -0.037-0.j],  ‖·‖₂=3.720e-02
  iter 87: subdiag=[-0.   +0.j -0.   -0.j  0.036+0.j],  ‖·‖₂=3.574e-02
  iter 88: subdiag=[ 0.   -0.j -0.   -0.j -0.034-0.j],  ‖·‖₂=3.433e-02
  iter 89: subdiag=[-0.   +0.j -0.   -0.j  0.033+0.j],  ‖·‖₂=3.298e-02
  iter 90: subdiag=[ 0.   -0.j -0.   -0.j -0.032-0.j],  ‖·‖₂=3.168e-02
  iter 91: subdiag=[-0.  +0.j -0.  -0.j  0.03+0.j],  ‖·‖₂=3.043e-02
  iter 92: subdiag=[ 0.   -0.j -0.   -0.j -0.029-0.j],  ‖·‖₂=2.924e-02
  iter 93: subdiag=[-0.   +0.j -0.   -0.j  0.028+0.j],  ‖·‖₂=2.809e-02
  iter 94: subdiag=[ 0.   -0.j -0.   -0.j -0.027-0.j],  ‖·‖₂=2.698e-02
  iter 95: subdiag=[-0.   +0.j -0.   -0.j  0.026+0.j],  ‖·‖₂=2.592e-02
  iter 96: subdiag=[ 0.   -0.j -0.   -0.j -0.025-0.j],  ‖·‖₂=2.490e-02
  iter 97: subdiag=[-0.   +0.j -0.   -0.j  0.024+0.j],  ‖·‖₂=2.392e-02
  iter 98: subdiag=[ 0.   -0.j -0.   -0.j -0.023-0.j],  ‖·‖₂=2.298e-02
  iter 99: subdiag=[-0.   +0.j -0.   -0.j  0.022+0.j],  ‖·‖₂=2.207e-02
  after : subdiag=[-0.   +0.j -0.   -0.j  0.022+0.j],  ‖·‖₂=2.207e-02
│  iterations    = 100/100
│  sub‑diag magnitudes after last step:
│  [0.    0.    0.022]
└─ converged?    = no
```

From this monstrosity we conclude that using one constant shift $mu$ (taken from the trailing $2 times 2$ block) attacks only the last sub-diagonal element. That entry shrinks quickly, but the higher sub-diagonals are influenced only through round-off-sized couplings, so they plateau at some non-zero value $approx inv(10, 2)$ Consequently the matrix rarely satisfies the stringent "$abs(H_(i, i-1)) < inv(10, 10)$" test, and the script reports converged = no even after 100 iterations.

Using the Wilkinson shift would definitely force convergence, but since we are using one $mu = lambda_1$ of the trailing $2 times 2$ block, we can expect convergence only for the last sub-diagonal element.

#bibliography("bibliography.bib")
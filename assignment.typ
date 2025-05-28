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
  image("images/introduction_example_plot_1.png", width: 50%), caption: [
    $p(x)$ via the coefficients in @equation_example_polynomial_introduction
  ]
) <figure_example_plot_1_introduction>

#figure(
  image("images/introduction_example_plot_2.png", width: 50%), caption: [
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

The following function calculates the Householder reflectors that reduce a matrix to Hessenberg form. It returns the reflector vectors, the compact Hessenberg matrix $H$, and the accumulated orthogonal factor $Q$.

```python
import numpy as np
import time
from typing import List, Tuple


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
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display, Markdown
from ast import literal_eval

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

    plt.figure(figsize=(7,5))
    mark = {"general":"o", "symmetric":"s"}
    for label, sub in df.groupby("type"):
        plt.loglog(sub["size"], sub["avg"], marker=mark[label], ls="-", label=label)
        if len(sub) > 1:
            a,b = np.polyfit(np.log10(sub["size"]), np.log10(sub["avg"]), 1)
            plt.loglog(sub["size"], 10**(b+a*np.log10(sub["size"])),
                       "--", label=f"{label} fit ~ $n^{a:.2f}$")
    plt.xlabel("matrix size  (log)")
    plt.ylabel("runtime [s]  (log)")
    plt.title("Hessenberg (general)  vs  Tridiagonal (symmetric)")
    plt.grid(True, which="both", ls=":")
    plt.legend(); plt.tight_layout(); plt.show()
    return df


#===INTERACTIVE PART=========================================================
try:
    raw = input("\nMatrix sizes (Python list) (e.g): [64,128,256,512,1024]: ")
    sizes = literal_eval(raw) if raw.strip() else [64,128,256,512,1024]
except Exception:
    print("Bad list -> using default.")
    sizes = [64,128,256,512,1024]

dist = input("Distribution ('normal'/'uniform')  [normal]: ").strip().lower() or "normal"
mode_txt = input("Matrix type g=general, s=symmetric, b=both  [g]: ").strip().lower() or "g"
mode = "symmetric" if mode_txt=="s" else "both" if mode_txt=="b" else "general"
seed_txt = input("Random seed (None/int) [None]: ").strip()
seed_val = None if seed_txt.lower() in {"", "none"} else int(seed_txt)

for sym in ([False, True] if mode=="both" else [mode=="symmetric"]): #accuracy on largest size
    verify_factorisation_once(max(sizes), dist, sym, seed_val)

benchmark_hessenberg(sizes, dist, mode, seed_val) #timings

```

The reader should be aware that my poor #link("https://www.dell.com/support/manuals/pt-br/inspiron-15-5590-laptop/inspiron-5590-setup-and-specifications/specifications-of-inspiron-5590?guid=guid-7c9f07ce-626e-44ca-be3a-a1fb036413f9&lang=en-us")[Dell Inspiro 5590] has crashed precisely $5$ times while i was writing this. The runtime was around $4$ minutes for a matrix $A approx 10^3 times 10^3$.

An expected output is:

```python
64×64 general
‖A − Q T Qᵀ‖ = 7.51e-14
‖QᵀQ − I‖ = 7.07e-15

64×64 symmetric
‖A − Q T Qᵀ‖ = 4.83e-14
‖QᵀQ − I‖ = 7.39e-15

128×128 general
‖A − Q T Qᵀ‖ = 1.84e-13
‖QᵀQ − I‖ = 1.26e-14

128×128 symmetric
‖A − Q T Qᵀ‖ = 1.14e-13
‖QᵀQ − I‖ = 1.25e-14

256×256 general
‖A − Q T Qᵀ‖ = 4.70e-13
‖QᵀQ − I‖ = 2.28e-14

256×256 symmetric
‖A − Q T Qᵀ‖ = 2.78e-13
‖QᵀQ − I‖ = 2.25e-14

512×512 general
‖A − Q T Qᵀ‖ = 1.16e-12
‖QᵀQ − I‖ = 4.10e-14

512×512 symmetric
‖A − Q T Qᵀ‖ = 7.10e-13
‖QᵀQ − I‖ = 4.09e-14

1024×1024 general
‖A − Q T Qᵀ‖ = 3.05e-12
‖QᵀQ − I‖ = 7.57e-14

1024×1024 symmetric
‖A − Q T Qᵀ‖ = 1.84e-12
‖QᵀQ − I‖ = 7.64e-14
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

@figure_plot_evaluation_hessenberg_function shows the expected $O(n^3)$ complexity for the general case and $O(n^2)$ for the symmetric case. The latter is better discussed in @section_symmetric_case.

To understand why the complexity is $O(n^3)$ in the general case, we can look at the algorithm. The outer loop runs $n - 2$ times, and inside it, we have two matrix-vector products and two outer products, which are all $O(n^2)$. Thus, the total complexity is $O(n^3)$.

=== The Symmetric Case (d)
<section_symmetric_case>

On the symmetric case we know that reflectors will be applied in only one side of the matrix, since $transpose(v) A = transpose(A v)$. That is precisely what the function `generate_random_matrix` does. Which cuts complexity from the expected $O(n^3)$ seen in the previous section to a $O(n^2)$ #footnote[See page 194 of #link("https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf")[Trefethen & Bau's Numerical Linear Algebra book]].







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

Since $phi_j / abs(phi_j) = plus.minus 1$, the sequence converges to $plus.minus v_j$ uga buga

== Inverse Iteration
<section_inverse_iteration>

Consider $mu in RR without Lambda$, where $Lambda$ is the set of eigenvalues of $A$. The eigenvalues $hat(lambda)$ of $inv((A - mu I), 1)$ are:

$
  det(A - mu I - hat(lambda) I) = 0 <=> det(A - (mu + hat(lambda)) I) = 0\

    <=> hat(lambda)_j = 1 / (lambda_j - mu)
$

Where $lambda_j$ are the eigenvalues of $A$. So if $mu$ is close to an eigenvalue, then $hat(lambda)$ will be large. Power iteration seems interesting here, so the sequence:

$
  x / norm(x), (inv((A - mu I), 1) x) / norm(inv((A - mu I), 1) x), (inv((A - mu I), 2) x) / norm(inv((A - mu I), 2) x), dots
$

Converges to the eigenvector associated to the eigenvalue $hat(lambda)$.

== QR Iteration
<section_qr_iteration>

== QR Iteration with Shifts
<section_qr_iteration_with_shifts>

= Orthogonal Matrices (Problem 2) (a)
<section_orthogonal_matrices>

Here we will discuss how orthogonal matrices behave when we appluy the iterations discussed in @section_power_iteration, @section_inverse_iteration and @section_qr_iteration_with_shifts.

So let $Q in CC^(m times n)$ be an orthogonal matrix. We are interested in its eigenvalues $lambda$. We know that:

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

Where $lambda_i$ are the eigenvalues of $Q$, $phi_i$ are the coefficients of the expansion of $x$ in the basis of eigenvectors $v_i$. Since we have that $abs(lambda_i) = 1$, we have:

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

If we apply inverse iteration to an orthogonal matrix with a shift $mu$, we have:

$
  det(Q - mu I - hat(lambda) I) = 0 <=> det(Q - (mu + hat(lambda)) I) = 0\

  <=> hat(lambda)_j = 1 / (lambda_j - mu)
$

We know that the eigenvalues of $Q$ are on the unit circle, so if $mu$ is close to an eigenvalue $lambda_j$, $hat(lambda_j)$ will be huge (dominant), which makes power iteration converge to the eigenvector associated to $hat(lambda_j)_j$, which is the eigenvector associated to $lambda_j$. The fact that the eigenvalues are on the unit circle also contributes to the convergence of the method.

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





== Orthogonal Matrices and QR Iteration With A Specific Shift
<section_orthogonal_matrices_and_qr_iteration_with_shift>



== Shift With an Eigenvalue (d)
<section_shift_with_an_eigenvalue>

= Conclusion
<section_conclusion>

#bibliography("bibliography.bib")
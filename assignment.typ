#import "@preview/ctheorems:1.1.3": *
#import "@preview/plotst:0.2.0": *
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#codly(languages: codly-languages)

#show: codly-init.with()
#show: thmrules.with(qed-symbol: $square$)
#show link: underline

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

#let inv(arg) = $arg^(-1)$
#let herm(arg) = $arg^*$
#let transpose(arg) = $arg^T$


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

== Evaluating the Function (b)
<section_evaluating_the_function>

== Complexity (c)
<section_complexity>

== The Symmetric Case (d)
<section_symmetric_case>

= Orthogonal Matrices (Problem 2)
<section_orthogonal_matrices>

== Eigenvalues and Iterative Methods (a)
<section_eigenvalues_and_iterative_methods>

== The *$2 times 2$* Case (b)
<section_2x2_case>

== Random Orthogonal Matrices (c)
<section_random_orthogonal_matrices>

== Shift With an Eigenvalue (d)
<section_shift_with_an_eigenvalue>

= Conclusion
<section_conclusion>

#bibliography("bibliography.bib")
r"""Commutator constraints for the quantized Poisson equation.

See ``docs/constraints.md`` for construction modes and derivations.
"""

from collections.abc import Mapping

import numpy as np
import scipy.linalg
import scipy.linalg.interpolative
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import quflow as qf


__all__ = [
    "CommutatorPoissonSolver",
    "ConstraintPlotter",
    "constraint_matrix",
    "largest_eigenvector",
    "plotter",
    "project_commutator",
    "project_domain",
    "spectral_matrix",
]


_ROW_NORM_TOL = 1e-12


# Matrix-free commutator constraints


def _commutator_operator(F):
    """Return the matrix-free operator ``X -> bracket(F, X)``."""
    N = F.shape[0]
    N2 = N**2

    def matvec(X):
        return qf.geometry.bracket(F, np.asarray(X).reshape(N, N)).ravel()

    def rmatvec(X):
        return -matvec(X)

    return spla.LinearOperator(
        (N2, N2), matvec=matvec, rmatvec=rmatvec, dtype=F.dtype
    )


def _commutator_coordinate_row_norms(F):
    """Return all commutator coordinate-row norms without forming the matrix."""
    row_sq = np.sum(np.abs(F) ** 2, axis=1)
    col_sq = np.sum(np.abs(F) ** 2, axis=0)
    diag = np.diag(F)
    diag_sq = np.abs(diag) ** 2
    overlap = (
        -diag_sq[:, np.newaxis]
        - diag_sq[np.newaxis, :]
        + np.abs(diag[:, np.newaxis] - diag[np.newaxis, :]) ** 2
    )
    norms_sq = row_sq[:, np.newaxis] + col_sq[np.newaxis, :] + overlap
    return np.sqrt(np.maximum(norms_sq.real, 0.0)).ravel() / qf.geometry.hbar(
        F.shape[0]
    )


def _materialize_commutator_rows(F, selected_row_indices):
    """Return selected row-major commutator rows as a CSR matrix."""
    N = F.shape[0]
    N2 = N**2
    selected_row_indices = np.asarray(selected_row_indices, dtype=int)
    if selected_row_indices.size == 0:
        return sp.csr_matrix((0, N2), dtype=F.dtype)

    rows = []
    cols = []
    values = []

    # This is the sparse coefficient form of qf.geometry.bracket.  Calling
    # bracket on one dense basis matrix per selected row would be much costlier.
    for row_position, flat_index in enumerate(selected_row_indices):
        output_row = flat_index // N
        output_column = flat_index % N

        matrix_row = F[output_row, :]
        nonzero = np.flatnonzero(matrix_row)
        rows.extend([row_position] * nonzero.size)
        cols.extend((nonzero * N + output_column).tolist())
        values.extend(matrix_row[nonzero].tolist())

        matrix_column = F[:, output_column]
        nonzero = np.flatnonzero(matrix_column)
        rows.extend([row_position] * nonzero.size)
        cols.extend((output_row * N + nonzero).tolist())
        values.extend((-matrix_column[nonzero]).tolist())

    values = np.asarray(values, dtype=F.dtype) / qf.geometry.hbar(N)
    selected_rows = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(selected_row_indices.size, N2),
    ).tocsr()
    selected_rows.sum_duplicates()
    selected_rows.eliminate_zeros()
    return selected_rows


def _orthonormal_eigenvector_columns(eigenvectors):
    """Return orthonormal vectors as columns of an ``N x K`` matrix.

    A two-dimensional NumPy array follows the eigensolver convention and is
    interpreted column-wise.  A list or other iterable may instead contain
    one one-dimensional vector per item.
    """
    if eigenvectors is None:
        raise ValueError("eigenvectors must be a nonempty iterable of vectors.")

    input_is_array = isinstance(eigenvectors, np.ndarray)
    if input_is_array:
        array = np.asarray(eigenvectors)
        if array.ndim == 1:
            eigenvector_matrix = array[:, np.newaxis]
        elif array.ndim == 2:
            eigenvector_matrix = array
        else:
            raise ValueError(
                "eigenvectors must be a one-dimensional vector, an N-by-K "
                "array, or an iterable of one-dimensional vectors."
            )
    else:
        try:
            vectors = tuple(eigenvectors)
        except TypeError as error:
            raise ValueError(
                "eigenvectors must be a nonempty iterable of vectors."
            ) from error
        if not vectors:
            raise ValueError(
                "eigenvectors must be a nonempty iterable of vectors."
            )
        vectors = tuple(np.asarray(vector) for vector in vectors)
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("Each eigenvector must be one-dimensional.")
        if any(vector.size != vectors[0].size for vector in vectors):
            raise ValueError("Eigenvectors must all have the same length.")
        eigenvector_matrix = np.column_stack(vectors)

    N, K = eigenvector_matrix.shape
    if N == 0 or K == 0:
        raise ValueError(
            "eigenvectors must be a nonempty iterable of vectors."
        )
    if K > N:
        raise ValueError(
            f"At most {N} orthonormal eigenvectors can be supplied, got {K}."
        )

    dtype = np.result_type(eigenvector_matrix.dtype, np.complex128)
    eigenvector_matrix = eigenvector_matrix.astype(dtype, copy=False)
    if not np.isfinite(eigenvector_matrix).all():
        raise ValueError("eigenvectors must contain only finite values.")

    gram = eigenvector_matrix.conj().T @ eigenvector_matrix
    identity = np.eye(K, dtype=dtype)
    if not np.allclose(gram, identity, rtol=1e-10, atol=1e-12):
        defect = float(np.max(np.abs(gram - identity)))
        raise ValueError(
            "eigenvectors must be orthonormal; the largest Gram-matrix "
            f"error is {defect:.3e}."
        )
    return eigenvector_matrix


class _EigenvectorConstraintOperator(spla.LinearOperator):
    r"""Matrix-free constraints induced by distinct active eigenvectors.

    If ``E`` contains the active eigenvectors and ``Q`` spans their
    orthogonal complement, set ``A = [E, Q]``.  The operator extracts the
    entries of ``A^* X A`` which touch an active coordinate, except for the
    active diagonal.  Its rows are orthonormal and its null space consists of
    the matrices which preserve every active eigendirection and their common
    complementary eigenspace.
    """

    def __init__(self, active_eigenvectors):
        active_eigenvectors = np.asarray(active_eigenvectors)
        N, K = active_eigenvectors.shape
        complement = scipy.linalg.null_space(active_eigenvectors.conj().T)
        if complement.shape != (N, N - K):
            raise RuntimeError(
                "Could not construct the orthogonal eigenvector complement."
            )

        self.active_eigenvectors = active_eigenvectors
        self.complement_eigenvectors = complement
        self.eigenbasis = np.column_stack(
            (active_eigenvectors, complement)
        )
        self.matrix_size = N
        self.num_active_eigenvalues = K

        upper_rows, upper_columns = np.triu_indices(N, k=1)
        keep = upper_rows < K
        self._upper_rows = upper_rows[keep]
        self._upper_columns = upper_columns[keep]
        self.half_rank = self._upper_rows.size
        rank = 2 * self.half_rank
        expected_rank = K * (2 * N - K - 1)
        if rank != expected_rank:
            raise RuntimeError(
                f"Constructed constraint rank {rank}, expected {expected_rank}."
            )

        dtype = np.dtype(
            np.result_type(active_eigenvectors.dtype, np.complex128)
        )
        super().__init__(dtype=dtype, shape=(rank, N**2))

    def _as_matrix(self, vector):
        vector = np.asarray(vector, dtype=self.dtype)
        if vector.size != self.matrix_size**2:
            raise ValueError(
                f"Expected {self.matrix_size**2} matrix entries, "
                f"got {vector.size}."
            )
        return vector.reshape(self.matrix_size, self.matrix_size)

    def _as_multipliers(self, multipliers):
        multipliers = np.asarray(multipliers, dtype=self.dtype).ravel()
        if multipliers.size != self.shape[0]:
            raise ValueError(
                f"Expected {self.shape[0]} constraint multipliers, "
                f"got {multipliers.size}."
            )
        return multipliers

    def _matvec(self, vector):
        """Apply the full complex-linear constraint operator."""
        matrix = self._as_matrix(vector)
        E = self.active_eigenvectors
        A = self.eigenbasis
        rows = self._upper_rows
        columns = self._upper_columns

        # Parenthesization keeps both products O(K N^2), rather than O(N^3)
        # when K is small.
        active_rows = (E.conj().T @ matrix) @ A
        active_columns = A.conj().T @ (matrix @ E)
        upper = active_rows[rows, columns]
        lower = active_columns[columns, rows]
        return np.concatenate((upper, lower))

    def _rmatvec(self, multipliers):
        """Apply the adjoint of the full complex-linear operator."""
        multipliers = self._as_multipliers(multipliers)
        N = self.matrix_size
        K = self.num_active_eigenvalues
        half_rank = self.half_rank
        rows = self._upper_rows
        columns = self._upper_columns
        upper = multipliers[:half_rank]
        lower = multipliers[half_rank:]

        # In eigenbasis coordinates the multiplier matrix is supported in
        # the first K rows and columns.  Keep that low-rank structure through
        # the change of basis instead of forming A @ Y @ A.H densely.
        top = np.zeros((K, N), dtype=self.dtype)
        top[rows, columns] = upper

        active_lower = columns < K
        top[columns[active_lower], rows[active_lower]] = lower[active_lower]

        E = self.active_eigenvectors
        A = self.eigenbasis
        result = E @ (top @ A.conj().T)
        if K < N:
            bottom_left = np.zeros((N - K, K), dtype=self.dtype)
            bottom_left[
                columns[~active_lower] - K, rows[~active_lower]
            ] = lower[~active_lower]
            Q = self.complement_eigenvectors
            result += (Q @ bottom_left) @ E.conj().T
        return result.ravel()

    def matvec_skew_hermitian(self, matrix):
        """Apply the constraint using one triangle of a skew-Hermitian input."""
        matrix = self._as_matrix(matrix)
        E = self.active_eigenvectors
        A = self.eigenbasis
        rows = self._upper_rows
        columns = self._upper_columns

        active_columns = A.conj().T @ (matrix @ E)
        lower = active_columns[columns, rows]
        upper = -lower.conj()
        return np.concatenate((upper, lower))

    def rmatvec_skew_hermitian(self, multipliers):
        """Apply the adjoint on the skew-Hermitian multiplier subspace."""
        multipliers = self._as_multipliers(multipliers)
        half_rank = self.half_rank
        upper = multipliers[:half_rank]
        lower = multipliers[half_rank:]

        # Project away roundoff that violates upper = -conj(lower).  If R is
        # the strictly-lower part in eigenbasis coordinates, the full matrix
        # is R - R.H, so only one change-of-basis product is required.
        lower = 0.5 * (lower - upper.conj())
        lower_coordinates = np.zeros(
            (self.matrix_size, self.num_active_eigenvalues), dtype=self.dtype
        )
        lower_coordinates[self._upper_columns, self._upper_rows] = lower
        transformed_lower = (
            self.eigenbasis @ lower_coordinates
        ) @ self.active_eigenvectors.conj().T
        result = transformed_lower - transformed_lower.conj().T
        return result.ravel()


def project_commutator(W, eigenvectors):
    r"""Project ``W`` onto the commutant defined by active eigenvectors.

    The columns of an ``N x K`` array ``eigenvectors`` are interpreted as
    orthonormal eigenvectors with distinct nonzero eigenvalues; a list of
    one-dimensional vectors is also accepted.  Their orthogonal complement
    is the remaining, repeated zero eigenspace.  If ``E`` contains the active
    vectors and ``R = I - E E^*``, the Hilbert--Schmidt projection is

    .. math::

        W \longmapsto R W R
        + \sum_{j=1}^K e_j e_j^* W e_j e_j^*.

    Only the eigenspaces matter, so the distinct labels ``1, ..., K`` do not
    need to be formed explicitly.  The calculation uses thin ``N x K``
    products and does not construct an orthogonal-complement basis.
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] == 0:
        raise ValueError(f"W must be a nonempty square matrix, got {W.shape}.")

    E = _orthonormal_eigenvector_columns(eigenvectors)
    N, K = E.shape
    if W.shape != (N, N):
        raise ValueError(
            f"W and eigenvectors must have the same matrix size; got "
            f"W.shape={W.shape} and eigenvectors with length {N}."
        )

    dtype = np.result_type(W.dtype, E.dtype, np.complex128)
    W = W.astype(dtype, copy=False)
    E = E.astype(dtype, copy=False)

    active_rows = E.conj().T @ W
    active_columns = W @ E
    active_block = active_rows @ E

    # R W R + E diag(diag(E.H W E)) E.H, expanded without forming R.
    active_and_diagonal = active_block.copy()
    diagonal_indices = np.arange(K)
    active_and_diagonal[diagonal_indices, diagonal_indices] += np.diag(
        active_block
    )
    return (
        W
        - E @ active_rows
        - active_columns @ E.conj().T
        + (E @ active_and_diagonal) @ E.conj().T
    )


# Spectral construction


def largest_eigenvector(W):
    r"""Return :math:`1j\,e_i e_i^\dagger` for the largest mode.

    QuFlow matrices are skew-Hermitian, so "largest" refers to the largest
    real eigenvalue of ``-1j * W``.  One rank-one skew-Hermitian matrix is
    returned even when the largest eigenvalue is repeated.  Leading batch axes
    are allowed.
    """

    W = np.asarray(W)
    if W.ndim < 2 or W.shape[-2] != W.shape[-1] or W.shape[-1] == 0:
        raise ValueError(
            f"W must end with nonempty square matrix axes, got {W.shape}."
        )

    _, eigenvectors = np.linalg.eigh(-1j * W)
    eigenvector = eigenvectors[..., :, -1]
    return 1j * (
        eigenvector[..., :, None] * eigenvector[..., None, :].conj()
    )


def spectral_matrix(
    functions,
    level_sets,
    new_eigenvalues=None,
    *,
    superlevel=False,
    num_extra_eigenvectors=0,
    only_lowest=False,
    only_highest=False,
    return_closest_eigenvalues=False,
):
    """Select spectral blocks and assign new eigenvalues.

    By default, only the eigenvector closest to each level is selected.  With
    ``superlevel=True``, the selected block begins at that eigenvalue and
    includes all larger eigenvalues.  Positive ``num_extra_eigenvectors``
    extends the block downward, while negative values move its lower boundary
    upward.  ``only_lowest=True`` or ``only_highest=True`` retains only the
    corresponding endpoint eigenvector of the resulting block.  If
    ``new_eigenvalues`` is omitted, each block retains its closest eigenvalue.
    With ``return_closest_eigenvalues=True``, also return a list containing
    the closest original eigenvalue for each function and level.
    """
    functions = tuple(np.asarray(F) for F in functions)
    level_sets = np.asarray(level_sets, dtype=float).ravel()
    if not isinstance(num_extra_eigenvectors, int):
        raise ValueError("num_extra_eigenvectors must be an integer")
    if only_lowest and only_highest:
        raise ValueError("only_lowest and only_highest cannot both be True")
    if not functions:
        raise ValueError("functions and level_sets must not be empty.")
    if len(functions) != level_sets.size:
        raise ValueError("functions and level_sets must have equal lengths.")
    if not np.isfinite(level_sets).all():
        raise ValueError("level_sets must be finite real values.")

    if new_eigenvalues is None:
        new_eigenvalues = (None,) * len(functions)
    else:
        try:
            new_eigenvalues = tuple(new_eigenvalues)
        except TypeError:
            new_eigenvalues = (new_eigenvalues,) * len(functions)
        if len(new_eigenvalues) != len(functions):
            raise ValueError(
                "new_eigenvalues and functions must have equal lengths."
            )

    matrix = 0.0
    closest_eigenvalues = []
    for function_index, (F, level, new_eigenvalue) in enumerate(
        zip(functions, level_sets, new_eigenvalues)
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(-1j * F)
        closest_index = np.argmin(np.abs(eigenvalues - level))
        closest_eigenvalue = eigenvalues[closest_index]
        closest_eigenvalues.append(closest_eigenvalue.item())

        if superlevel:
            keep = np.isclose(eigenvalues, closest_eigenvalue)
            keep |= eigenvalues > closest_eigenvalue
        else:
            keep = np.zeros(eigenvalues.size, dtype=bool)
            keep[closest_index] = True

        if num_extra_eigenvectors > 0:
            lower_indices = np.flatnonzero(
                (eigenvalues < closest_eigenvalue) & ~keep
            )
            if num_extra_eigenvectors > lower_indices.size:
                raise ValueError(
                    f"num_extra_eigenvectors={num_extra_eigenvectors} exceeds "
                    f"the {lower_indices.size} eigenvalues below the closest "
                    f"eigenvalue of functions[{function_index}]."
                )
            keep[lower_indices[-num_extra_eigenvectors:]] = True
        elif num_extra_eigenvectors < 0:
            selected_indices = np.flatnonzero(keep)
            remove_count = -num_extra_eigenvectors
            if remove_count >= selected_indices.size:
                raise ValueError(
                    f"num_extra_eigenvectors={num_extra_eigenvectors} "
                    f"removes all {selected_indices.size} selected "
                    f"eigenvectors of functions[{function_index}]."
                )
            keep[selected_indices[:remove_count]] = False

        if only_lowest or only_highest:
            selected_indices = np.flatnonzero(keep)
            selected_index = (
                selected_indices[-1] if only_highest else selected_indices[0]
            )
            keep[:] = False
            keep[selected_index] = True

        U = eigenvectors[:, keep]
        value = (
            closest_eigenvalue if new_eigenvalue is None else new_eigenvalue
        )
        matrix = matrix + np.asarray(value)[..., None, None] * (U @ U.conj().T)

    if return_closest_eigenvalues:
        return matrix, closest_eigenvalues
    return matrix


def constraint_matrix(
    functions,
    level_sets,
    keep_eigenvalues=False,
    *,
    superlevel=False,
    new_eigenvalues=None,
):
    """Construct a skew-Hermitian constraint matrix and report its levels.

    Replacement eigenvalues default to distinct labels from 1 to 2.  In
    superlevel mode, every eigenvector from the closest eigenvalue upward gets
    the corresponding replacement value.  Set ``keep_eigenvalues=True`` to
    use the closest original eigenvalues instead.  Returns the constraint
    matrix and a list containing the closest original eigenvalue for every
    function and level pair.
    """
    functions = tuple(functions)
    if keep_eigenvalues and new_eigenvalues is not None:
        raise ValueError(
            "keep_eigenvalues and new_eigenvalues cannot be used together."
        )
    if not keep_eigenvalues and new_eigenvalues is None:
        new_eigenvalues = np.linspace(1.0, 2.0, len(functions))
    matrix, closest_eigenvalues = spectral_matrix(
        functions,
        level_sets,
        new_eigenvalues,
        superlevel=superlevel,
        return_closest_eigenvalues=True,
    )
    return 1j * matrix, closest_eigenvalues


def _sum_prescribed_trace(prescribed_trace):
    """Return a finite real scalar from one value or summed contributions."""
    if isinstance(prescribed_trace, np.ndarray):
        contributions = prescribed_trace
    elif np.isscalar(prescribed_trace):
        contributions = np.asarray(prescribed_trace)
    else:
        try:
            contributions = np.asarray(tuple(prescribed_trace))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "prescribed_trace must be a finite real scalar or an iterable "
                "of finite real scalars."
            ) from error

    if contributions.ndim > 1 or not np.issubdtype(
        contributions.dtype, np.number
    ):
        raise ValueError(
            "prescribed_trace must be a finite real scalar or an iterable of "
            "finite real scalars."
        )
    if not np.isfinite(contributions).all():
        raise ValueError("prescribed_trace values must be finite.")
    if not np.isreal(contributions).all():
        raise ValueError("prescribed_trace values must be real.")
    if contributions.ndim == 0:
        prescribed_trace = contributions.real.item()
    else:
        prescribed_trace = sum(contributions.real.tolist())
    try:
        finite_sum = np.isfinite(prescribed_trace)
    except TypeError:
        finite_sum = False
    if not finite_sum:
        raise ValueError("prescribed_trace values must have a finite sum.")
    return prescribed_trace


def project_domain(W, functions, level_sets, prescribed_trace=0.0):
    """Project ``W`` and prescribe the imaginary part of its trace.

    Each excluded block begins at the eigenvalue closest to its paired level
    and includes every larger eigenvalue.  The excluded blocks and their cross
    terms with the retained domain are set to zero.  The trace is adjusted
    through the retained block.  Derived block projectors are assumed pairwise
    orthogonal.  ``prescribed_trace`` is the desired imaginary part of the
    result's trace.  It may be one finite real value or an iterable of finite
    real contributions, which are summed first.  It defaults to zero.

    ``W`` may be one square matrix or a batch whose final two axes are square.
    """

    W = np.asarray(W)
    if W.ndim < 2 or W.shape[-2] != W.shape[-1]:
        raise ValueError(f"W must end with square matrix axes, got {W.shape}.")

    prescribed_trace = _sum_prescribed_trace(prescribed_trace)
    functions = tuple(np.asarray(F) for F in functions)
    level_sets = np.asarray(level_sets, dtype=float).ravel()
    if len(functions) != level_sets.size:
        raise ValueError("functions and level_sets must have equal lengths.")
    if not np.isfinite(level_sets).all():
        raise ValueError("level_sets must be finite real values.")
    if any(F.shape != W.shape[-2:] for F in functions):
        raise ValueError(
            f"functions must have the same matrix shape as W, {W.shape[-2:]}."
        )

    if functions:
        excluded_projector = spectral_matrix(
            functions,
            level_sets,
            1.0,
            superlevel=True,
        )
    else:
        excluded_projector = 0.0

    identity = np.eye(W.shape[-1], dtype=W.dtype)
    domain_projector = identity - excluded_projector
    projected = domain_projector @ W @ domain_projector

    domain_rank = np.trace(domain_projector)
    if np.isclose(domain_rank, 0.0, rtol=1e-12, atol=1e-12):
        if prescribed_trace != 0.0:
            raise ValueError(
                "Cannot prescribe a nonzero trace because the excluded "
                "blocks span the full matrix space."
            )
        return np.zeros_like(projected)

    trace = np.trace(projected, axis1=-2, axis2=-1)
    target_trace = 0.0 if prescribed_trace == 0.0 else 1j * prescribed_trace
    trace_shift = np.asarray((target_trace - trace) / domain_rank)
    return projected + trace_shift[..., None, None] * domain_projector


# Constraint-aware plotting


class ConstraintPlotter:
    """Plot states with fixed level-set contours and an optional subtraction.

    Instances are callable and return the image artist created by
    :func:`quflow.plot`.  The same instance can be supplied to
    :class:`quflow.Animation` through its ``plotter`` argument, which keeps
    the contours fixed while updating the state beneath them.

    ``subtract_function`` is interpreted as a fixed field.  If it is
    callable, it is evaluated once on a zero array shaped like the first
    constraint function.

    Plotting uses at least ``min_N=256`` samples in latitude (and ``2*N-1``
    in longitude).  Above that floor, ``N`` is inherited from the plotted
    state unless supplied explicitly.  Set ``min_N=None`` to allow lower
    plotting resolutions.
    """

    def __init__(
        self,
        functions,
        level_sets,
        subtract_function=None,
        *,
        min_N=256,
        contour_kwargs=None,
        **plot_kwargs,
    ):
        self.functions = tuple(np.array(F, copy=True) for F in functions)
        self.level_sets = np.asarray(level_sets, dtype=float).ravel()
        if not self.functions:
            raise ValueError("functions and level_sets must not be empty.")
        if len(self.functions) != self.level_sets.size:
            raise ValueError("functions and level_sets must have equal lengths.")
        if not np.isfinite(self.level_sets).all():
            raise ValueError("level_sets must be finite real values.")

        reference = self.functions[0]
        if subtract_function is None:
            self.subtract_function = None
        else:
            if callable(subtract_function):
                subtract_function = subtract_function(np.zeros_like(reference))
            subtraction = np.asarray(subtract_function)
            if subtraction.shape != reference.shape:
                raise ValueError(
                    "subtract_function must have the same shape as "
                    f"functions[0], got {subtraction.shape} and "
                    f"{reference.shape}."
                )
            self.subtract_function = np.array(subtraction, copy=True)

        self.min_N = self._validate_resolution(min_N, "min_N", allow_none=True)
        self.contour_kwargs = self._normalize_contour_kwargs(contour_kwargs)
        self.plot_kwargs = dict(plot_kwargs)
        self._validate_plot_kwargs(self.plot_kwargs)
        if self.plot_kwargs.get("N") is not None:
            self._validate_resolution(self.plot_kwargs["N"], "N")
        self._contour_fun_cache = {}

    @staticmethod
    def _validate_resolution(value, name, *, allow_none=False):
        if value is None and allow_none:
            return None
        error_message = f"{name} must be a positive integer"
        if np.iscomplexobj(value):
            raise ValueError(error_message)
        try:
            resolution = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(error_message) from error
        if (
            np.ndim(value) != 0
            or isinstance(value, (bool, np.bool_))
            or resolution != value
            or resolution < 1
        ):
            raise ValueError(error_message)
        return resolution

    @staticmethod
    def _infer_resolution(state):
        """Infer spherical bandwidth from matrix, function, or coefficients."""
        state = np.asarray(state)
        if state.ndim == 2:
            if state.shape[0] < 1:
                raise ValueError("Cannot infer N from an empty state.")
            return state.shape[0]
        if state.ndim == 1 and state.size:
            return int(np.ceil(np.sqrt(state.size)))
        raise ValueError(
            "Cannot infer N: state must be a nonempty one- or two-dimensional "
            "array."
        )

    def _resolve_resolution(self, state, N=None):
        if N is None:
            N = self._infer_resolution(state)
        else:
            N = self._validate_resolution(N, "N")
        if self.min_N is not None:
            N = max(N, self.min_N)
        return N

    @staticmethod
    def _validate_plot_kwargs(plot_kwargs):
        reserved = {"contours", "contour_data", "contour_kwargs"}
        unsupported = reserved.intersection(plot_kwargs)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"{names} cannot be used with ConstraintPlotter; configure "
                "the fixed level-set contours when constructing the plotter."
            )

    def _normalize_contour_kwargs(self, contour_kwargs):
        """Return one independent contour keyword dictionary per field."""
        if contour_kwargs is None:
            kwargs_per_contour = [{} for _ in self.functions]
        elif isinstance(contour_kwargs, Mapping):
            kwargs_per_contour = [
                dict(contour_kwargs) for _ in self.functions
            ]
        else:
            try:
                kwargs_per_contour = [dict(kwargs) for kwargs in contour_kwargs]
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "contour_kwargs must be a mapping or one mapping per "
                    "constraint function."
                ) from error
            if len(kwargs_per_contour) != len(self.functions):
                raise ValueError(
                    "A contour_kwargs sequence must have one entry per "
                    "constraint function."
                )

        for kwargs in kwargs_per_contour:
            if "levels" in kwargs:
                raise ValueError(
                    "contour levels are set by level_sets, not contour_kwargs."
                )
        return tuple(kwargs_per_contour)

    def prepare(self, state):
        """Return the state after removing the configured fixed field."""
        state = np.asarray(state)
        if self.subtract_function is None:
            return state
        try:
            return state - self.subtract_function
        except ValueError as error:
            raise ValueError(
                f"state shape {state.shape} is incompatible with subtraction "
                f"shape {self.subtract_function.shape}."
            ) from error

    def _contour_functions(self, N):
        """Convert and cache all contour fields at plotting resolution ``N``."""
        cache_key = None if N is None else int(N)
        if cache_key not in self._contour_fun_cache:
            contour_functions = []
            for field in self.functions:
                if N is not None:
                    field = qf.graphics.resample(field, N)
                contour_fun = qf.as_fun(field)
                if np.iscomplexobj(contour_fun):
                    contour_fun = contour_fun.real
                contour_functions.append(contour_fun)
            self._contour_fun_cache[cache_key] = tuple(contour_functions)
        return self._contour_fun_cache[cache_key]

    @staticmethod
    def _uses_cartopy(ax):
        """Return whether ``ax`` expects geographic data in degrees."""
        return qf.graphics._is_cartopy_axes(ax)

    def _draw_contours(self, ax, N, user_annotate=None):
        use_cartopy = self._uses_cartopy(ax)
        for contour_fun, level, user_kwargs in zip(
            self._contour_functions(N),
            self.level_sets,
            self.contour_kwargs,
        ):
            lon = np.linspace(
                -np.pi,
                np.pi,
                contour_fun.shape[1],
                endpoint=False,
            )
            lat = np.linspace(
                -np.pi / 2,
                np.pi / 2,
                contour_fun.shape[0],
            )
            kwargs = {
                "colors": "black",
                "linewidths": 1.0,
                "negative_linestyles": "solid",
            }
            kwargs.update(user_kwargs)
            kwargs["levels"] = [level]
            if use_cartopy:
                lon = np.rad2deg(lon)
                lat = np.rad2deg(lat)
                kwargs.setdefault("transform", qf.graphics.ccrs.PlateCarree())
            ax.contour(lon, lat, contour_fun, **kwargs)

        if user_annotate is not None:
            user_annotate(ax)

    def __call__(self, state, **plot_kwargs):
        """Plot ``state`` with the configured subtraction and contours."""
        kwargs = {**self.plot_kwargs, **plot_kwargs}
        self._validate_plot_kwargs(kwargs)
        N = self._resolve_resolution(state, kwargs.get("N"))
        kwargs["N"] = N
        user_annotate = kwargs.pop("annotate", None)
        im = qf.plot(self.prepare(state), **kwargs)
        im.axes.set_autoscale_on(False)
        self._draw_contours(im.axes, N, user_annotate=user_annotate)
        im._quflow_constraint_plotter_N = N
        return im

    def update(self, im, state, *, N=None):
        """Update ``im`` without redrawing contours or changing resolution."""
        image_N = getattr(im, "_quflow_constraint_plotter_N", None)
        if image_N is not None:
            if N is not None:
                requested_N = self._resolve_resolution(state, N)
                if requested_N != image_N:
                    raise ValueError(
                        f"Cannot update an image created with N={image_N} "
                        f"using N={requested_N}. Create a new image to change "
                        "plotting resolution."
                    )
            N = image_N
        elif N is None:
            N = self.plot_kwargs.get("N")
        N = self._resolve_resolution(state, N)

        data = self.prepare(state)
        if N is not None:
            data = qf.graphics.resample(data, N)
        fun = qf.as_fun(data)
        if np.iscomplexobj(fun):
            fun = fun.real

        if hasattr(im, "get_array") and np.size(im.get_array()) != fun.size:
            raise ValueError(
                "The updated state has a different plotting resolution from "
                "the existing image. Pass a consistent N to the plotter and "
                "Animation."
            )
        if hasattr(im, "set_data"):
            im.set_data(fun)
        elif hasattr(im, "set_array"):
            im.set_array(fun.ravel())
        else:
            raise AttributeError("Could not find method for setting data.")
        return im


def plotter(
    functions,
    level_sets,
    subtract_function=None,
    *,
    min_N=256,
    contour_kwargs=None,
    **plot_kwargs,
):
    """Return a reusable :class:`ConstraintPlotter`.

    The effective plotting bandwidth is ``max(256, N)`` by default.  If ``N``
    is omitted, it is inferred from each initial state.  Set ``min_N=None``
    to disable the default floor.

    Examples
    --------
    Create one plotter and reuse it for every state::

        cplot = plotter(functions, level_sets, N=N)
        cplot(W0)

        with qf.Animation("simulation.mp4", plotter=cplot) as animation:
            for state, time in zip(states, times):
                animation.update(state, time=time)
    """
    return ConstraintPlotter(
        functions,
        level_sets,
        subtract_function,
        min_N=min_N,
        contour_kwargs=contour_kwargs,
        **plot_kwargs,
    )


# Constrained Poisson solver


class CommutatorPoissonSolver:
    """Reusable Poisson solver enforcing ``bracket(F_c, P) == 0``.

    A positive active count ``K`` assumes ``K`` distinct nonzero eigenvalues
    and an ``(N-K)``-fold zero eigenvalue.  Pass zero to detect the rank
    numerically.  Alternatively, pass ``eigenvectors=[e_1, ..., e_K]`` to
    construct the same distinct-active-mode constraint directly, without an
    interpolative decomposition or a stored constraint matrix.  Construction
    briefly selects QuFlow's generic Poisson mode and is not thread-safe with
    concurrent Poisson calls.
    """

    def __init__(
        self,
        constraint_matrix=None,
        num_active_eigenvalues=None,
        rng=None,
        *,
        eigenvectors=None,
    ):
        if eigenvectors is not None and constraint_matrix is not None:
            raise ValueError(
                "Pass either constraint_matrix or eigenvectors, not both."
            )

        if eigenvectors is not None:
            active_eigenvectors = _orthonormal_eigenvector_columns(eigenvectors)
            N, K = active_eigenvectors.shape

            if num_active_eigenvalues is not None:
                count_error = (
                    "num_active_eigenvalues must equal the number of supplied "
                    f"eigenvectors ({K}), got {num_active_eigenvalues}."
                )
                try:
                    supplied_count = int(num_active_eigenvalues)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(count_error) from error
                if (
                    np.ndim(num_active_eigenvalues) != 0
                    or supplied_count != num_active_eigenvalues
                    or supplied_count != K
                ):
                    raise ValueError(count_error)

            self.matrix_size = N
            self.num_active_eigenvalues = K
            self.rng = np.random.default_rng(rng)
            self.constraint_rank = K * (2 * N - K - 1)
            self.constraint_operator = _EigenvectorConstraintOperator(
                active_eigenvectors
            )
            # Keep the established attribute name, but deliberately store a
            # LinearOperator rather than an r-by-N^2 matrix in this mode.
            self.constraint_row_basis = self.constraint_operator
            self.constraint_eigenvectors = active_eigenvectors
            self._uses_eigenvector_constraints = True
        else:
            if constraint_matrix is None:
                raise ValueError(
                    "Pass a constraint_matrix and num_active_eigenvalues, or "
                    "pass eigenvectors."
                )
            self._initialize_matrix_constraints(
                constraint_matrix, num_active_eigenvalues, rng
            )

        num_rows = self.constraint_operator.shape[0]
        print(
            f"CommutatorPoissonSolver: retained {num_rows} constraint rows.",
            flush=True,
        )
        self._factor_schur_complement()

    @classmethod
    def from_eigenvectors(cls, eigenvectors):
        """Construct a solver directly from an iterable of active vectors."""
        return cls(eigenvectors=eigenvectors)

    def __setstate__(self, state):
        """Restore operator attributes missing from older solver pickles."""
        self.__dict__.update(state)
        if "constraint_operator" not in state:
            self.constraint_operator = spla.aslinearoperator(
                self.constraint_row_basis.astype(complex, copy=False)
            )
        if "_uses_eigenvector_constraints" not in state:
            self._uses_eigenvector_constraints = False

    def _initialize_matrix_constraints(
        self, constraint_matrix, num_active_eigenvalues, rng
    ):
        """Initialize the backward-compatible commutator/ID construction."""
        constraint_matrix = np.asarray(constraint_matrix)
        shape = constraint_matrix.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"constraint_matrix must be square, got shape {shape}."
            )

        N = shape[0]
        count_error = (
            "num_active_eigenvalues must be an integer between 0 and "
            f"{N}, got {num_active_eigenvalues}."
        )
        try:
            K = int(num_active_eigenvalues)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(count_error) from error
        if (
            np.ndim(num_active_eigenvalues) != 0
            or K != num_active_eigenvalues
            or not 0 <= K <= N
        ):
            raise ValueError(count_error)

        self.matrix_size = N
        self.num_active_eigenvalues = K
        self.rng = np.random.default_rng(rng)
        self.constraint_rank = K * (2 * N - K - 1)
        self.constraint_row_basis = self._select_constraint_row_basis(
            constraint_matrix
        )
        self.constraint_operator = spla.aslinearoperator(
            self.constraint_row_basis.astype(complex, copy=False)
        )
        self._uses_eigenvector_constraints = False

    def _select_constraint_row_basis(self, constraint_matrix):
        """Select normalized independent rows of the commutator."""
        N2 = self.matrix_size**2
        automatic_rank = self.num_active_eigenvalues == 0
        empty = sp.csr_matrix((0, N2), dtype=constraint_matrix.dtype)
        if self.constraint_rank == 0 and not automatic_rank:
            return empty

        commutator = _commutator_operator(constraint_matrix)
        norms = _commutator_coordinate_row_norms(constraint_matrix)
        max_norm = float(np.max(norms))
        if automatic_rank and max_norm == 0.0:
            return empty

        cutoff = _ROW_NORM_TOL * max_norm
        usable = norms > cutoff
        num_usable = int(np.count_nonzero(usable))
        if not automatic_rank and num_usable < self.constraint_rank:
            raise ValueError(
                f"Only {num_usable} commutator rows have norm above "
                f"{cutoff:.3e}, but the expected rank is {self.constraint_rank}."
            )

        inverse_norms = np.zeros_like(norms)
        inverse_norms[usable] = 1.0 / norms[usable]
        inverse_norm_operator = spla.aslinearoperator(
            sp.diags(inverse_norms, format="csr")
        )

        # Columns of C.H are conjugated rows of C.  Right scaling makes those
        # columns unit norm before interpolative decomposition.
        normalized_adjoint = commutator.H @ inverse_norm_operator
        if automatic_rank:
            detected_rank, pivots, _ = scipy.linalg.interpolative.interp_decomp(
                normalized_adjoint,
                _ROW_NORM_TOL,
                rng=self.rng,
            )
            self.constraint_rank = int(detected_rank)
        else:
            pivots, _ = scipy.linalg.interpolative.interp_decomp(
                normalized_adjoint,
                self.constraint_rank,
                rng=self.rng,
            )

        if num_usable < self.constraint_rank:
            raise RuntimeError(
                "Interpolative decomposition detected more independent rows "
                "than have norm above the row cutoff."
            )
        if self.constraint_rank == 0:
            return empty

        indices = np.asarray(pivots[: self.constraint_rank], dtype=int)
        if np.any(~usable[indices]):
            raise RuntimeError("Interpolative decomposition selected a zero row.")

        rows = _materialize_commutator_rows(constraint_matrix, indices)
        return rows.multiply(inverse_norms[indices, np.newaxis]).tocsr()

    def _factor_schur_complement(self):
        """Materialize and LU-factor ``-V A**-1 V*`` for reuse."""
        if self.constraint_rank == 0:
            self._schur_factorization = None
            return

        N = self.matrix_size
        V = self.constraint_operator
        schur = np.empty(
            (self.constraint_rank, self.constraint_rank), dtype=complex
        )
        coordinate = np.zeros(self.constraint_rank, dtype=complex)

        previous_mode = qf.laplacian.cpu.select_skewherm(False)
        try:
            for column in range(self.constraint_rank):
                coordinate[column] = 1.0
                poisson_right_hand_side = np.asarray(
                    V.rmatvec(coordinate)
                ).reshape(N, N)
                P = qf.solve_poisson(poisson_right_hand_side)
                schur[:, column] = -np.asarray(V.matvec(P.ravel())).ravel()
                coordinate[column] = 0.0
        finally:
            qf.laplacian.cpu.select_skewherm(previous_mode)

        self._schur_factorization = scipy.linalg.lu_factor(schur)

    def solve(self, W):
        """Solve the constrained Poisson equation for one matrix."""
        W = np.asarray(W)
        N = self.matrix_size
        if W.shape != (N, N):
            raise ValueError(f"W must have shape {(N, N)}, got {W.shape}.")

        W = W.astype(np.result_type(W.dtype, np.complex128), copy=False)
        P = qf.solve_poisson(W)

        if self.constraint_rank:
            V = self.constraint_operator
            use_skew_shortcut = self._uses_eigenvector_constraints and np.allclose(
                W.conj().T, -W, rtol=1e-10, atol=1e-12
            )
            if use_skew_shortcut:
                schur_rhs = V.matvec_skew_hermitian(P)
            else:
                schur_rhs = np.asarray(V.matvec(P.ravel())).ravel()
            multipliers = scipy.linalg.lu_solve(
                self._schur_factorization, -schur_rhs
            )
            if use_skew_shortcut:
                correction = V.rmatvec_skew_hermitian(multipliers).reshape(N, N)
            else:
                correction = np.asarray(V.rmatvec(multipliers)).reshape(N, N)
            P = qf.solve_poisson(W - correction)

        return P

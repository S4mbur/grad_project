"""
similarity_metrics.py
=====================
Vector similarity metrics tailored to the SkinSight MIL pipeline.

Drop this file at ``app/similarity_metrics.py`` inside the phase1_project
repository.  It is imported by both the experiment harness
(``scripts/experiment_similarity_metrics.py``) and the Flask server
(``app/server.py``).

The module collects the metrics studied in the
"Algebra-aware Similarity Search for Skin Pathology" experiment.
Each metric returns *similarity* scores (higher = more similar) so that
the existing ``argsort`` logic in ``_retrieve_similar_cases`` can be
reused unchanged.

Families
--------
1.  Generic baselines (cosine, negative Euclidean) -- agnostic to the
    trained model and serve as the reference point.
2.  Statistical baselines (Mahalanobis with class-conditional covariance,
    hubness-corrected cosine via mutual proximity) -- still classifier
    agnostic but data-aware.
3.  Classifier-pullback metrics that exploit the trained MIL classifier
    head ``f : R^d -> R^C``.  These reduce to meaningless noise when ``f``
    is not the model's own classifier.
4.  Information-geometric metrics (Fisher--Rao / Bhattacharyya) on the
    softmax simplex with the Chentsov uniqueness justification.

A composite "decomposed" metric mixes the diagnostic component
(Fisher--Rao on softmax) with a style component restricted to the kernel
of the classifier's local linearisation.  This is the metric the report
recommends as the production replacement for plain cosine.

The module is intentionally NumPy-only at the metric layer; it can be
imported in the Flask server without pulling Torch into the request path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

EPS = 1e-8


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return x / norm


def _softmax(z: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    z = z / max(float(temperature), EPS)
    z = z - z.max(axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.maximum(ez.sum(axis=axis, keepdims=True), EPS)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(x, 0.0, None))


# ---------------------------------------------------------------------------
# Classifier description
# ---------------------------------------------------------------------------

@dataclass
class ClassifierHead:
    """Linear approximation of the trained MIL classifier ``f : R^d -> R^C``.

    The MIL classifier in ``train_weak_supervision.py`` is a small MLP
    (Linear -> ReLU -> Dropout -> Linear).  At inference the network is
    deterministic, so we approximate it by its first-order expansion at a
    representative reference point ``z_ref`` (typically the bank centroid).

    All algebraic constructions in this module -- the row-space projector,
    the kernel projector, the Fisher information -- are defined relative
    to this linear approximation.  When the classifier is itself linear
    (as in a logistic-regression baseline) the approximation is exact.
    """

    W: np.ndarray  # shape (C, d)
    b: np.ndarray  # shape (C,)
    z_ref: Optional[np.ndarray] = None  # shape (d,)
    f_ref: Optional[np.ndarray] = None  # shape (C,)
    temperature: float = 1.0
    label: str = "linear"

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=np.float32)
        self.b = np.asarray(self.b, dtype=np.float32).reshape(-1)
        if self.z_ref is not None:
            self.z_ref = np.asarray(self.z_ref, dtype=np.float32).reshape(-1)
        if self.f_ref is not None:
            self.f_ref = np.asarray(self.f_ref, dtype=np.float32).reshape(-1)

    @property
    def C(self) -> int:
        return int(self.W.shape[0])

    @property
    def d(self) -> int:
        return int(self.W.shape[1])

    def logits(self, Z: np.ndarray) -> np.ndarray:
        """Linear logits ``W z + b``.  When the classifier was provided
        directly via Jacobian linearisation this is the local approximation
        of the true MLP output around ``z_ref``."""
        Z = _as_2d(Z)
        return Z @ self.W.T + self.b

    def probabilities(self, Z: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(Z), axis=-1, temperature=self.temperature)


# ---------------------------------------------------------------------------
# Algebraic decomposition utilities
# ---------------------------------------------------------------------------

def diagnostic_basis(head: ClassifierHead) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal bases for ``im(W^T)`` (diagnostic subspace) and
    ``ker(W)`` (style subspace).

    These are the two factors of the direct-sum decomposition
    ``R^d = im(W^T) (+) ker(W)`` that underlies the abstract-algebra view
    of the metric.

    Returns:
        U_diag : (d, r) orthonormal columns spanning im(W^T) where
                 r = rank(W).
        U_null : (d, d-r) orthonormal columns spanning ker(W).
    """
    W = head.W
    # W = U_W diag(s) V_W^T with U_W in (C,r), s in (r,), V_W in (d,r).
    # im(W^T) = span(V_W) and ker(W) = orthogonal complement.
    _, sigma, Vt = np.linalg.svd(W, full_matrices=True)
    rank = int((sigma > max(sigma.max(), 1.0) * 1e-6).sum()) if sigma.size else 0
    V = Vt.T  # (d, d)
    U_diag = V[:, :rank].astype(np.float32)
    U_null = V[:, rank:].astype(np.float32)
    return U_diag, U_null


def project_pair(head: ClassifierHead, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project rows of ``Z`` onto the diagnostic and style subspaces.

    Returns:
        Z_diag : projection coefficients in the diagnostic basis (n, r)
        Z_null : projection coefficients in the style basis (n, d-r)
    """
    Z = _as_2d(Z)
    U_diag, U_null = diagnostic_basis(head)
    return Z @ U_diag, Z @ U_null


# ---------------------------------------------------------------------------
# Pairwise metric implementations
# ---------------------------------------------------------------------------

def cosine_similarity(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = _l2_normalise(_as_2d(query))[0]
    B = _l2_normalise(_as_2d(bank))
    return (B @ q).astype(np.float32)


def negative_euclidean(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    q = _as_2d(query)[0]
    B = _as_2d(bank)
    diff = B - q[None, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
    return (-dist).astype(np.float32)


def mahalanobis_similarity(
    query: np.ndarray,
    bank: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    q = _as_2d(query)[0]
    B = _as_2d(bank)
    diff = B - q[None, :]
    dist = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    dist = np.sqrt(np.maximum(dist, 0.0))
    return (-dist).astype(np.float32)


def fit_inverse_covariance(
    bank: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Class-pooled covariance with Ledoit--Wolf style shrinkage.

    Within-class covariance is averaged across classes when ``labels`` are
    provided; otherwise the full data covariance is used.  Shrinkage
    blends towards the diagonal to keep the matrix invertible in the
    typical d > n_per_class regime.
    """
    X = _as_2d(bank).astype(np.float64)
    if labels is None:
        cov = np.cov(X, rowvar=False, bias=True)
    else:
        labels_arr = np.asarray(list(labels))
        covs = []
        weights = []
        for c in np.unique(labels_arr):
            Xc = X[labels_arr == c]
            if Xc.shape[0] < 2:
                continue
            covs.append(np.cov(Xc, rowvar=False, bias=True))
            weights.append(Xc.shape[0])
        if not covs:
            cov = np.cov(X, rowvar=False, bias=True)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            cov = sum(w * c for w, c in zip(weights_arr, covs)) / weights_arr.sum()
    diag_target = np.eye(cov.shape[0]) * float(np.trace(cov) / max(cov.shape[0], 1))
    cov = (1.0 - shrinkage) * cov + shrinkage * diag_target
    inv = np.linalg.pinv(cov)
    return inv.astype(np.float32)


def pullback_euclidean_similarity(
    query_logits: np.ndarray,
    bank_logits: np.ndarray,
) -> np.ndarray:
    """Negative Euclidean distance in the *logit* space defined by the
    classifier head.  This is the simplest classifier-pullback metric."""
    q = _as_2d(query_logits)[0]
    B = _as_2d(bank_logits)
    diff = B - q[None, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
    return (-dist).astype(np.float32)


def symmetrised_kl_similarity(
    query_probs: np.ndarray,
    bank_probs: np.ndarray,
) -> np.ndarray:
    """Negative symmetrised KL divergence between class-probability
    distributions.  Bounded above by 0; the full Jensen--Shannon would
    saturate too quickly for our well-separated softmaxes."""
    q = np.clip(_as_2d(query_probs)[0], EPS, 1.0)
    B = np.clip(_as_2d(bank_probs), EPS, 1.0)
    kl_qb = (q * (np.log(q) - np.log(B))).sum(axis=-1)
    kl_bq = (B * (np.log(B) - np.log(q))).sum(axis=-1)
    return (-(kl_qb + kl_bq) * 0.5).astype(np.float32)


def fisher_rao_similarity(
    query_probs: np.ndarray,
    bank_probs: np.ndarray,
) -> np.ndarray:
    """Fisher--Rao distance on the categorical simplex,
    ``d_FR(p,q) = 2 * arccos(sum_c sqrt(p_c q_c))``.

    Returned as similarity = pi - distance (so that range is [0, pi]).
    Chentsov's theorem characterises this metric as the unique (up to
    constant) Riemannian metric on a categorical model invariant under
    sufficient statistics; it is the natural distance the trained
    classifier induces on its prediction simplex.
    """
    q = _as_2d(query_probs)[0]
    B = _as_2d(bank_probs)
    rho = (_safe_sqrt(B) * _safe_sqrt(q)[None, :]).sum(axis=-1)
    rho = np.clip(rho, -1.0, 1.0)
    distance = 2.0 * np.arccos(rho)
    return (math.pi - distance).astype(np.float32)


def bhattacharyya_affinity(
    query_probs: np.ndarray,
    bank_probs: np.ndarray,
) -> np.ndarray:
    q = _as_2d(query_probs)[0]
    B = _as_2d(bank_probs)
    return ((_safe_sqrt(B) * _safe_sqrt(q)[None, :]).sum(axis=-1)).astype(np.float32)


def style_cosine_similarity(
    query: np.ndarray,
    bank: np.ndarray,
    head: ClassifierHead,
) -> np.ndarray:
    """Cosine similarity restricted to the classifier's null space.

    The component along ``ker(W)`` carries information that does not affect
    the predicted class -- morphology, staining variation, scanner
    artefacts.  Two slides with similar style components are
    *within-class* analogues of each other.
    """
    _, U_null = diagnostic_basis(head)
    q_proj = _as_2d(query) @ U_null
    B_proj = _as_2d(bank) @ U_null
    q_proj = _l2_normalise(q_proj)[0]
    B_proj = _l2_normalise(B_proj)
    return (B_proj @ q_proj).astype(np.float32)


def decomposed_similarity(
    query: np.ndarray,
    bank: np.ndarray,
    head: ClassifierHead,
    alpha: float = 0.7,
) -> np.ndarray:
    """The recommended hybrid metric.

    ``alpha`` weights the diagnostic (Fisher--Rao on softmax(W z))
    contribution; ``1 - alpha`` weights the style-cosine contribution
    on ``ker(W)``.

    Both components are normalised to [0, 1] before mixing so the alpha
    interpretation is meaningful regardless of the dimensionalities of
    the two subspaces.
    """
    q_logits = head.logits(query)[0]
    B_logits = head.logits(bank)
    q_probs = _softmax(q_logits, temperature=head.temperature)
    B_probs = _softmax(B_logits, axis=-1, temperature=head.temperature)
    fr = fisher_rao_similarity(q_probs, B_probs)
    # fisher_rao_similarity returns pi - d_FR, with d_FR in [0, pi];
    # so fr in [0, pi].  Map linearly to [0, 1].
    fr_norm = np.clip(fr / math.pi, 0.0, 1.0)
    style = style_cosine_similarity(query, bank, head)
    style_norm = (style + 1.0) * 0.5  # cosine in [-1,1] -> [0,1]
    score = alpha * fr_norm + (1.0 - alpha) * style_norm
    return score.astype(np.float32)


# ---------------------------------------------------------------------------
# Original metrics: IGPD, ABD, DBRD
# ---------------------------------------------------------------------------
#
# These three metrics are introduced in the SkinSight thesis report.  They
# are designed for the specific MIL+classifier setup and are not direct
# adaptations of any existing similarity construction in the
# pathology-retrieval literature.
#
#   IGPD - Information-Geometric Pullback Distance
#          Riemannian metric on R^d induced by the classifier's Fisher
#          information; midpoint linearisation gives a closed-form
#          point-dependent Mahalanobis distance.
#
#   ABD  - Attention-Bures Distance
#          2-Wasserstein distance between attention-weighted Gaussian
#          surrogates (mu_z, Sigma_z) of each slide's tile distribution.
#          Stored low-rank (top-k eigenvectors of Sigma_z) for efficiency.
#
#   DBRD - Diagnostic Bures-Riemann Distance
#          The recommended composite: lambda-mix of IGPD on means and ABD
#          on attention covariances.  Reduces to either of the components
#          for lambda = 0 or lambda -> infinity respectively.


def _fisher_tensor_at(head: "ClassifierHead", z: np.ndarray) -> np.ndarray:
    """Classifier Fisher information tensor at point z.

    For our linearised classifier, J(z) = W (constant), so

        g(z) = W^T (diag(p(z)) - p(z) p(z)^T) W

    with p(z) = softmax(W z + b).  The operation is O(C^2 d) and we
    return the d x d matrix.
    """
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    p = head.probabilities(z)[0]
    A = np.diag(p) - np.outer(p, p)  # (C, C)
    return (head.W.T @ A @ head.W).astype(np.float32)


def igpd_distance_squared(
    query: np.ndarray,
    bank: np.ndarray,
    head: "ClassifierHead",
) -> np.ndarray:
    """Squared Information-Geometric Pullback Distance against the bank.

    Uses the midpoint linearisation,

        d^2(z, z') = (z - z')^T g((z + z') / 2) (z - z'),

    which simplifies to the variance of the projected difference under
    the midpoint class probabilities,

        d^2 = Var_{p_mid}(W (z - z')) = sum_c p_c (W d)_c^2 - (sum_c p_c (W d)_c)^2.

    This vectorises perfectly: we never form g explicitly.
    """
    q = _as_2d(query)[0]
    B = _as_2d(bank)
    diff = B - q[None, :]                          # (N, d)
    midpoints = 0.5 * (B + q[None, :])             # (N, d)
    p_mid = head.probabilities(midpoints)          # (N, C)
    proj = diff @ head.W.T                         # (N, C) = W(z - z')
    moment1 = (proj * p_mid).sum(axis=-1)          # (N,)
    moment2 = (proj * proj * p_mid).sum(axis=-1)   # (N,)
    var = np.maximum(moment2 - moment1 * moment1, 0.0)
    return var.astype(np.float32)


def igpd_similarity(
    query: np.ndarray,
    bank: np.ndarray,
    head: "ClassifierHead",
) -> np.ndarray:
    return (-np.sqrt(igpd_distance_squared(query, bank, head))).astype(np.float32)


def low_rank_covariance(
    features: np.ndarray,
    weights: np.ndarray,
    rank: int = 32,
    centre: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated eigendecomposition of an attention-weighted empirical
    covariance Sigma = sum_i a_i (h_i - mu)(h_i - mu)^T.

    Returns (mu, U, s) with ``Sigma ~= U diag(s) U^T``.  The function is
    pure NumPy and is intended to be called once per case at indexing
    time.
    """
    H = np.asarray(features, dtype=np.float64)
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    if a.sum() <= 0:
        a = np.ones_like(a)
    a = a / a.sum()
    mu = centre if centre is not None else (a[:, None] * H).sum(axis=0)
    Hc = H - mu[None, :]
    sqrt_a = np.sqrt(a)
    Hw = Hc * sqrt_a[:, None]                    # rows are sqrt(a_i) (h_i - mu)
    # Use SVD on Hw rather than forming d x d covariance explicitly.
    n_rank = int(min(rank, Hw.shape[0], Hw.shape[1]))
    U_left, sigma, Vt = np.linalg.svd(Hw, full_matrices=False)
    Vt = Vt[:n_rank]                             # (k, d)
    sigma = sigma[:n_rank]
    s = (sigma ** 2).astype(np.float32)
    U = Vt.T.astype(np.float32)                  # (d, k)
    return mu.astype(np.float32), U, s


def bures_squared_lowrank(
    U_a: np.ndarray, s_a: np.ndarray,
    U_b: np.ndarray, s_b: np.ndarray,
) -> float:
    """Bures-Wasserstein squared distance between two low-rank PSD
    matrices Sigma_a = U_a diag(s_a) U_a^T and Sigma_b similarly.

    Uses the identity tr((A^{1/2} B A^{1/2})^{1/2}) computed in the
    span of U_a, which is O(k^3).
    """
    s_a = np.maximum(np.asarray(s_a, dtype=np.float64), 0.0)
    s_b = np.maximum(np.asarray(s_b, dtype=np.float64), 0.0)
    if s_a.size == 0 or s_b.size == 0:
        return float(s_a.sum() + s_b.sum())
    sqrt_sa = np.sqrt(s_a)
    cross = U_a.T @ U_b                          # (k_a, k_b)
    M = (sqrt_sa[:, None] * cross) * np.sqrt(s_b)[None, :]   # (k_a, k_b)
    # eigenvalues of (Sigma_a^{1/2} Sigma_b Sigma_a^{1/2}) within span(U_a)
    # are squares of singular values of M.
    sing = np.linalg.svd(M, compute_uv=False)
    trace_sqrt = float(np.abs(sing).sum())
    bures2 = float(s_a.sum() + s_b.sum() - 2.0 * trace_sqrt)
    return max(bures2, 0.0)


def abd_distance_squared(
    q_mu: np.ndarray, q_U: np.ndarray, q_s: np.ndarray,
    bank_mus: np.ndarray, bank_U: np.ndarray, bank_s: np.ndarray,
) -> np.ndarray:
    """Pairwise Attention-Bures Distance squared.

    ``bank_U`` has shape (N, d, k); ``bank_s`` has shape (N, k).  The
    function loops over the bank, which is acceptable for N <= a few
    thousand and pulls in the Bures formula vectorised inside the loop.
    """
    q_mu = np.asarray(q_mu, dtype=np.float32).reshape(-1)
    diffs = bank_mus - q_mu[None, :]
    mean_term = np.einsum("ij,ij->i", diffs, diffs).astype(np.float64)
    bures = np.empty(bank_mus.shape[0], dtype=np.float64)
    for i in range(bank_mus.shape[0]):
        bures[i] = bures_squared_lowrank(q_U, q_s, bank_U[i], bank_s[i])
    return (mean_term + bures).astype(np.float32)


def abd_similarity(
    q_mu: np.ndarray, q_U: np.ndarray, q_s: np.ndarray,
    bank_mus: np.ndarray, bank_U: np.ndarray, bank_s: np.ndarray,
) -> np.ndarray:
    return (-np.sqrt(abd_distance_squared(q_mu, q_U, q_s, bank_mus, bank_U, bank_s))).astype(np.float32)


def dbrd_similarity(
    q_mu: np.ndarray, q_U: np.ndarray, q_s: np.ndarray,
    bank_mus: np.ndarray, bank_U: np.ndarray, bank_s: np.ndarray,
    head: "ClassifierHead",
    lam: float = 1.0,
) -> np.ndarray:
    """Diagnostic Bures-Riemann Distance.

    d^2 = d_IGPD^2(q_mu, b_mu) + lam * d_Bures^2(q_Sigma, b_Sigma)
    """
    igpd2 = igpd_distance_squared(q_mu, bank_mus, head)
    bures2 = np.empty(bank_mus.shape[0], dtype=np.float64)
    for i in range(bank_mus.shape[0]):
        bures2[i] = bures_squared_lowrank(q_U, q_s, bank_U[i], bank_s[i])
    total = igpd2.astype(np.float64) + float(lam) * bures2
    return (-np.sqrt(np.maximum(total, 0.0))).astype(np.float32)


def mutual_proximity_cosine(
    query: np.ndarray,
    bank: np.ndarray,
    bank_self_sims: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hubness-corrected cosine similarity (Schnitzer et al., 2012).

    The raw cosine similarity ``s(q,b)`` is replaced by the empirical
    probability that another bank sample lies further from ``q`` than
    ``b`` *and* further from ``b`` than ``q``.  This dramatically reduces
    hub bias in high-dimensional retrieval.
    """
    base = cosine_similarity(query, bank)
    if bank_self_sims is None:
        bank_norm = _l2_normalise(_as_2d(bank))
        bank_self_sims = (bank_norm @ bank_norm.T).astype(np.float32)
    sorted_base = np.sort(base)
    p_q = np.searchsorted(sorted_base, base, side="right") / max(len(base), 1)
    p_b = np.empty_like(base)
    for i in range(bank_self_sims.shape[0]):
        row = np.sort(bank_self_sims[i])
        p_b[i] = np.searchsorted(row, base[i], side="right") / max(len(row), 1)
    return (p_q * p_b).astype(np.float32)


def mmpc_similarity(
    query: np.ndarray,
    bank: np.ndarray,
    inv_cov: np.ndarray,
    bank_self_mahalanobis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """MMPC -- Mahalanobis + Mutual Proximity Composite (original).

    The empirical observation in Section 6 / 8 is that two of the
    catalogue's metrics dominate disjoint axes of retrieval quality:

      * Mahalanobis improves *global* mAP@K because re-weighting
        directions by within-class covariance amplifies the signal that
        cosine spreads uniformly.
      * Mutual proximity cosine improves *operational* melanoma
        first-hit rank because it dampens hubs that a single dominant
        direction creates in the embedding space.

    MMPC composes them: the underlying dissimilarity is the
    Mahalanobis distance ``d_M``, and we apply Schnitzer's mutual
    proximity transform on the empirical CDF of ``d_M`` rather than on
    cosine.  This recovers the hubness fix in the metric where the
    re-weighting actually lives.

    Concretely, for query q and bank candidate b,

        s(q, b) = P_X[ d_M(q, X) >= d_M(q, b) ] *
                  P_Y[ d_M(b, Y) >= d_M(b, q) ],

    estimated from the bank's own d_M distribution.  Higher = more
    similar.

    The pre-computation ``bank_self_mahalanobis`` is the N x N pairwise
    distance matrix; it can be cached once per bank.  When omitted, the
    function recomputes it from ``bank`` and ``inv_cov``.
    """
    q = _as_2d(query)[0]
    B = _as_2d(bank)
    diff = B - q[None, :]
    base = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv_cov, diff), 0.0))
    if bank_self_mahalanobis is None:
        bank_self_mahalanobis = mahalanobis_pairwise(B, inv_cov)
    sorted_query = np.sort(base)
    # P_X(d_M(q, X) >= d_M(q, b)) = fraction of bank entries with d_M >=
    # the candidate's distance.  Since lower distance = more similar, we
    # want the right tail.
    rank_in_query = np.searchsorted(sorted_query, base, side="left")
    p_q = 1.0 - (rank_in_query / max(len(base), 1))
    p_b = np.empty_like(base)
    for i in range(B.shape[0]):
        row = np.sort(bank_self_mahalanobis[i])
        rank = np.searchsorted(row, base[i], side="left")
        p_b[i] = 1.0 - (rank / max(len(row), 1))
    score = (p_q * p_b).astype(np.float32)
    # Score is in [0, 1]; argmax order is what we want.  Self-pair gets a
    # near-zero distance and consequently the maximum p product, which the
    # caller masks out via leave-one-out.
    return score


def mahalanobis_pairwise(bank: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Symmetric pairwise Mahalanobis distance matrix for a bank."""
    X = _as_2d(bank)
    # ||x_i - x_j||_M^2 = (x_i - x_j)^T M (x_i - x_j)
    XM = X @ inv_cov                    # (N, d)
    XMX = (XM * X).sum(axis=-1)          # (N,) -- diag of X M X^T
    cross = XM @ X.T                     # (N, N)
    sq = XMX[:, None] + XMX[None, :] - 2.0 * cross
    sq = np.maximum(sq, 0.0)
    return np.sqrt(sq).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level dispatch
# ---------------------------------------------------------------------------

@dataclass
class MetricContext:
    """Bundle of precomputed bank-side artefacts shared by all metrics."""

    bank: np.ndarray
    head: Optional[ClassifierHead] = None
    bank_logits: Optional[np.ndarray] = None
    bank_probs: Optional[np.ndarray] = None
    inv_cov: Optional[np.ndarray] = None
    bank_self_cosine: Optional[np.ndarray] = None
    bank_self_mahalanobis: Optional[np.ndarray] = None
    decomposed_alpha: float = 0.7
    # Attention-covariance bank for ABD / DBRD.  ``bank_cov_U`` has shape
    # (N, d, k) and ``bank_cov_s`` has shape (N, k); both are produced by
    # ``scripts/build_attention_covariance_bank.py``.  Per-query mean is
    # taken from ``bank`` itself, so only the spectra need to be loaded.
    bank_cov_U: Optional[np.ndarray] = None
    bank_cov_s: Optional[np.ndarray] = None
    # Per-query attention covariance (set by the router when a query slide
    # has just been analysed, otherwise we approximate with a near-zero
    # dispersion -- which makes ABD collapse to the mean Euclidean term).
    query_cov_U: Optional[np.ndarray] = None
    query_cov_s: Optional[np.ndarray] = None
    dbrd_lambda: float = 1.0

    def ensure_logits(self) -> None:
        if self.head is None:
            return
        if self.bank_logits is None:
            self.bank_logits = self.head.logits(self.bank).astype(np.float32)
        if self.bank_probs is None:
            self.bank_probs = _softmax(
                self.bank_logits, axis=-1, temperature=self.head.temperature,
            ).astype(np.float32)


def available_metrics(context: MetricContext) -> List[str]:
    metrics = ["cosine", "negative_euclidean"]
    if context.inv_cov is not None:
        metrics.append("mahalanobis")
    if context.head is not None:
        metrics.extend([
            "pullback_euclidean",
            "symmetrised_kl",
            "fisher_rao",
            "bhattacharyya",
            "style_cosine",
            "decomposed",
            "igpd",  # original: information-geometric pullback distance
        ])
    if context.bank_self_cosine is not None:
        metrics.append("mutual_proximity_cosine")
    if context.inv_cov is not None:
        metrics.append("mmpc")  # original: Mahalanobis + MutualProx composite
    if context.bank_cov_U is not None and context.bank_cov_s is not None:
        metrics.append("abd")    # original: attention-Bures distance
        if context.head is not None:
            metrics.append("dbrd")   # original composite
    return metrics


def score(metric: str, query: np.ndarray, context: MetricContext) -> np.ndarray:
    """Compute similarity scores for ``query`` against ``context.bank``.

    Higher = more similar.  The dispatcher exists so that the server can
    accept ``metric`` as a request parameter without growing a long
    if/else chain.
    """
    metric = metric.lower()
    if metric == "cosine":
        return cosine_similarity(query, context.bank)
    if metric in {"negative_euclidean", "euclidean"}:
        return negative_euclidean(query, context.bank)
    if metric == "mahalanobis":
        if context.inv_cov is None:
            raise ValueError("Mahalanobis metric requires inv_cov in context")
        return mahalanobis_similarity(query, context.bank, context.inv_cov)
    if metric in {
        "pullback_euclidean",
        "symmetrised_kl",
        "fisher_rao",
        "bhattacharyya",
        "decomposed",
        "style_cosine",
    }:
        if context.head is None:
            raise ValueError(f"Metric '{metric}' requires a ClassifierHead")
        context.ensure_logits()
        if metric == "pullback_euclidean":
            q_logits = context.head.logits(query)[0]
            return pullback_euclidean_similarity(q_logits, context.bank_logits)
        if metric == "symmetrised_kl":
            q_probs = context.head.probabilities(query)[0]
            return symmetrised_kl_similarity(q_probs, context.bank_probs)
        if metric == "fisher_rao":
            q_probs = context.head.probabilities(query)[0]
            return fisher_rao_similarity(q_probs, context.bank_probs)
        if metric == "bhattacharyya":
            q_probs = context.head.probabilities(query)[0]
            return bhattacharyya_affinity(q_probs, context.bank_probs)
        if metric == "style_cosine":
            return style_cosine_similarity(query, context.bank, context.head)
        if metric == "decomposed":
            return decomposed_similarity(
                query, context.bank, context.head, alpha=context.decomposed_alpha,
            )
    if metric == "mutual_proximity_cosine":
        return mutual_proximity_cosine(query, context.bank, context.bank_self_cosine)
    if metric == "mmpc":
        if context.inv_cov is None:
            raise ValueError("MMPC requires inv_cov")
        if context.bank_self_mahalanobis is None:
            context.bank_self_mahalanobis = mahalanobis_pairwise(
                context.bank, context.inv_cov,
            )
        return mmpc_similarity(
            query, context.bank, context.inv_cov, context.bank_self_mahalanobis,
        )
    if metric == "igpd":
        if context.head is None:
            raise ValueError("IGPD metric requires a ClassifierHead")
        return igpd_similarity(query, context.bank, context.head)
    if metric in {"abd", "dbrd"}:
        if context.bank_cov_U is None or context.bank_cov_s is None:
            raise ValueError(f"Metric '{metric}' requires bank attention covariances")
        # If a per-query covariance is not provided we fall back to the
        # bank entry whose mean is closest to the query (leave-one-out
        # benchmark sets this explicitly).
        if context.query_cov_U is None or context.query_cov_s is None:
            # locate by exact mean match
            q_arr = _as_2d(query)[0]
            diffs = context.bank - q_arr[None, :]
            idx = int(np.argmin((diffs * diffs).sum(axis=-1)))
            q_U = context.bank_cov_U[idx]
            q_s = context.bank_cov_s[idx]
        else:
            q_U = context.query_cov_U
            q_s = context.query_cov_s
        if metric == "abd":
            return abd_similarity(
                query, q_U, q_s,
                context.bank, context.bank_cov_U, context.bank_cov_s,
            )
        return dbrd_similarity(
            query, q_U, q_s,
            context.bank, context.bank_cov_U, context.bank_cov_s,
            context.head,
            lam=float(context.dbrd_lambda),
        )
    raise ValueError(f"Unknown similarity metric: {metric}")


# ---------------------------------------------------------------------------
# Diagnostic / reporting helpers
# ---------------------------------------------------------------------------

def discriminative_spread(scores: np.ndarray) -> float:
    """Standard deviation of the score distribution -- the headline number
    used in the report to quantify the "concentration of measure" failure
    mode of cosine similarity."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size < 2:
        return 0.0
    return float(np.std(s))


def top_gap(scores: np.ndarray, k: int = 5) -> float:
    """Gap between the top-1 score and the k-th score."""
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if s.size <= k:
        return float(s[0] - s[-1]) if s.size else 0.0
    return float(s[0] - s[k])


__all__ = [
    "ClassifierHead",
    "MetricContext",
    "abd_distance_squared",
    "abd_similarity",
    "available_metrics",
    "bhattacharyya_affinity",
    "bures_squared_lowrank",
    "cosine_similarity",
    "dbrd_similarity",
    "decomposed_similarity",
    "diagnostic_basis",
    "discriminative_spread",
    "fisher_rao_similarity",
    "fit_inverse_covariance",
    "igpd_distance_squared",
    "igpd_similarity",
    "low_rank_covariance",
    "mahalanobis_pairwise",
    "mahalanobis_similarity",
    "mmpc_similarity",
    "mutual_proximity_cosine",
    "negative_euclidean",
    "project_pair",
    "pullback_euclidean_similarity",
    "score",
    "style_cosine_similarity",
    "symmetrised_kl_similarity",
    "top_gap",
]
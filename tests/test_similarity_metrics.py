"""Unit tests for ``app/similarity_metrics.py``.

Drop at ``tests/test_similarity_metrics.py``.  Runs with the standard
unittest harness; no GPU, model weights, or dataset access required.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import similarity_metrics as smet  # noqa: E402


class FisherRaoSimplexTests(unittest.TestCase):
    def test_identical_distributions_have_zero_distance(self):
        p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sim = smet.fisher_rao_similarity(p, p[None, :])
        self.assertAlmostEqual(float(sim[0]), math.pi, places=4)

    def test_disjoint_distributions_are_furthest(self):
        p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        sim = smet.fisher_rao_similarity(p, q[None, :])
        # 2*arccos(0) = pi -> similarity = pi - pi = 0.
        self.assertAlmostEqual(float(sim[0]), 0.0, places=4)

    def test_chentsov_invariance_to_uniform_renaming(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(4)).astype(np.float32)
        q = rng.dirichlet(np.ones(4)).astype(np.float32)
        perm = np.array([2, 0, 3, 1])
        d = smet.fisher_rao_similarity(p, q[None, :])[0]
        d_perm = smet.fisher_rao_similarity(p[perm], q[perm][None, :])[0]
        self.assertAlmostEqual(float(d), float(d_perm), places=5)


class DiagnosticBasisTests(unittest.TestCase):
    def test_decomposition_is_orthogonal(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((4, 16)).astype(np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(4, dtype=np.float32))
        U_diag, U_null = smet.diagnostic_basis(head)
        # Bases must be orthonormal (within tolerance).
        self.assertTrue(np.allclose(U_diag.T @ U_diag, np.eye(U_diag.shape[1]), atol=1e-5))
        self.assertTrue(np.allclose(U_null.T @ U_null, np.eye(U_null.shape[1]), atol=1e-5))
        # And mutually orthogonal.
        self.assertTrue(np.allclose(U_diag.T @ U_null, 0.0, atol=1e-5))
        # And together span R^d.
        full = np.concatenate([U_diag, U_null], axis=1)
        self.assertTrue(np.allclose(full @ full.T, np.eye(W.shape[1]), atol=1e-5))

    def test_kernel_kills_logits(self):
        rng = np.random.default_rng(2)
        W = rng.standard_normal((4, 32)).astype(np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(4, dtype=np.float32))
        _, U_null = smet.diagnostic_basis(head)
        # Anything in the kernel should produce zero logits.
        v = U_null @ rng.standard_normal(U_null.shape[1]).astype(np.float32)
        logits = head.logits(v)[0]
        self.assertTrue(np.allclose(logits, 0.0, atol=1e-4))


class MetricDispatchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.bank = rng.standard_normal((20, 32)).astype(np.float32)
        self.bank /= np.linalg.norm(self.bank, axis=1, keepdims=True) + 1e-8
        self.head = smet.ClassifierHead(
            W=rng.standard_normal((4, 32)).astype(np.float32),
            b=rng.standard_normal(4).astype(np.float32),
            z_ref=self.bank.mean(axis=0),
        )
        self.context = smet.MetricContext(
            bank=self.bank,
            head=self.head,
            inv_cov=smet.fit_inverse_covariance(self.bank),
            bank_self_cosine=self.bank @ self.bank.T,
        )
        self.context.ensure_logits()
        self.query = self.bank[0]

    def test_cosine_query_with_self_is_max(self):
        scores = smet.score("cosine", self.query, self.context)
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_decomposed_alpha_collapses_to_fisher_rao_at_one(self):
        self.context.decomposed_alpha = 1.0
        scores_decomp = smet.score("decomposed", self.query, self.context)
        scores_fr = smet.score("fisher_rao", self.query, self.context)
        # Both should pick the same top-1 (the query itself).
        self.assertEqual(int(np.argmax(scores_decomp)), int(np.argmax(scores_fr)))

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            smet.score("nonexistent", self.query, self.context)


class SyntheticBenchmark(unittest.TestCase):
    """Quick synthetic experiment that mirrors the report's protocol."""

    def test_decomposed_beats_cosine_on_classifier_aligned_data(self):
        rng = np.random.default_rng(4)
        d = 64
        n_per_class = 20
        # Random classifier matrix.
        W = rng.standard_normal((4, d)).astype(np.float32)
        # Class centroids in the diagnostic subspace.
        U_diag, U_null = smet.diagnostic_basis(
            smet.ClassifierHead(W=W, b=np.zeros(4, dtype=np.float32)),
        )
        centroids = rng.standard_normal((4, U_diag.shape[1])).astype(np.float32) * 2.0
        embeddings = []
        labels = []
        for c in range(4):
            cls_emb = (
                centroids[c][None, :] @ U_diag.T
                + 0.3 * rng.standard_normal((n_per_class, d)).astype(np.float32)
            )
            # Add lots of noise in the null space (style component).
            null_noise = rng.standard_normal((n_per_class, U_null.shape[1])).astype(np.float32) @ U_null.T
            embeddings.append(cls_emb + 1.5 * null_noise)
            labels.extend([c] * n_per_class)
        bank = np.concatenate(embeddings, axis=0)
        labels_arr = np.asarray(labels)
        head = smet.ClassifierHead(W=W, b=np.zeros(4, dtype=np.float32))
        ctx = smet.MetricContext(bank=bank, head=head)
        ctx.ensure_logits()

        cosine_recall = self._recall_at_k("cosine", bank, labels_arr, ctx, k=5)
        ctx.decomposed_alpha = 0.7
        decomp_recall = self._recall_at_k("decomposed", bank, labels_arr, ctx, k=5)
        # Decomposed should equal or beat cosine in this regime where
        # null-space noise dominates the raw embedding.
        self.assertGreaterEqual(decomp_recall + 1e-3, cosine_recall)

    def _recall_at_k(self, metric, bank, labels, ctx, k):
        n = bank.shape[0]
        hits = 0
        relevant = 0
        for i in range(n):
            scores = smet.score(metric, bank[i], ctx)
            scores[i] = -np.inf
            top = np.argsort(scores)[::-1][:k]
            n_relevant = (labels == labels[i]).sum() - 1
            relevant += n_relevant
            hits += (labels[top] == labels[i]).sum()
        return hits / max(relevant, 1)


class IGPDTests(unittest.TestCase):
    def test_self_distance_is_zero(self):
        rng = np.random.default_rng(10)
        d, C = 32, 4
        W = rng.standard_normal((C, d)).astype(np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(C, dtype=np.float32))
        z = rng.standard_normal(d).astype(np.float32)
        d2 = smet.igpd_distance_squared(z, z[None, :], head)
        self.assertAlmostEqual(float(d2[0]), 0.0, places=4)

    def test_zero_classifier_collapses_distance(self):
        rng = np.random.default_rng(11)
        d, C = 16, 4
        W = np.zeros((C, d), dtype=np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(C, dtype=np.float32))
        z = rng.standard_normal(d).astype(np.float32)
        z2 = rng.standard_normal(d).astype(np.float32)
        d2 = smet.igpd_distance_squared(z, z2[None, :], head)
        # No information => zero squared distance.
        self.assertAlmostEqual(float(d2[0]), 0.0, places=4)

    def test_distance_grows_in_diagnostic_direction(self):
        rng = np.random.default_rng(12)
        d, C = 32, 4
        W = rng.standard_normal((C, d)).astype(np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(C, dtype=np.float32))
        # diagnostic direction -- top right singular vector of W
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
        v_diag = Vt[0]
        v_null = np.linalg.svd(W, full_matrices=True)[2].T[:, -1]
        z = rng.standard_normal(d).astype(np.float32) * 0.1
        d_diag = smet.igpd_distance_squared(z, (z + 0.5 * v_diag)[None, :], head)
        d_null = smet.igpd_distance_squared(z, (z + 0.5 * v_null)[None, :], head)
        self.assertGreater(float(d_diag[0]), float(d_null[0]) + 1e-3)


class BuresTests(unittest.TestCase):
    def test_bures_self_distance_is_zero(self):
        rng = np.random.default_rng(20)
        U = np.linalg.qr(rng.standard_normal((16, 4)))[0].astype(np.float32)
        s = np.abs(rng.standard_normal(4)).astype(np.float32)
        b = smet.bures_squared_lowrank(U, s, U, s)
        self.assertAlmostEqual(b, 0.0, places=4)

    def test_bures_diagonal_case_matches_simple_formula(self):
        # For diagonal Sigma_a = diag(s_a), Sigma_b = diag(s_b) shared basis,
        # Bures squared = sum (sqrt(s_a) - sqrt(s_b))^2.
        d, k = 8, 4
        U = np.eye(d, k).astype(np.float32)
        s_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        s_b = np.array([0.5, 2.5, 2.0, 5.0], dtype=np.float32)
        expected = float(np.sum((np.sqrt(s_a) - np.sqrt(s_b)) ** 2))
        actual = smet.bures_squared_lowrank(U, s_a, U, s_b)
        self.assertAlmostEqual(actual, expected, places=4)


class LowRankCovarianceTests(unittest.TestCase):
    def test_attention_weighted_mean_recovers_bag_embedding(self):
        rng = np.random.default_rng(30)
        H = rng.standard_normal((20, 16)).astype(np.float32)
        a = np.abs(rng.standard_normal(20)).astype(np.float32)
        a /= a.sum()
        mu_expected = (a[:, None] * H).sum(axis=0)
        mu, _, _ = smet.low_rank_covariance(H, a, rank=4)
        self.assertTrue(np.allclose(mu, mu_expected, atol=1e-4))

    def test_low_rank_factor_reproduces_full_covariance_at_full_rank(self):
        rng = np.random.default_rng(31)
        H = rng.standard_normal((50, 8)).astype(np.float32)
        a = np.abs(rng.standard_normal(50)).astype(np.float32)
        a /= a.sum()
        mu, U, s = smet.low_rank_covariance(H, a, rank=8)
        Sigma_lr = U @ np.diag(s) @ U.T
        Hc = (H - mu) * np.sqrt(a)[:, None]
        Sigma_full = Hc.T @ Hc
        self.assertTrue(np.allclose(Sigma_lr, Sigma_full, atol=1e-3))


class MMPCTests(unittest.TestCase):
    def test_mmpc_self_score_is_maximum(self):
        rng = np.random.default_rng(50)
        d = 16
        bank = rng.standard_normal((30, d)).astype(np.float32)
        inv_cov = smet.fit_inverse_covariance(bank)
        scores = smet.mmpc_similarity(bank[0], bank, inv_cov)
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_mmpc_inherits_class_separation_from_mahalanobis(self):
        rng = np.random.default_rng(51)
        d = 8
        # Two classes well-separated.
        a = rng.standard_normal((20, d)).astype(np.float32) + np.array([5.0] + [0.0] * (d - 1))
        b = rng.standard_normal((20, d)).astype(np.float32) - np.array([5.0] + [0.0] * (d - 1))
        bank = np.vstack([a, b]).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 20)
        inv_cov = smet.fit_inverse_covariance(bank, labels=labels.tolist())
        # Self-class candidate should outrank cross-class candidate.
        scores = smet.mmpc_similarity(a[0], bank, inv_cov)
        scores[0] = -np.inf  # leave-one-out self
        order = np.argsort(scores)[::-1]
        self.assertLess(int(order[0]), 20)  # top match is from class 0


class DBRDTests(unittest.TestCase):
    def test_lambda_zero_collapses_to_igpd(self):
        rng = np.random.default_rng(40)
        d, C, k = 16, 4, 4
        W = rng.standard_normal((C, d)).astype(np.float32)
        head = smet.ClassifierHead(W=W, b=np.zeros(C, dtype=np.float32))
        bank = rng.standard_normal((10, d)).astype(np.float32)
        bank_U = rng.standard_normal((10, d, k)).astype(np.float32)
        bank_s = np.abs(rng.standard_normal((10, k))).astype(np.float32)
        q_idx = 0
        q_mu = bank[q_idx]
        q_U = bank_U[q_idx]
        q_s = bank_s[q_idx]
        d_dbrd = smet.dbrd_similarity(
            q_mu, q_U, q_s, bank, bank_U, bank_s, head, lam=0.0,
        )
        d_igpd = smet.igpd_similarity(q_mu, bank, head)
        self.assertTrue(np.allclose(d_dbrd, d_igpd, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
import numpy as np
from pathlib import Path
from scipy.special import digamma, polygamma


class LDA:
    """
    LDA model with EM inference.

    Parameters:
      - K: number of topics
      - N: number of words per document
      - max_em_iters: maximum number of EM iterations
      - max_e_iters: maximum number of E-step iterations
      - tol_e: tolerance for E-step convergence
      - tol_alpha: tolerance for alpha convergence
      - seed: random seed
    """

    def __init__(
        self,
        K: int = 25,
        N: int = 200,
        max_em_iters: int = 100,
        max_e_iters: int = 100,
        tol_e: float = 1e-2,
        tol_alpha: float = 1e-4,
        seed: int = 0,
    ):
        self.K = K
        self.N = N
        self.max_em_iters = max_em_iters
        self.max_e_iters = max_e_iters
        self.tol_e = tol_e
        self.tol_alpha = tol_alpha
        self.rng = np.random.default_rng(seed)

        self.alpha = None  # [K]
        self.beta = None  # [K, V]
        self.gamma = None  # [M, K]
        self.phi = None  # [M, N, K]

    def _initialize(self, M: int, V: int):
        self.alpha = np.full(self.K, 0.1, dtype=np.float64)  # [K]

        beta = self.rng.uniform(0.0, 1.0, size=(self.K, V))  # [K, V]
        self.beta = beta / beta.sum(axis=1, keepdims=True)

        self.phi = np.full(
            (M, self.N, self.K), 1.0 / self.K, dtype=np.float64
        )  # [M, N, K]
        self.gamma = np.tile(self.alpha + self.N / self.K, (M, 1)).astype(
            np.float64
        )  # [M, K]

    def _estep(self, words: np.ndarray):
        """Variational inference: update phi and gamma until convergence."""
        M = words.shape[0]

        for t in range(self.max_e_iters):
            phi_prev = self.phi.copy()
            gamma_prev = self.gamma.copy()

            dig = digamma(self.gamma)  # [M, K]

            beta_lookup = np.zeros((M, self.N, self.K))  # [M, N, K]
            for k in range(self.K):
                beta_lookup[:, :, k] = self.beta[k, words]

            log_theta = dig[:, None, :]  # [M, 1, K]
            phi_unnorm = beta_lookup * np.exp(log_theta)  # [M, N, K]
            self.phi = phi_unnorm / (phi_unnorm.sum(axis=2, keepdims=True) + 1e-12)

            self.gamma = self.alpha[None, :] + self.phi.sum(axis=1)  # [M, K]

            delta = (
                np.linalg.norm(self.phi - phi_prev)
                + np.linalg.norm(self.gamma - gamma_prev)
            ) / (2.0 * M)

            if (t % 5) == 0:
                print(f"[E-step] iter={t:02d} avg-delta={delta:.3e}")
            if delta <= self.tol_e:
                print(f"[E-step] converged at iter={t} (avg-delta={delta:.3e})")
                break

    def _mstep_beta(self, words: np.ndarray):
        """M-step for beta: update topic-word distributions."""
        V = self.beta.shape[1]

        one_hot = np.eye(V, dtype=np.float64)[words]  # [M, N, V]
        kv = np.einsum("mnk,mnv->kv", self.phi, one_hot)  # [K, V]
        self.beta = kv / (kv.sum(axis=1, keepdims=True) + 1e-12)

    def _alpha_update(self):
        """Newton-Raphson step for alpha update."""
        M = self.gamma.shape[0]
        alpha = self.alpha
        sum_alpha = alpha.sum()

        psi_sum_alpha = digamma(sum_alpha)
        psi_alpha = digamma(alpha)  # [K]
        psi_gamma = digamma(self.gamma)  # [M, K]
        psi_sum_gamma = digamma(self.gamma.sum(axis=1))  # [M]

        g = M * (psi_sum_alpha - psi_alpha) + (psi_gamma - psi_sum_gamma[:, None]).sum(
            axis=0
        )  # [K]
        h = M * polygamma(1, alpha)  # [K]
        z = polygamma(1, sum_alpha)  # scalar

        c = np.sum(g / h) / ((1.0 / z) + np.sum(1.0 / h))
        raw_step = (g - c) / h
        alpha_new = alpha + raw_step
        return np.maximum(alpha_new, 1e-8)

    def _mstep_alpha(self):
        """M-step for alpha: update Dirichlet prior with damping."""
        for t in range(100):
            alpha_prev = self.alpha.copy()
            alpha_prop = self._alpha_update()

            eta = 0.5  # damping factor
            self.alpha = alpha_prev + eta * (alpha_prop - alpha_prev)

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if (t % 5) == 0:
                print(
                    f"[M-step] iter={t:02d} ||Δα||={diff:.3e} α_mean={self.alpha.mean():.3g}"
                )
            if diff <= self.tol_alpha:
                print(f"[M-step] α converged at iter={t} (||Δα||={diff:.3e})")
                break

    def fit(self, words: np.ndarray, V: int):
        """Run variational EM algorithm."""
        M, N = words.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"
        self._initialize(M, V)

        for it in range(self.max_em_iters):
            print(f"\n=== EM iter {it} ===")
            self._estep(words)
            self._mstep_beta(words)
            self._mstep_alpha()

        print("\n[Training complete]")

    def save_params(self, out_dir: str):
        """Save learned parameters to disk."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(out_dir) / "alpha.npy", self.alpha)
        np.save(Path(out_dir) / "beta.npy", self.beta)
        np.save(Path(out_dir) / "gamma.npy", self.gamma)

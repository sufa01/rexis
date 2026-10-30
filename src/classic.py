"""Non-transformer recommenders: popularity, ItemKNN, EASE, ALS, BPR, SLIM, SeqEASE.

Every model is fitted on an events frame and returns a dense score matrix
(users x catalogue items) for the requested users; `top_k` turns it into the
shared candidate format (user_id, item_id, score, rank).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp


class Dataset:
    """Index mapping and the user-item matrix built from training events."""

    def __init__(self, events: pd.DataFrame, users, items, positive_threshold: float | None = None,
                 weight: str = "binary"):
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        self.uidx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.iidx = pd.Series(np.arange(len(self.items)), index=self.items)
        ev = events[events["user_id"].isin(self.uidx.index) & events["item_id"].isin(self.iidx.index)]
        rows = self.uidx[ev["user_id"]].to_numpy()
        cols = self.iidx[ev["item_id"]].to_numpy()
        shape = (len(self.users), len(self.items))
        # `seen` always holds every training event: those are excluded from recommendations.
        self.seen = sp.csr_matrix((np.ones(len(ev), dtype=np.float32), (rows, cols)), shape=shape)
        self.seen.data[:] = 1.0
        if weight == "rating":
            vals = ev["rating"].to_numpy(dtype=np.float32)
        else:
            vals = np.ones(len(ev), dtype=np.float32)
        if positive_threshold is not None:
            vals = vals * (ev["rating"].to_numpy() >= positive_threshold)
        X = sp.csr_matrix((vals, (rows, cols)), shape=shape)
        X.eliminate_zeros()
        self.X = X
        # Training events with timestamps; used by SeqEASE for event order.
        self.events = ev


# ---------------------------------------------------------------- models

class Popular:
    def fit(self, ds: Dataset):
        self.pop = np.asarray(ds.X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(u), 1))


class ItemKNN:
    """Cosine item-item similarity, top-k neighbours per item, optional shrinkage."""

    def __init__(self, k: int = 200, shrink: float = 10.0, alpha: float = 0.5):
        self.k, self.shrink, self.alpha = k, shrink, alpha

    def fit(self, ds: Dataset):
        X = ds.X.tocsc().astype(np.float32)
        norms = np.asarray(X.power(2).sum(axis=0)).ravel()
        G = (X.T @ X).toarray()
        # Asymmetric cosine: alpha=0.5 is the usual cosine.
        denom = np.outer(norms ** self.alpha, norms ** (1 - self.alpha)) + self.shrink + 1e-9
        S = G / denom
        np.fill_diagonal(S, 0.0)
        if self.k and self.k < S.shape[0]:
            thr = -np.partition(-S, self.k - 1, axis=0)[self.k - 1]
            S[S < thr] = 0.0
        self.S = S.astype(np.float32)
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        return np.asarray(ds.X[u] @ self.S)


class EASE:
    def __init__(self, reg: float = 500.0):
        self.reg = reg

    def fit(self, ds: Dataset):
        X = ds.X.astype(np.float64)
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        return np.asarray(ds.X[u] @ self.B)


class ImplicitMF:
    """ALS or BPR from the `implicit` library."""

    def __init__(self, algo: str = "als", factors: int = 128, reg: float = 0.05,
                 iterations: int = 30, alpha: float = 10.0, lr: float = 0.05, seed: int = 42):
        self.algo, self.factors, self.reg = algo, factors, reg
        self.iterations, self.alpha, self.lr, self.seed = iterations, alpha, lr, seed

    def fit(self, ds: Dataset):
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking
        if self.algo == "als":
            m = AlternatingLeastSquares(factors=self.factors, regularization=self.reg,
                                        iterations=self.iterations, alpha=self.alpha,
                                        random_state=self.seed, num_threads=1)
        else:
            m = BayesianPersonalizedRanking(factors=self.factors, regularization=self.reg,
                                            learning_rate=self.lr, iterations=self.iterations,
                                            random_state=self.seed, num_threads=1)
        m.fit(ds.X.tocsr().astype(np.float32), show_progress=False)
        self.U = np.asarray(m.user_factors)
        self.V = np.asarray(m.item_factors)
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        return self.U[u] @ self.V.T


class SLIM:
    """SLIM with ElasticNet, restricted to the top-`k` co-occurring items per target."""

    def __init__(self, l1: float = 1e-3, l2: float = 1e-2, k: int = 200):
        self.l1, self.l2, self.k = l1, l2, k

    def fit(self, ds: Dataset):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import ElasticNet
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X = ds.X.tocsc().astype(np.float64)
        n_users, n_items = X.shape
        G = (X.T @ X).toarray()
        np.fill_diagonal(G, 0.0)
        alpha = self.l1 + self.l2
        model = ElasticNet(alpha=alpha / n_users, l1_ratio=self.l1 / alpha, positive=True,
                           fit_intercept=False, copy_X=False, max_iter=200, tol=1e-4, selection="cyclic")
        W = np.zeros((n_items, n_items), dtype=np.float32)
        for j in range(n_items):
            if G[:, j].max() == 0:
                continue
            nb = np.argpartition(-G[:, j], self.k)[: self.k]
            nb = nb[G[nb, j] > 0]
            y = X[:, j].toarray().ravel()
            model.fit(X[:, nb], y)
            W[nb, j] = model.coef_
        self.W = W
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        return np.asarray(ds.X[u] @ self.W)


class SeqEASE:
    """
    Sequential blend for a test made of each user's latest events.

    - EASE scored with a recency-weighted user vector: weight exp(-k / h_ease),
      where k is the position counted from the user's last event;
    - forward transitions: T[i, j] counts how often j follows i within `horizon`
      steps (weight decay^(d-1)), divided by the popularity of i; the user's last
      `last_n` items (weight exp(-k / h_last)) are propagated through T.
    Both parts are z-normalised per user and summed with weight `beta` on T.
    """

    def __init__(self, reg: float = 400.0, h_ease: float = 10.0, horizon: int = 10,
                 decay: float = 0.8, last_n: int = 10, h_last: float = 3.0, beta: float = 1.0):
        self.ease = EASE(reg)
        self.h_ease, self.horizon, self.decay = h_ease, horizon, decay
        self.last_n, self.h_last, self.beta = last_n, h_last, beta

    def fit(self, ds: Dataset):
        self.ease.fit(ds)
        ev = ds.events.sort_values(["user_id", "timestamp"])
        u = ds.uidx[ev["user_id"]].to_numpy()
        it = ds.iidx[ev["item_id"]].to_numpy()
        k = ev.groupby("user_id").cumcount(ascending=False).to_numpy()
        shape, n = ds.X.shape, len(ds.items)
        rows, cols, vals = [], [], []
        for d in range(1, self.horizon + 1):
            same = u[d:] == u[:-d]
            rows.append(it[:-d][same]); cols.append(it[d:][same])
            vals.append(np.full(same.sum(), self.decay ** (d - 1), dtype=np.float32))
        T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        pop = np.asarray(ds.X.sum(axis=0)).ravel() + 1.0
        self.T = (sp.diags(1.0 / pop) @ T).tocsr()
        self.X_recent = sp.csr_matrix((np.exp(-k / self.h_ease).astype(np.float32), (u, it)), shape=shape)
        w_last = np.where(k < self.last_n, np.exp(-k / self.h_last), 0.0).astype(np.float32)
        self.X_last = sp.csr_matrix((w_last, (u, it)), shape=shape)
        self.X_last.eliminate_zeros()
        return self

    def score(self, ds: Dataset, u: np.ndarray) -> np.ndarray:
        base = np.asarray(self.X_recent[u] @ self.ease.B)
        trans = np.asarray((self.X_last[u] @ self.T).todense())
        return _rownorm(base) + self.beta * _rownorm(trans)


def _rownorm(S: np.ndarray) -> np.ndarray:
    S = S - S.mean(axis=1, keepdims=True)
    return S / (S.std(axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- helpers

def top_k(model, ds: Dataset, users, k: int = 100, batch: int = 1024) -> pd.DataFrame:
    """Top-k unseen items per user in the shared format: user_id, item_id, score, rank."""
    u_all = ds.uidx[np.asarray(users)].to_numpy()
    out = []
    for s in range(0, len(u_all), batch):
        u = u_all[s:s + batch]
        S = model.score(ds, u).astype(np.float32)
        S[ds.seen[u].nonzero()] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        sc = np.take_along_axis(S, idx, axis=1)
        o = np.argsort(-sc, axis=1, kind="stable")
        idx, sc = np.take_along_axis(idx, o, axis=1), np.take_along_axis(sc, o, axis=1)
        out.append(pd.DataFrame({
            "user_id": np.repeat(ds.users[u], k),
            "item_id": ds.items[idx.ravel()],
            "score": sc.ravel(),
            "rank": np.tile(np.arange(1, k + 1), len(u)),
        }))
    return pd.concat(out, ignore_index=True)


def recall(cands: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> float:
    """Mean per-user recall@k: |top-k ∩ held-out| / |held-out|."""
    top = cands[cands["rank"] <= k][["user_id", "item_id"]]
    hits = top.merge(truth[["user_id", "item_id"]], on=["user_id", "item_id"]).groupby("user_id").size()
    n_rel = truth.groupby("user_id").size()
    return float((hits.reindex(n_rel.index, fill_value=0) / n_rel).mean())

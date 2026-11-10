"""
Scalable edge features for big graphs.

The version 3 "affected-set" features need, for every candidate edge (u, v), a pass over ALL n nodes.
Here they are ESTIMATED from R random landmark nodes (counts scaled by n / R), so the cost per edge
is O(R) instead of O(n). All other edge features (version 2) are exact.
Candidate edges are processed in blocks of u, in the canonical order of src.fastgain.all_gains
(u ascending, v > u ascending, existing edges skipped), so nothing of size (#candidates x features)
is ever held in memory.
Feature columns ("xs", 24): 9 version-2 edge features, 7 affected-set features, 8 level histograms.
"""
import numpy as np, networkx as nx, numba as nb
from scipy.sparse.csgraph import shortest_path

LEVELS = 4


@nb.njit(parallel=True, cache=True)
def _sampled_hist(Ds, eu, ev, L):
    R = Ds.shape[0]
    E = eu.shape[0]
    out = np.zeros((E, 5 + 2 * LEVELS))           # nS nT sumA sumB ub | A1..A4+ | B1..B4+
    for e in nb.prange(E):
        u = eu[e]; v = ev[e]
        ca = np.zeros(L + 1); cb = np.zeros(L + 1)
        for l in range(R):
            a = Ds[l, v] - Ds[l, u] - 1
            if a >= 1:
                ca[a] += 1
            elif a <= -3:
                cb[-a - 2] += 1
        nS = 0.0; nT = 0.0; sA = 0.0; sB = 0.0
        for i in range(1, L + 1):
            nS += ca[i]; nT += cb[i]; sA += i * ca[i]; sB += i * cb[i]
            k = min(i, LEVELS) - 1
            out[e, 5 + k] += ca[i]
            out[e, 5 + LEVELS + k] += cb[i]
        ub = 0.0
        for i in range(1, L + 1):
            if ca[i] == 0:
                continue
            for j in range(1, L + 1):
                if cb[j] != 0:
                    ub += ca[i] * cb[j] * min(i, j)
        out[e, 0] = nS; out[e, 1] = nT; out[e, 2] = sA; out[e, 3] = sB; out[e, 4] = ub
    return out


class GraphPrep:
    def __init__(self, G, R=128, seed=0):
        n = G.number_of_nodes()
        self.n = n
        Asp = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr").astype(np.float32)
        self.D = shortest_path(Asp, unweighted=True, directed=False).astype(np.int16)
        self.L = int(self.D.max())
        self.A = Asp.toarray() > 0
        self.deg = np.asarray(Asp.sum(1)).ravel()
        self.clo = (n - 1) / self.D.sum(1, dtype=np.float64)
        A2 = (Asp @ Asp).toarray()
        w = 1.0 / np.log(np.maximum(self.deg, 2))
        AW = (Asp.multiply(w[None, :]) @ Asp).toarray()
        self.A2, self.AW = A2.astype(np.float32), AW.astype(np.float32)
        rng = np.random.default_rng(seed)
        self.R = min(R, n)
        self.land = np.sort(rng.choice(n, self.R, replace=False))
        self.Ds = np.ascontiguousarray(self.D[self.land])
        self.f = n / self.R
        self.mu = self.sd = None

    def raw(self, eu, ev):
        d = self.deg; c = self.clo
        cn = self.A2[eu, ev]
        xe = np.stack([self.D[eu, ev].astype(np.float64), cn, cn / np.maximum(d[eu] + d[ev] - cn, 1),
                       self.AW[eu, ev], np.abs(c[eu] - c[ev]), np.minimum(c[eu], c[ev]), np.maximum(c[eu], c[ev]),
                       np.log1p(np.minimum(d[eu], d[ev])), np.log1p(np.maximum(d[eu], d[ev]))], 1)
        h = _sampled_hist(self.Ds, eu.astype(np.int64), ev.astype(np.int64), self.L)
        f = self.f
        nS, nT, sA, sB, ub = h[:, 0] * f, h[:, 1] * f, h[:, 2] * f, h[:, 3] * f, h[:, 4] * f * f
        xp = np.stack([np.minimum(nS, nT), np.maximum(nS, nT), nS * nT, np.minimum(sA, sB), np.maximum(sA, sB),
                       sA + sB, ub], 1)
        HA, HB = h[:, 5:5 + LEVELS] * f, h[:, 5 + LEVELS:] * f
        xh = np.concatenate([np.minimum(HA, HB), np.maximum(HA, HB)], 1)
        return np.concatenate([xe, np.log1p(xp), np.log1p(xh)], 1)

    def fit_stats(self, max_pairs=300_000, seed=1):
        """Per-graph standardization stats from (a sample of) the candidate edges."""
        n = self.n
        if n * (n - 1) // 2 <= max_pairs * 1.5:
            eu, ev = self.candidates(0, n)
        else:
            rng = np.random.default_rng(seed)
            eu = rng.integers(0, n, 2 * max_pairs); ev = rng.integers(0, n, 2 * max_pairs)
            ok = (eu != ev) & ~self.A[eu, ev]
            eu, ev = np.minimum(eu, ev)[ok][:max_pairs], np.maximum(eu, ev)[ok][:max_pairs]
        X = self.raw(eu, ev)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0

    def candidates(self, u0, u1):
        eu, ev = [], []
        for u in range(u0, u1):
            vs = np.nonzero(~self.A[u, u + 1:])[0] + u + 1
            eu.append(np.full(len(vs), u)); ev.append(vs)
        return np.concatenate(eu).astype(np.int64), np.concatenate(ev).astype(np.int64)

    def features(self, eu, ev):
        if self.mu is None:
            self.fit_stats()
        return ((self.raw(eu, ev) - self.mu) / self.sd).astype(np.float32)

    def blocks(self, max_edges=1_000_000):
        """Yield (eu, ev) blocks of candidates in canonical order."""
        u0 = 0
        n = self.n
        while u0 < n:
            u1, cnt = u0, 0
            while u1 < n and cnt < max_edges:
                cnt += (n - u1 - 1); u1 += 1
            yield self.candidates(u0, u1)
            u0 = u1

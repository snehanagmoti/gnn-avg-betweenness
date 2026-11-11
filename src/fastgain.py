"""
Fast exact gain of adding one edge (affected sets) and the cheap affected-set clues.

Notation (unweighted, connected, undirected graph; D = all-pairs distance matrix):
    total betweenness = sum over unordered pairs {s,t} of (d(s,t) - 1)
so the reduction from adding (u, v) is the total shortening of distances.

Key fact (affected sets, cf. Parotsidis, Pitoura & Tsaparas, SDM 2015):
    a pair {s,t} can only get shorter through the new edge in the direction s -> u -> v -> t if
        a_s = d(s,v) - d(s,u) - 1 >= 1   (s is at least 2 hops closer to u than to v)   s in S_u
        b_t = d(t,u) - d(t,v) - 1 >= 1   (t is at least 2 hops closer to v than to u)   t in T_v
    and then its new distance is d(s,u) + 1 + d(v,t). S_u and T_v are disjoint, and the other
    direction gives the same pairs, so every improving pair is counted exactly once:
        gain(u,v) = sum_{s in S_u, t in T_v} max(0, d(s,t) - d(s,u) - 1 - d(v,t))
    Upper bound per pair: d(s,t) <= d(s,v) + d(v,t) and d(s,t) <= d(s,u) + d(u,t) give
        max(0, ...) <= min(a_s, b_t)
    so  UB(u,v) = sum_{s in S_u, t in T_v} min(a_s, b_t)  >= gain(u,v),
    computable from the two level histograms of a and b in O(n + L^2) per edge.

reduction in average betweenness (normalized like NetworkX) = gain / n * 2 / ((n-1)(n-2)).
"""
import numpy as np


def scale_factor(n, normalize=True):
    return (2.0 / ((n - 1) * (n - 2)) if normalize else 1.0) / n


def level_histograms(D, edges, L=None):
    """For every candidate edge: counts of a-levels (S side) and b-levels (T side).
    Returns cntA, cntB of shape (E, L+1) with column i = number of nodes at level i (i >= 1)."""
    u, v = edges[:, 0], edges[:, 1]
    A = D[v] - D[u] - 1          # (E, n): a_s for every s
    B = -A - 2                   # b_t = d(t,u) - d(t,v) - 1 = -(a_t) - 2
    if L is None:
        L = int(D.max())
    cntA = np.stack([(A == i).sum(1) for i in range(L + 1)], 1)
    cntB = np.stack([(B == i).sum(1) for i in range(L + 1)], 1)
    cntA[:, 0] = 0
    cntB[:, 0] = 0
    return cntA, cntB


def pair_features(D, edges):
    """Cheap O(n)-per-edge features from the affected sets (all symmetric in u, v)."""
    cntA, cntB = level_histograms(D, edges)
    L = cntA.shape[1]
    lv = np.arange(L)
    nS, nT = cntA.sum(1), cntB.sum(1)
    sumA, sumB = cntA @ lv, cntB @ lv
    M = np.minimum.outer(lv, lv).astype(np.float64)
    ub = np.einsum("ei,ij,ej->e", cntA.astype(np.float64), M, cntB.astype(np.float64))
    # "one-sided" exact gains: pairs (s, v) for s in S_u and (u, t) for t in T_v
    one_sided = sumA + sumB
    return {
        "aff_min": np.minimum(nS, nT), "aff_max": np.maximum(nS, nT),
        "aff_prod": nS * nT, "lvl_sum_min": np.minimum(sumA, sumB),
        "lvl_sum_max": np.maximum(sumA, sumB), "one_sided": one_sided, "ub": ub,
    }


def exact_gain(D, u, v):
    """Exact gain of edge (u, v), touching only the affected block S_u x T_v."""
    a = D[:, v] - D[:, u] - 1
    S = np.nonzero(a >= 1)[0]
    T = np.nonzero(a <= -3)[0]           # b_t = -a_t - 2 >= 1
    if len(S) == 0 or len(T) == 0:
        return 0.0
    new = D[S, u][:, None] + 1 + D[v, T][None, :]
    return float(np.maximum(0, D[np.ix_(S, T)] - new).sum())


def exact_gains(D, edges):
    return np.array([exact_gain(D, int(u), int(v)) for u, v in edges])


# ---------------------------------------------------------------------------------------------
# Compiled versions (numba). Same maths, C speed; parallel over candidate edges.
try:
    import numba as _nb

    @_nb.njit(parallel=True, cache=True)
    def _gains_nb(D, eu, ev):
        n = D.shape[0]
        E = eu.shape[0]
        out = np.zeros(E)
        for e in _nb.prange(E):
            u = eu[e]
            v = ev[e]
            S = np.empty(n, np.int64)
            T = np.empty(n, np.int64)
            ns = 0
            nt = 0
            for x in range(n):
                a = D[x, v] - D[x, u] - 1
                if a >= 1:
                    S[ns] = x
                    ns += 1
                elif a <= -3:
                    T[nt] = x
                    nt += 1
            g = 0.0
            for i in range(ns):
                s = S[i]
                dsu = D[s, u] + 1
                for j in range(nt):
                    t = T[j]
                    d = D[s, t] - dsu - D[v, t]
                    if d > 0:
                        g += d
            out[e] = g
        return out

    def exact_gains_fast(D, edges):
        D = np.ascontiguousarray(D, dtype=np.int64)
        return _gains_nb(D, edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64))

except ImportError:            # numba not installed: fall back to the numpy version
    exact_gains_fast = exact_gains

"""
Efficient exact gain of adding one edge, cheap upper bounds, and certified exact top-k.

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


def upper_bounds(D, edges):
    """UB(u,v) = sum_{i,j} cntA[i] * cntB[j] * min(i, j)  (>= exact gain)."""
    cntA, cntB = level_histograms(D, edges)
    L = cntA.shape[1]
    M = np.minimum.outer(np.arange(L), np.arange(L)).astype(np.float64)
    return np.einsum("ei,ij,ej->e", cntA.astype(np.float64), M, cntB.astype(np.float64))


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


def certified_topk(D, edges, k, order=None, ub=None):
    """Exact top-k (all ties at the k-th value included) with a certificate.
    Evaluates edges in `order` (default: by upper bound, best first) and stops when no remaining
    edge's upper bound can reach the current k-th best exact gain.
    Returns (idx_sorted_by_gain, gains, n_evaluated)."""
    if ub is None:
        ub = upper_bounds(D, edges)
    if order is None:
        order = np.argsort(-ub, kind="stable")
    # for a learned order, the stopping test needs the best UB among edges not yet evaluated
    import heapq
    rem_max = np.maximum.accumulate(ub[order][::-1])[::-1]
    heap = []                       # the k best exact gains so far (min-heap)
    gains = {}
    kth = -np.inf
    n_eval = 0
    for pos, i in enumerate(order):
        if pos >= k and rem_max[pos] < kth:
            break
        g = exact_gain(D, int(edges[i, 0]), int(edges[i, 1]))
        gains[i] = g
        n_eval += 1
        if len(heap) < k:
            heapq.heappush(heap, g)
        elif g > heap[0]:
            heapq.heapreplace(heap, g)
        if len(heap) == k:
            kth = heap[0]
    idx = np.array(list(gains.keys()))
    val = np.array([gains[i] for i in idx])
    o = np.argsort(-val, kind="stable")
    return idx[o], val[o], n_eval


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

    @_nb.njit(parallel=True, cache=True)
    def _ub_nb(D, eu, ev, L):
        n = D.shape[0]
        E = eu.shape[0]
        out = np.zeros(E)
        for e in _nb.prange(E):
            u = eu[e]
            v = ev[e]
            ca = np.zeros(L + 1)
            cb = np.zeros(L + 1)
            for x in range(n):
                a = D[x, v] - D[x, u] - 1
                if a >= 1:
                    ca[a] += 1
                elif a <= -3:
                    cb[-a - 2] += 1
            s = 0.0
            for i in range(1, L + 1):
                if ca[i] == 0:
                    continue
                for j in range(1, L + 1):
                    s += ca[i] * cb[j] * min(i, j)
            out[e] = s
        return out

    def exact_gains_fast(D, edges):
        D = np.ascontiguousarray(D, dtype=np.int64)
        return _gains_nb(D, edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64))

    def upper_bounds_fast(D, edges):
        D = np.ascontiguousarray(D, dtype=np.int64)
        return _ub_nb(D, edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64), int(D.max()))
except ImportError:            # numba not installed: fall back to numpy versions
    exact_gains_fast = exact_gains
    upper_bounds_fast = upper_bounds

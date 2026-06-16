"""Feature grouping strategies for CovNCG blocks.

Three options:
  - group_greedy:        highest-pairwise-correlation greedy (original "covariance" rule)
  - group_hierarchical:  Ward linkage on |1 - corr|
  - group_random:        uniform-random grouping (ablation control)

NOTE: multi-seed ablation across 5 medical datasets shows that the greedy
covariance rule is statistically indistinguishable from random grouping. The
accuracy gain of CovNCG arises from the non-commuting architecture + projected
kernel + bandwidth tuning, NOT from the grouping rule. See docs/ + FINAL_REPORT.
"""
from __future__ import annotations

import numpy as np


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X, rowvar=False)
    return np.abs(corr)


def group_greedy(X: np.ndarray, group_size: int = 3) -> list[list[int]]:
    """Greedy: highest-correlation pair, then add feature most correlated with pair."""
    corr = correlation_matrix(X)
    np.fill_diagonal(corr, 0.0)
    n_features = X.shape[1]
    remaining = set(range(n_features))
    groups: list[list[int]] = []

    while len(remaining) >= 2:
        rem = list(remaining)
        sub = corr[np.ix_(rem, rem)]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        a, b = rem[i], rem[j]
        group = [a, b]
        remaining -= {a, b}

        while len(group) < group_size and remaining:
            rem = list(remaining)
            scores = corr[rem][:, group].mean(axis=1)
            k = rem[int(np.argmax(scores))]
            group.append(k)
            remaining.discard(k)

        groups.append(group)

    if remaining:
        groups.append([remaining.pop()])

    return groups


def group_hierarchical(X: np.ndarray, n_groups: int) -> list[list[int]]:
    """Ward linkage on (1 - |corr|) distance."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = correlation_matrix(X)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(Z, t=n_groups, criterion="maxclust")
    return [np.where(labels == i + 1)[0].tolist() for i in range(n_groups)]


def group_random(X: np.ndarray, group_size: int = 3, seed: int = 0) -> list[list[int]]:
    """Random grouping control (ablation 1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[1]).tolist()
    return [perm[i:i + group_size] for i in range(0, len(perm), group_size)]


def qubit_layout(groups: list[list[int]]) -> tuple[int, list[tuple[list[int], list[int]]]]:
    """Map groups → (n_qubits, [(qubit_indices, feature_indices), ...]).

    Triple/pair groups use 2 qubits. Singleton groups use 1 qubit.
    """
    n_qubits = 0
    layout: list[tuple[list[int], list[int]]] = []
    for g in groups:
        if len(g) >= 2:
            layout.append(([n_qubits, n_qubits + 1], g[:3]))
            n_qubits += 2
        else:
            layout.append(([n_qubits], g))
            n_qubits += 1
    return n_qubits, layout

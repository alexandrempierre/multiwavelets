__all__ = [
    's_callable', 'mu_callable', 'sigma_callable',
    's_array', 'mu_list', 'sigma_list',
]
__author__ = 'Alexandre Pierre <alexandrempierre [at] gmail [dot] com>'


from typing import Callable, List
#
import numpy as np


def s_callable(k: int, start_idx: int = 0) -> Callable[[int], int]:
    def s(i: int) -> int:
        return (i - start_idx)*2*k
    return s


def s_array(n: int, k: int, start_idx: int = 0) -> np.ndarray:
    '''valores de s indexados por 0'''
    return np.array(
        [
            (i - start_idx)*2*k
            for i in range(start_idx, n // (2*k) + start_idx)
        ],
        dtype=int
    )


def mu_callable(
    x: np.ndarray,
    k: int,
    start_idx: int = 0,
) -> Callable[[int, int], float]:
    def mu(j: int, i: int) -> float:
        idx = (
            (i - start_idx)*k*2**(j + 1 - start_idx),
            (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
        )
        return (x[idx[0]] + x[idx[1]]) / 2
    return mu


def mu_list(
    x: np.ndarray,
    k: int,
    start_idx: int = 0,
) -> List[List[float]]:
    n = len(x)
    l = int(np.log2(n // k))
    mu = []
    for row, j in enumerate(range(start_idx, l + start_idx)):
        mu.append([])
        for i in range(start_idx, n // (k*2**(j + 1 - start_idx)) + start_idx):
            idx = (
                (i - start_idx)*k*2**(j + 1 - start_idx),
                (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
            )
            mu[row].append((x[idx[0]] + x[idx[1]]) / 2)
    return mu


def sigma_callable(
    x: np.ndarray,
    k: int,
    start_idx: int = 0,
) -> Callable[[int, int], float]:
    def sigma(j: int, i: int) -> float:
        idx = (
            (i - start_idx)*k*2**(j + 1 - start_idx),
            (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
        )
        return (x[idx[1]] - x[idx[0]]) / 2
    return sigma


def sigma_list(
    x: np.ndarray,
    k: int,
    start_idx: int = 0,
) -> np.ndarray:
    n = len(x)
    l = int(np.log2(n // k))
    sigma = []
    for row, j in enumerate(range(start_idx, l + start_idx)):
        sigma.append([])
        for i in range(start_idx, n // (k*2**(j + 1 - start_idx)) + start_idx):
            idx = (
                (i - start_idx)*k*2**(j + 1 - start_idx),
                (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
            )
            sigma[row].append((x[idx[1]] - x[idx[0]]) / 2)
    return sigma



'''módulo step_3
'''


__all__ = []
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from collections.abc import Callable, Iterator
#
import numpy as np
import numpy.typing as npt


# def singularity_handling

def operator_matrix(
    xs: npt.NDArray[float],
    K: Callable[[float, float], float],
) -> npt.NDArray[float]:
    # pylint: disable=invalid-name
    '''matriz do operador usando método de quadratura do trapézio'''
    n = xs.shape
    return np.array([
        [
            0 if i == j else 1 / (n - 1) * K(x_i, x_j)
            for j, x_j in enumerate(xs)
        ]
        for i, x_i in enumerate(xs)
    ])


def operator_matrix_extracts(
    T: npt.NDArray[float],
    k: int
) -> list[npt.NDArray[float]]:
    # pylint: disable=invalid-name
    '''extrair submatrizes'''
    n = T.shape[0]
    extracts = {}
    for start_idx, sub_size, jmp_odd in extract_params(n, k):
        extracts[sub_size] = (
            extracts.get(sub_size, [])
            + [extract_submatrices(T, start_idx, sub_size, jmp_odd)]
        )
    return [sum(Vs) for Vs in extracts.values()]


def extract_params(n: int, k: int) -> Iterator[tuple[int, int, bool]]:
    # pylint: disable=invalid-name
    '''gerar parâmetros para extrair submatrizes'''
    start_idx, sub_size = 0, k
    while start_idx < n:
        jmp_odd = is_zero_or_power_of_2(start_idx // k)
        yield start_idx, sub_size, jmp_odd
        start_idx += sub_size
        if jmp_odd:
            sub_size *= 2


def extract_submatrices(
    T: npt.NDArray[float],
    start_idx: int,
    sub_size: int,
    jmp_odd: bool,
) ->  npt.NDArray[float]:
    # pylint: disable=invalid-name
    '''extrai submatrizees'''
    n = T.shape[0]
    X = np.zeros_like(T)
    for step, idx in enumerate(range(start_idx, n, sub_size)):
        end = idx + sub_size
        idx_orth = step * sub_size
        end_orth = idx_orth + sub_size
        if step % 2 == 0 or not jmp_odd:
            X[idx:end, idx_orth:end_orth] = T[idx:end, idx_orth:end_orth]
            X[idx_orth:end_orth, idx:end] = T[idx_orth:end_orth, idx:end]
    return X


def is_zero_or_power_of_2(value: int) -> bool:
    # pylint: disable=invalid-name
    # pylint: disable=missing-function-docstring
    return value & (value - 1) == 0

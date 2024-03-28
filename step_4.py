# ruff: noqa: E741
'''módulo step_4
'''


__all__ = [
    'operator_matrix', 'extract_submatrices', 'is_zero_or_power_of_2',
    'visualize_blocks', 'remove_singularity'
]
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import itertools as it
import math

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


NDArrayFloat = npt.NDArray[np.float64]
Real2DFunction = Callable[[float, float], float]


def remove_singularity(
    f: Real2DFunction,
    x: float,
    y: float,
    fallback: float = 0,
) -> float:
    try:
        z = f(x, y)
    except ZeroDivisionError:
        return fallback
    if not math.isfinite(z):
        return fallback
    return z


def operator_matrix(
    xs: NDArrayFloat,
    kernel_fn: Real2DFunction,
) -> NDArrayFloat:
    # pylint: disable=invalid-name
    '''matriz do operador usando método de quadratura do trapézio'''
    n = xs.shape[0]
    return np.array([
        [
            remove_singularity(kernel_fn, x_i, x_j, fallback=0) / (n - 1)
            for j, x_j in enumerate(xs)
        ]
        for i, x_i in enumerate(xs)
    ])


def extract_symmetric_block(
    M: NDArrayFloat,
    index: int,
    size: int,
) -> dict[tuple[int, int], int]:
    '''Apesar do nome indicar extração de blocos, essa função apenas usa a
tupla (linha, coluna) para apontar o tamanho da submatriz [quadrada]
'''
    n, _ = M.shape
    if index + size > n:
        raise IndexError(f'{index} + {size} > {n}')
    blocks = dict()
    for j in range(index, n, size):
        i = j - index
        blocks[i, j] = size
        if i == j:
            continue
        blocks[j, i] = size
    return blocks


def extract_submatrices(
    T: NDArrayFloat,
    k: int,
    l: int | None = None,
) -> tuple[list[NDArrayFloat], set[NDArrayFloat]]:
    '''_'''
    n = T.shape[0]
    if l is None:
        l = int(np.log2(n // k))
    blocks = dict()
    idx = 4*k
    for block_size in (2**i * k for i in range(2, l)):
        blocks = dict(
            it.chain(
                extract_symmetric_block(T, idx, block_size).items(),
                blocks.items()
            )
        )
        idx += block_size
    blocks |= extract_symmetric_block(T, 0, 2*k)
    blocks |= extract_symmetric_block(T, 2*k, 2*k)
    for ((row, col), block_size) in list(blocks.items()):
        half_size = block_size // 2
        half_row = row + half_size
        half_col = col + half_size
        if blocks.get((row, col), float('inf')) > half_size:
            blocks[row, col] = half_size
        if blocks.get((half_row, col), float('inf')) > half_size:
            blocks[half_row, col] = half_size
        if blocks.get((row, half_col), float('inf')) > half_size:
            blocks[row, half_col] = half_size
        if blocks.get((half_row, half_col), float('inf')) > half_size:
            blocks[half_row, half_col] = half_size
    return blocks


def visualize_blocks(
    T: npt.NDArray,
    blocks: dict[tuple[int, int], int],
) -> dict[int, npt.NDArray]:
    n = T.shape[0]
    Ts = {sub_n: np.zeros((n, n)) for sub_n in set(blocks.values())}
    for ((row, col), size) in blocks.items():
        Ts[size][row:row + size, col:col + size] = \
            T[row:row + size, col:col + size]
    return Ts


def interpolate_matrix(M: npt.NDArray, k: int) -> npt.NDArray:
    n = M.shape[0]
    interpolated = M[n//k - k::n//k, n//k - k::n//k]
    return interpolated


def is_zero_or_power_of_2(value: int) -> bool:
    # pylint: disable=invalid-name, missing-function-docstring
    return value & (value - 1) == 0

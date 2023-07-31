'''módulo step_4
'''


__all__ = [
    'operator_matrix', 'extract_submatrices', 'is_zero_or_power_of_2',
    'Submatrix',
]
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import functools as ft
import operator as op
#
from collections.abc import Callable
from dataclasses import dataclass
#
import numpy as np
import numpy.typing as npt


NDArrayFloat = npt.NDArray[np.float64]


@dataclass
class Submatrix:
    '''submatriz de operador a ser aproximada por uma matriz de posto k'''
    subarray: NDArrayFloat
    row_start: int
    col_start: int

    def __eq__(self, s: 'Submatrix') -> bool:
        # pylint: disable=invalid-name
        return (
            self.row_start == s.row_start
            and self.col_start == s.col_start
            and np.all(self.subarray == s.subarray)
        )

    def __hash__(self) -> int:
        astuple = tuple(
            (row, ) if isinstance(row, np.number) else tuple(row)
            for row in self.subarray
        )
        return hash(
            (astuple, self.row_start, self.col_start)
        )


def operator_matrix(
    xs: NDArrayFloat,
    kernel_fn: Callable[[float, float], float],
) -> NDArrayFloat:
    # pylint: disable=invalid-name
    '''matriz do operador usando método de quadratura do trapézio'''
    n = xs.shape
    return np.array([
        [
            0 if i == j else 1 / (n - 1) * kernel_fn(x_i, x_j)
            for j, x_j in enumerate(xs)
        ]
        for i, x_i in enumerate(xs)
    ])


def extract_submatrices(
    T: NDArrayFloat, k: int,
    r0: int = 0, c0: int = 0,
    size: int | None = None,
) -> set[Submatrix]:
    # pylint: disable=invalid-name
    '''extrai submatrizes'''
    if size is None:
        size = T.shape[0]
    if size == k:
        return {Submatrix(T[r0:r0 + k, c0:c0 + k], r0, c0)}
    if size == 4*k:
        return {
            Submatrix(
                T[
                    r0 + rows_step:r0 + rows_step + k,
                    c0 + cols_step:c0 + cols_step + k
                ],
                r0 + rows_step, c0 + cols_step
            )
            for rows_step in range(0, 4*k, k)
            for cols_step in range(0, 4*k, k)
        }
    s = (0, size//4, size//2, 3*size//4, size)
    return ft.reduce(
        op.or_, [
            extract_submatrices(T, k, r0, c0, size // 2),
            extract_submatrices(T, k, r0 + s[1], c0 + s[1], size // 2),
            extract_submatrices(T, k, r0 + s[2], c0 + s[2], size // 2),
        ], set()
    ) | {
        Submatrix(T[r0:r0 + s[1], c0 + s[2]:c0 + s[3]], r0, c0 + s[2]),
        Submatrix(T[r0:r0 + s[1], c0 + s[3]:c0 + s[4]], r0, c0 + s[3]),
        Submatrix(
            T[r0 + s[1]:r0 + s[2], c0 + s[3]:c0 + s[4]],
            r0 + s[1],
            c0 + s[3],
        ),
        Submatrix(T[r0 + s[2]:r0 + s[3], c0:c0 + s[1]], r0 + s[2], c0),
        Submatrix(T[r0 + s[3]:r0 + s[4], c0:c0 + s[1]], r0 + s[3], c0),
        Submatrix(
            T[r0 + s[3]:r0 + s[4], c0 + s[1]:c0 + s[2]],
            r0 + s[3],
            c0 + s[1],
        ),
    }


def is_zero_or_power_of_2(value: int) -> bool:
    # pylint: disable=invalid-name, missing-function-docstring
    return value & (value - 1) == 0

# ruff: noqa: E741
'''módulo step_3
'''


__all__ = [
    'orthonormalized_shifted_scaled_moments_matrices',
    'shift_scale_matrices'
]
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from collections.abc import Iterator
#
import numpy as np
import numpy.typing as npt
#
from . import step_1, step_2


def shift_scale_matrices(
    n: int,
    k: int,
    mus: list[list[float]],
    sigmas: list[list[float]],
) -> Iterator[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    # pylint: disable=invalid-name
    '''produz as matrizes S_1 e S_2 que no texto são usadas para gerar as
matrizes de momentos indexadas por j > 1 (no texto)
'''
    l = int(np.log2(n // k))
    for j in range(1, l):
        for i in range(n // (k * 2**j)):
            S_1 = step_1.shift_scale_matrix(
                k,
                (mus[j][i] - mus[j - 1][2*i]) / sigmas[j - 1][2*i],
                sigmas[j][i] / sigmas[j - 1][2*i]
            )
            S_2 = step_1.shift_scale_matrix(
                k,
                (mus[j][i] - mus[j - 1][2*i + 1]) / sigmas[j - 1][2*i + 1],
                sigmas[j][i] / sigmas[j - 1][2*i + 1]
            )
            yield S_1, S_2


def orthonormalized_shifted_scaled_moments_matrices(
    n: int,
    k: int,
    mus: list[list[float]],
    sigmas: list[list[float]],
    M1s: list[npt.NDArray],
) -> list[list[npt.NDArray[float]]]:
    # pylint: disable=invalid-name
    '''produz as matrizes de momentos ortogonalizadas, o que efetivamente é a
saída do passo 3
'''
    l = np.log2(n // k)
    Us = [[step_2.orthonormalize(M) for M in M1s]]
    S_gen = shift_scale_matrices(n, k, mus, sigmas)
    prev_Ms = M1s[:]
    for j in range(1, l):
        Us.append([])
        curr_Ms = []
        for i in range(n / (k*2**(j + 1))):
            S_1, S_2 = next(S_gen)
            upper = Us[j - 1][2*i][:k, :] @ prev_Ms[2*i] @ S_1
            lower = Us[j - 1][2*i + 1][k:, :] @ prev_Ms[2*i + 1] @ S_2
            M = np.concat((upper, lower), axis=0)
            curr_Ms.append(M)
            Us[-1].append(step_2.orthonormalize(M))
        prev_Ms = curr_Ms[:]
    return Us

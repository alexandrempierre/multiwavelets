'''módulo step_1
'''


__all__ = [
    'shifted_scaled_moments_matrices',
    'shifted_scaled_moments_matrices_list',
    'shifted_scaled_moments_matrices_callable',
    'shifted_scaled_moments_matrices_prod',
    'shift_scale_matrix',
]
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from collections.abc import Callable
from copy import deepcopy
import numpy as np
import numpy.typing as npt
from scipy.special import comb
from simple import step_0


def shifted_scaled_moments_matrices(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcular as matrizes de momentos transladadas e dilatadas que no texto
são chamadas de M'_{1,i}
'''
    n = len(xs)
    assert n % (2*k) == 0
    s_fn = step_0.s_callable(k, start_idx)
    mu_fn = step_0.mu_callable(xs, k, start_idx)
    sigma_fn = step_0.sigma_callable(xs, k, start_idx)
    matrices_row = []
    for i in range(start_idx, n // (2*k) + start_idx):
        s = s_fn(i)
        mu = mu_fn(start_idx, i)
        sigma = sigma_fn(start_idx, i)
        matrix = np.ones((2*k, 2*k))
        for r in range(2*k):
            for c in range(1, 2*k):
                matrix[r, c] = ((xs[s + r] - mu) / sigma)**c
        matrices_row.append(deepcopy(matrix))
    return matrices_row


def shifted_scaled_moments_matrices_callable(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> Callable[[int], npt.NDArray[np.float64]]:
    # pylint: disable=invalid-name
    '''calcular as matrizes de momentos transladadas e dilatadas que no texto
são chamadas de M'_{1,i} para start_idx <= i < n/(2*k) + start_idx
'''
    n = xs.shape[0]
    assert n % (2*k) == 0
    s_fn = step_0.s_callable(k, start_idx)
    mu_fn = step_0.mu_callable(xs, k, start_idx)
    sigma_fn = step_0.sigma_callable(xs, k, start_idx)
    def shifted_scaled_moments_matrices_inner(
        i: int,
    ) -> npt.NDArray[np.float64]:
        # pylint: disable=invalid-name
        '''calcular a matriz de momentos transladada e dilatada que no texto é
chamada de M'_{1,i}
'''
        s = s_fn(i)
        mu = mu_fn(start_idx, i)
        sigma = sigma_fn(start_idx, i)
        return np.array([
            [((xs[s + r] - mu) / sigma)**c for c in range(2*k)]
            for r in range(2*k)
        ])
    return shifted_scaled_moments_matrices_inner


def shifted_scaled_moments_matrix(
    xs: npt.NDArray[np.float64],
    k: int,
    mu: float,
    sigma: float,
    s: int,
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcular a matriz de momentos transladada e dilatada que no texto é
chamada de M'_{1,i}
'''
    return np.array([
        [((xs[s + r] - mu) / sigma)**c for c in range(2*k)]
        for r in range(2*k)
    ])


def shifted_scaled_moments_matrices_list(
    xs: npt.NDArray[np.float64],
    k: int,
    mus: npt.NDArray[np.float64 | float] | list[np.float64 | float],
    sigmas: npt.NDArray[np.float64 | float] | list[np.float64 | float],
    ss: npt.NDArray[np.float64 | float] | list[np.float64 | float],
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcular as matrizes de momentos transladada e dilatada que no texto é
chamada de M'_{1,i}
'''
    n = xs.shape[0]
    assert n % (2*k) == 0
    return [
        shifted_scaled_moments_matrix(xs, k, mus[0][i], sigmas[0][i], ss[i])
        for i in range(n // (2*k))
    ]


def moments_matrices(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcula a matriz de momentos sem translação nem dilatação M_{1, i}'''
    n = xs.shape[0]
    assert n % (2*k) == 0
    s = step_0.s_array(n, k, start_idx)
    return [
        np.array([
            [xs[s[i] + r]**c for c in range(2*k)]
            for r in range(2*k)
        ])
        for i in range(n // (2*k))
    ]


def shift_scale_matrix(
    k: int,
    mu: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    # pylint: disable=invalid-name
    '''calcula matriz de aplicação de translação e dilatação S(mu, sigma)'''
    return np.array([
        [
            comb(c, r, exact=True) * np.power(-mu, c - r) / np.power(sigma, c)
            if r <= c else 0
            for c in range(2*k)
        ]
        for r in range(2*k)
    ])

def shifted_scaled_moments_matrices_prod(
    xs: npt.NDArray[np.float64],
    k: int,
    mus: npt.NDArray[np.float64 | float] | list[np.float64 | float],
    sigmas: npt.NDArray[np.float64 | float] | list[np.float64 | float],
    start_idx: int = 0,
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcular as matrizes de momentos transladadas e dilatadas que no texto
são chamadas de M'_{1,i}
'''
    matrices = moments_matrices(xs, k, start_idx)
    return [
        matrix @ shift_scale_matrix(k, mu, sigma)
        for matrix, mu, sigma in zip(matrices, mus[0], sigmas[0])
    ]

'''módulo step_1
'''

__all__ = ['shifted_scaled_matrices']
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from copy import deepcopy
import numpy as np
import numpy.typing as npt
import step_0


def shifted_scaled_matrices(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> 'list[npt.NDArray[np.float64]]':
    # pylint: disable=invalid-name
    '''calcular a matriz de momentos transladada e dilatada que no texto é
chamada de M_{1,i}
'''
    n = len(xs)
    assert n % (2 * k) == 0
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

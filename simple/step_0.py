'''módulo step_0

Valores e funções necessários para fazer os cáculos do passo 1 em diante.
'''

__all__ = [
    's_callable', 'mu_callable', 'sigma_callable',
    's_array', 'mu_list', 'sigma_list',
]
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from collections.abc import Callable
#
import numpy as np
import numpy.typing as npt


def s_callable(k: int, start_idx: int = 0) -> 'Callable[[int], int]':
    '''função para gerar o procedimento que vai calcular os valores de s'''
    def s_inner(i: int) -> int:
        '''calcula o valor de s'''
        return (i - start_idx)*2*k
    return s_inner


def s_array(n: int, k: int, start_idx: int = 0) -> npt.NDArray[np.int64]:
    # pylint: disable=invalid-name
    '''array de valores de s'''
    return np.array(
        [
            (i - start_idx)*2*k
            for i in range(start_idx, n // (2*k) + start_idx)
        ],
        dtype=np.int64
    )


def mu_callable(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> 'Callable[[int, int], float]':
    # pylint: disable=invalid-name
    '''função para gerar o procedimento que vai calcular os valores de mu
(translação)'''
    def mu_inner(j: int, i: int) -> float:
        '''calcula o valor de mu (translação)'''
        idx = (
            (i - start_idx)*k*2**(j + 1 - start_idx),
            (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
        )
        return (xs[idx[0]] + xs[idx[1]]) / 2
    return mu_inner


def mu_list(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> 'list[list[float]]':
    # pylint: disable=invalid-name
    '''lista de valores de mu (translação)'''
    n = len(xs)
    l = int(np.log2(n // k))  # noqa: E741
    mu = []
    for row, j in enumerate(range(start_idx, l + start_idx)):
        mu.append([])
        for i in range(
            start_idx,
            n // (k*2**(j + 1 - start_idx)) + start_idx
        ):
            idx = (
                (i - start_idx)*k*2**(j + 1 - start_idx),
                (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
            )
            mu[row].append((xs[idx[0]] + xs[idx[1]]) / 2)
    return mu


def sigma_callable(
    xs: npt.NDArray[np.float64],
    k: int,
    start_idx: int = 0,
) -> 'Callable[[int, int], float]':
    # pylint: disable=invalid-name
    '''função para gerar o procedimento que vai calcular os valores de sigma
(dilatação)'''
    def sigma_inner(j: int, i: int) -> float:
        '''calcula o valor de sigma (dilatação)'''
        idx = (
            (i - start_idx)*k*2**(j + 1 - start_idx),
            (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
        )
        return (xs[idx[1]] - xs[idx[0]]) / 2
    return sigma_inner


def sigma_list(
    xs: np.ndarray,
    k: int,
    start_idx: int = 0,
) -> 'list[list[np.ndarray]]':
    # pylint: disable=invalid-name
    '''lista de valores de sigma (dilatação)'''
    n = len(xs)
    l = int(np.log2(n // k))  # noqa: E741
    sigma = []
    for row, j in enumerate(range(start_idx, l + start_idx)):
        sigma.append([])
        for i in range(
            start_idx,
            n // (k*2**(j + 1 - start_idx)) + start_idx
        ):
            idx = (
                (i - start_idx)*k*2**(j + 1 - start_idx),
                (i + 1 - start_idx)*k*2**(j + 1 - start_idx) - 1
            )
            sigma[row].append((xs[idx[1]] - xs[idx[0]]) / 2)
    return sigma

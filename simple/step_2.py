'''módulo step_2
'''


__all__ = ['orthonormalize']
__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
import numpy.typing as npt


def orthonormalize(matrix: npt.NDArray) -> npt.NDArray:
    '''alias para processo de ortonormalização de Gram-Schmidt usado no texto
para calcular as matrizes U (ortogonais) a partir das matrizes M (matriz de
momentos transladada e dilatada)
'''
    return np.linalg.qr(matrix, mode='complete')[0]

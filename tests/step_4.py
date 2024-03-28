'''módulo step_4
'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import random

import numpy as np
import numpy.typing as npt
#
from .. import step_4 as s4


EXP_MAX = 100


class Test_ZeroOrPowerOf2:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste com alguns valores para verificar se a função dá os resultados
esperados
'''
    def test_zero_or_power_of_2(self):
        powers_of_2 = {0} | {2**exponent for exponent in range(1_000)}
        numbers = set(range(100_000))
        for pow2, notpow2 in zip(powers_of_2, numbers - powers_of_2):
            assert s4.is_zero_or_power_of_2(pow2)
            assert not s4.is_zero_or_power_of_2(notpow2)


class TestRemoveSingularity:
    '''teste para conferir se a função está lidando corretamente com os valores
inf, -inf e NaN, e se valores muito grandes (em módulo) são mantidos ou
considerados infinitos pela função
'''
    def test_singularity_removal(self):
        assert s4.remove_singularity(
            lambda x, y: float('nan'),
            random.random(),
            random.random(),
        ) == 0
        assert s4.remove_singularity(
            lambda x, y: float('inf'),
            random.random(),
            random.random(),
        ) == 0
        assert s4.remove_singularity(
            lambda x, y: float('-inf'),
            random.random(),
            random.random(),
        ) == 0
        assert s4.remove_singularity(
            lambda x, y: 1e+100,
            random.random(),
            random.random(),
        ) == 1e+100
        assert s4.remove_singularity(
            lambda x, y: -1e+100,
            random.random(),
            random.random(),
        ) == -1e+100


class TestSymetricExtraction:
    '''testar se a função que extrai blocos paralelos à diagonal principal está
funcionando como o esperado
'''
    def test_block_extraction_from_diagonal_ones_8_index_0_size_1(self):
        n, size = 8, 1
        Ones = np.ones((n, n))
        blocks_check = {(p, p): size for p in range(0, n, size)}
        blocks = s4.extract_symmetric_block(Ones, 0, size)
        assert blocks_check == blocks
    

    def test_block_extraction_from_diagonal_ones_8_index_0_size_2(self):
        n, size = 8, 2
        Ones = np.ones((n, n))
        blocks_check = {(p, p): size for p in range(0, n, size)}
        blocks = s4.extract_symmetric_block(Ones, 0, size)
        assert blocks_check == blocks
    

    def test_block_extraction_from_diagonal_ones_8_index_0_size_4(self):
        n, size = 8, 4
        Ones = np.ones((n, n))
        blocks_check = {(p, p): size for p in range(0, n, size)}
        blocks = s4.extract_symmetric_block(Ones, 0, size)
        assert blocks_check == blocks
    

    def test_block_extraction_from_diagonal_ones_8_index_0_size_8(self):
        n, size = 8, 8
        Ones = np.ones((n, n))
        blocks_check = {(p, p): size for p in range(0, n, size)}
        blocks = s4.extract_symmetric_block(Ones, 0, size)
        assert blocks_check == blocks


    def test_symmetric_block_extraction_ones_8_index_3_size_1(self):
        n, index, size = 4, 3, 1
        Ones = np.ones((n, n))
        blocks_check = {(3, 0): size, (0, 3): size}
        blocks = s4.extract_symmetric_block(Ones, index, size)
        assert blocks_check == blocks
    

    def test_symmetric_block_extraction_ones_8_index_2_size_1(self):
        n, index, size = 4, 2, 1
        Ones = np.ones((n, n))
        blocks_check = {
            (0, 2): size,
            (1, 3): size,
            (2, 0): size,
            (3, 1): size,
        }
        blocks = s4.extract_symmetric_block(Ones, index, size)
        assert blocks_check == blocks
    

    def test_symmetric_block_extraction_ones_8_index_2_size_2(self):
        n, index, size = 4, 2, 2
        Ones = np.ones((n, n))
        blocks_check = {(0, 2): size, (2, 0): size}
        blocks = s4.extract_symmetric_block(Ones, index, size)
        assert blocks_check == blocks
    

    def test_symmetric_block_extraction_ones_8_index_1_size_1(self):
        n, index, size = 4, 1, 1
        Ones = np.ones((n, n))
        blocks_check = {
            (0, 1): size,
            (1, 2): size,
            (2, 3): size,
            (1, 0): size,
            (2, 1): size,
            (3, 2): size,
        }
        blocks = s4.extract_symmetric_block(Ones, index, size)
        assert blocks_check == blocks


class TestSubmatrixExtraction:
    # pylint: disable=invalid-name, missing-function-docstring
    '''testes para conferir se o processo de extrair as submatrizes está
correto
'''
    def test_submatrix_extraction_n_4_k_1(self):
        n, k = 4, 1
        Ones = np.ones((n ,n))
        blocks_check = {
            (0, 0): k, (0, 1): k, (0, 2): k, (0, 3): k,
            (1, 0): k, (1, 1): k, (1, 2): k, (1, 3): k,
            (2, 0): k, (2, 1): k, (2, 2): k, (2, 3): k,
            (3, 0): k, (3, 1): k, (3, 2): k, (3, 3): k,
        }
        blocks = s4.extract_submatrices(Ones, k)
        assert blocks_check == blocks
    

    def test_submatrix_extraction_n_8_k_2(self):
        n, k = 8, 2
        Ones = np.ones((n ,n))
        blocks_check = {
            (0, 0): k, (0, 2): k, (0, 4): k, (0, 6): k,
            (2, 0): k, (2, 2): k, (2, 4): k, (2, 6): k,
            (4, 0): k, (4, 2): k, (4, 4): k, (4, 6): k,
            (6, 0): k, (6, 2): k, (6, 4): k, (6, 6): k,
        }
        blocks = s4.extract_submatrices(Ones, k)
        assert blocks_check == blocks
    
    def test_submatrix_extraction_n_8_k_1(self):
        n, k = 8, 1
        Ones = np.ones((n ,n))
        blocks_check = {
            (0, 6): 2*k,
            (0, 4): 2*k, (2, 6): 2*k,
            (0, 3): k, (2, 5): k, (4, 7): k,
            (0, 2): k, (1, 3): k, (2, 4): k, (3, 5): k, (4, 6): k, (5, 7): k,
            (0, 1): k, (1, 2): k, (2, 3): k, (3, 4): k, (4, 5): k, (5, 6): k, (6, 7): k,
            (0, 0): k, (1, 1): k, (2, 2): k, (3, 3): k, (4, 4): k, (5, 5): k, (6, 6): k, (7, 7): k,
            (1, 0): k, (2, 1): k, (3, 2): k, (4, 3): k, (5, 4): k, (6, 5): k, (7, 6): k,
            (2, 0): k, (3, 1): k, (4, 2): k, (5, 3): k, (6, 4): k, (7, 5): k,
            (3, 0): k, (5, 2): k, (7, 4): k,
            (4, 0): 2*k, (6, 2): 2*k,
            (6, 0): 2*k,
        }
        blocks = s4.extract_submatrices(Ones, k)
        assert blocks_check == blocks
    

class TestOperatorMatrix:
    '''_'''
    def test_operator_matrix_constant(self):
        n = 4
        xs = np.array([0, 1/3, 2/3, 1])
        T_check = np.ones((n, n)) / (n - 1)
        T = s4.operator_matrix(xs, lambda x, t: 1)
        assert np.all(T_check == T)
    

    def test_operator_matrix_inv_subtraction(self):
        def kernel(x, t):
            return 1/(x - t)
        n = 4
        xs = np.array([0, 1/3, 2/3, 1])
        T_check = np.array([
            [0,   -1,   -1/2, -1/3],
            [1,    0,   -1,   -1/2],
            [1/2,  1,    0,   -1  ],
            [1/3,  1/2,  1,    0  ],
        ])
        T = s4.operator_matrix(xs, kernel)
        assert np.allclose(T_check, T)

# 1 1 1 1
# 1 1 1 1
# 1 1 1 1
# 1 1 1 1
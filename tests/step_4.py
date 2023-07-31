'''módulo step_4
'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
#
from .. import step_4


class Test_ZeroOrPowerOf2:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste com alguns valores para verificar se a função dá os resultados
esperados
'''
    def test_zero_or_power_of_2(self):
        powers_of_2 = {0} | {2**exponent for exponent in range(1_000)}
        numbers = set(range(100_000))
        for pow2, notpow2 in zip(powers_of_2, numbers - powers_of_2):
            assert step_4.is_zero_or_power_of_2(pow2)
            assert not step_4.is_zero_or_power_of_2(notpow2)


class TestSubmatrixExtraction:
    # pylint: disable=invalid-name, missing-function-docstring
    '''testes para conferir se o processo de extrar as submatrizes está correto
'''

    def test_n_4_k_1_submatrix_extraction(self):
        # n = 4
        k = 1
        matrix = np.array([
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44],
        ])
        submatrices = {
            step_4.Submatrix(
                subarray=np.array([val]),
                row_start=row_idx,
                col_start=col_idx,
            )
            for row_idx, row in enumerate(matrix)
            for col_idx, val in enumerate(row)
        }
        extracted_submatrices = step_4.extract_submatrices(matrix, k)
        assert submatrices == extracted_submatrices

    def test_n_8_k_2_submatrix_extraction(self):
        n = 8
        k = 2
        matrix = np.array([
            [11, 12, 13, 14, 15, 16, 17, 18],
            [21, 22, 23, 24, 25, 26, 27, 28],
            [31, 32, 33, 34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, 53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, 65, 66, 67, 68],
            [71, 72, 73, 74, 75, 76, 77, 78],
            [81, 82, 83, 84, 85, 86, 87, 88],
        ])
        submatrices = {
            step_4.Submatrix(
                subarray=matrix[row:row+k, col:col+k],
                row_start=row, col_start=col
            )
            for row in range(0, n, k) for col in range(0, n, k)
        }
        extracted_submatrices = step_4.extract_submatrices(matrix, k)
        assert submatrices == extracted_submatrices


    def test_n_8_k_1_submatrix_extraction(self):
        # n = 8
        k = 1
        matrix = np.array([
            [11, 12, 13, 14, 15, 16, 17, 18],
            [21, 22, 23, 24, 25, 26, 27, 28],
            [31, 32, 33, 34, 35, 36, 37, 38],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [51, 52, 53, 54, 55, 56, 57, 58],
            [61, 62, 63, 64, 65, 66, 67, 68],
            [71, 72, 73, 74, 75, 76, 77, 78],
            [81, 82, 83, 84, 85, 86, 87, 88],
        ])
        submatrices = {
            step_4.Submatrix(np.array([10 * (row + 1) + col + 1]), row, col)
            for col in range(4) for row in range(4)
        } | {
            step_4.Submatrix(
                np.array([40 + 10 * (row + 1) + col + 5]),
                row + 4,
                col + 4
            )
            for col in range(4) for row in range(4)
        } | {
            step_4.Submatrix(
                subarray=np.array([35]), row_start=2, col_start=4
            ),
            step_4.Submatrix(
                subarray=np.array([45]), row_start=3, col_start=4
            ),
            step_4.Submatrix(
                subarray=np.array([36]), row_start=2, col_start=5
            ),
            step_4.Submatrix(
                subarray=np.array([46]), row_start=3, col_start=5
            ),
            step_4.Submatrix(
                subarray=np.array([53]), row_start=4, col_start=2
            ),
            step_4.Submatrix(
                subarray=np.array([54]), row_start=4, col_start=3
            ),
            step_4.Submatrix(
                subarray=np.array([63]), row_start=5, col_start=2
            ),
            step_4.Submatrix(
                subarray=np.array([64]), row_start=5, col_start=3
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [15, 16],
                    [25, 26]
                ]),
                row_start=0, col_start=4
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [17, 18],
                    [27, 28]
                ]),
                row_start=0, col_start=6
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [37, 38],
                    [47, 48]
                ]),
                row_start=2, col_start=6
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [51, 52],
                    [61, 62]
                ]),
                row_start=4, col_start=0
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [71, 72],
                    [81, 82]
                ]),
                row_start=6, col_start=0
            ),
            step_4.Submatrix(
                subarray=np.array([
                    [73, 74],
                    [83, 84]
                ]),
                row_start=6, col_start=2
            ),
        }
        extracted_submatrices = step_4.extract_submatrices(matrix, k)
        assert submatrices == extracted_submatrices

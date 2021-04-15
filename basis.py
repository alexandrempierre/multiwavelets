"""
Module that calculates the basis matrices.
"""


__all__ = ['shifted_scaled_moments_matrices']
__author__ = 'Alexandre Pierre'


from typing import Optional, List, Tuple

import numpy as np
from scipy import linalg
from scipy.special import binom

import wavelet_parameters as wave_params


def shift_scale_matrix(shift: float, scale: float,
                       zero_moments: int) -> np.ndarray:
    """Build S matrix based on shift and scale values."""
    return np.array([
        [
            0 if row > col else binom(col, row)*(-shift)**(col-row)/scale**col
            for col in range(2*zero_moments)
        ]
        for row in range(2*zero_moments)
    ])


def shifted_scaled_moments_matrices_firstrow(xs: np.ndarray,
                                             zero_moments: int,
                                             Shift: np.ndarray,
                                             Scale: np.ndarray,
                                             S: np.ndarray
                                             ) -> List[np.ndarray]: 
    """Initial shifted and scaled matrices."""
    points = wave_params.get_points(xs, zero_moments)

    Ms = []
    for i in range(points // (2*zero_moments)):
        s_i = S[i]
        shift = Shift[0, i]
        scale = Scale[0, i]

        M = np.ones((2*zero_moments, 2*zero_moments))
        for row in range(2*zero_moments):
            for col in range(1, 2*zero_moments):
                M[row, col] = ((xs[s_i + row] - shift)/scale)**col
        
        Ms.append(M.copy())

    return Ms


def shifted_scaled_moments_matrices_row(points: int,
                                        zero_moments: int,
                                        Ms_prev: List[np.ndarray],
                                        Shift: np.ndarray,
                                        Scale: np.ndarray,
                                        j: int) -> List[np.ndarray]:
    """Calculate one sequence of transformed moments matrices."""
    Ms: List[np.ndarray] = []
    for i in range(points // (2**(j+1)*zero_moments)):
        S1 = shift_scale_matrix((Shift[j, i] - Shift[j-1, 2*i])
                                / Scale[j-1, 2*i],
                                Scale[j, i] / Scale[j-1, 2*i],
                                zero_moments)

        S2 = shift_scale_matrix((Shift[j, i] - Shift[j-1, 2*i + 1])
                                / Scale[j-1, 2*i + 1],
                                Scale[j, i] / Scale[j-1, 2*i + 1],
                                zero_moments)

        U1 = linalg.qr(Ms[2*i])[0].T
        U2 = linalg.qr(Ms[2*i + 1])[0].T

        upper_half = U1[:zero_moments, :] @ Ms_prev[2*i] @ S1
        lower_half = U2[:zero_moments, :] @ Ms_prev[2*i + 1] @ S2

        Ms.append(np.block([[upper_half], [lower_half]]))

    return Ms


def shifted_scaled_moments_matrices(xs: np.ndarray, zero_moments: int,
                                    Shift: np.ndarray, Scale: np.ndarray,
                                    S: np.ndarray) -> List[List[np.ndarray]]:
    """Create all the moments matrices."""
    points = wave_params.get_points(xs, zero_moments)
    m = wave_params.get_m(points, zero_moments)

    Ms = [
            shifted_scaled_moments_matrices_firstrow(xs, zero_moments,
                                                     Shift, Scale, S)
    ]

    for j in range(1, m):
        Ms.append(
            shifted_scaled_moments_matrices_row(points, zero_moments, Ms[j-1],
                                                Shift, Scale, j)
        )

    return Ms


def basis_matrices(xs: np.ndarray, zero_moments: int, Shift: np.ndarray,
                   Scale: np.ndarray, S: np.ndarray) -> List[np.ndarray]:
    """Calculate the final basis matrices"""
    points = wave_params.get_points(xs, zero_moments)
    U_shape = (zero_moments, 2*zero_moments)

    Ms = shifted_scaled_moments_matrices(xs, zero_moments, Shift, Scale, S)
    Us = [[linalg.qr(M)[0].T for M in M_row] for M_row in Ms]
    basis = []
    
    for j, U_j in enumerate(Us):
        upper_half = []
        lower_half = []
        max_i = len(U_j) - 1

        for i, U_j_i in enumerate(U_row):
            upper_half += [
                    [np.zeros(U_shape)]*i
                    + [U_j_i[:zero_moments, :]]
                    + [np.zeros(U_shape)]*(max_i - i)
            ]
            lower_half += [
                    [np.zeros(U_shape)]*i
                    + [U_j_i[zero_moments:, :]]
                    + [np.zeros(U_shape)]*(max_i - i)
            ]
        
        blocks = upper_half + lower_half
        len_blocks = len(blocks)
        zero_shape = (len_blocks, len_blocks)
        U = np.block([np.eye(points - len_blocks), np.zeros(zero_shape)],
                     [np.zeros(zero_shape), blocks])
        basis.append(U.copy())

    return basis


# TODO: expand the data input to get only the interval and the number
# of points
def main(xs: np.ndarray, zero_moments: int) -> List[np.ndarray]:
    """"""
    return

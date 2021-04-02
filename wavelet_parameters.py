"""This file contains the functions used to calculate the shift and
scale parameters necessary to a wavelet bassis. It also has some
auxiliary functions.
"""


__all__ = ['shift', 'scale', 's', 'get_points', 'get_m']
__author__ = 'Alexandre Pierre'


from typing import Callable, Optional

import numpy as np


def intlog2(n: int) -> int:
    """Calculate the necessary number of bits to represent the integer
    n. It's equivalent to the ceiling of the log base 2 of n.
    """
    return n.bit_length()


def get_points(xs: np.ndarray, zero_moments: int) -> int:
    """Get the number of points and check whether it's valid."""
    points = len(xs)
    assert points % (2*zero_moments) == 0

    return points


#  checks whether m is the maximum window size
def get_m(points: int, zero_moments: int) -> int:
    """Calculate and check the value of m."""
    m = intlog2(points // zero_moments)
    assert 2**m == points // zero_moments

    return m


def idx(points: int, zero_moments: int) -> np.ndarray:
    """Calculate all the possible indices necessary for the shift and
    scale calculation. Returns the indices in a matrix.
    """
    m = get_m(points, zero_moments)

    return np.array([
        [
            (i*zero_moments*2**(j+1), (i+1)*zero_moments*2**(j+1)-1)
            for i in range(points // (2*zero_moments))
        ]
        for j in range(m)
    ], dtype=int)


def shift(xs: np.ndarray, zero_moments: int,
          Idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate all possible shifts. Return the result as a matrix."""
    points = get_points(xs, zero_moments)
    m = get_m(points, zero_moments)

    if Idx is None:
        Idx = idx(points, zero_moments)

    return np.array([
        [
            (xs[Idx[j, i, 0]] + xs[Idx[j, i, 1]]) / 2
            for i in range(points // (2**(j+1)*zero_moments))
        ]
        for j in range(m)
    ], dtype=np.float64)


def scale(xs: np.ndarray, zero_moments: int,
          Idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate all possible scales. Returns a matrix."""
    points = get_points(xs, zero_moments)
    m = get_m(points, zero_moments)

    if Idx is None:
        Idx = idx(points, zero_moments)

    return np.array([
        [
            (xs[Idx[j, i, 1]] - xs[Idx[j, i, 0]]) / 2
            for i in range(points // (2**(j+1)*zero_moments))
        ]
        for j in range(m)
    ], dtype=np.float64)


def s(points: int, zero_moments: int) -> np.ndarray:
    """Given the number of points and of zero moments calculate s.
    """
    assert points % (2*zero_moments) == 0

    return np.array([i for i in range(0, points, 2*zero_moments)], dtype=int)

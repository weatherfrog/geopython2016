"""
Random array algorithm
"""

from itertools import product
import random
from math import log2

import numpy as np


def random_array(shape, minvalue, maxvalue):
    """
    Create random data array using the diamond square algorithm.

    Args:
        shape: tuple (rows, cols)
        minvalue: desired minimal value of array
        maxvalue: desired maximum value of array

    Returns:
        Random 2D array with shape (size, size) and all values between
        ``minvalue`` and ``maxvalue``
    """
    rows, cols = shape
    # compute size (a power of 2, plus 1) large enough to fit cols/rows
    m = max(cols, rows)
    m_exp = int(log2(m))
    size = 2 ** m_exp + 1
    if size < m:
        size = 2 ** (m_exp + 1) + 1

    init_values = (
        random.random() * 2 - 1,  # values between -1 and 1
        random.random() * 2 - 1,
        random.random() * 2 - 1,
        random.random() * 2 - 1,
    )
    arr = diamond_square(size, init_values)

    # extract sub-array of desired size
    arr = arr[0: rows, 0:cols]

    arr_norm = normalize(arr, minvalue, maxvalue)

    return arr_norm


def diamond_square(size, init_values):
    """
    Diamond square algorithm that generates random data/terrains.

    Usage:
        random_array = diamond_square(size=129, (0.2, -0.4, 0.8, -0.4))

    Args:
        size: side length of a quadratic array to be created, must be equal
            to 2^n + 1, for some integer n.
        init_values: 4-tuple with values between -1 and 1

    Returns:
        Random 2D array with shape (size, size)
    """
    arr = np.zeros((size, size))
    # initialize corners
    arr[0, 0] = init_values[0]
    arr[0, -1] = init_values[1]
    arr[-1, 0] = init_values[2]
    arr[-1, -1] = init_values[3]

    __diamond_step(arr)

    length = arr.shape[0]
    step = (length - 1) // 2
    while step > 0:
        __square_step(arr, step)
        if step >= 2:
            # iterate through all squares (w.r.t. step)
            for row, col in product(range(0, length-1, step),
                                    range(0, length-1, step)):
                sub_arr = arr[row:row+step+1,
                              col:col+step+1]
                __diamond_step(sub_arr)
        step //= 2

    return arr


def __square_step(arr, step):
    """
    perform square step on ``arr``

    Args:
        arr: full 2D array
        step: "gap" between cells to be processed (step goes from big
            to small)
    """
    length = arr.shape[0]
    # iterate through all diamond midpoints (x):
    #    o
    # o  x  o
    #    o

    # indicates whether diamond vertices in this row are in even/odd positions
    row_odd = True  # diamond vertices are in odd positions in 1st row
    for row in range(0, length, step):
        col_start = step if row_odd else 0
        col_end = length - 1 if row_odd else length
        for col in range(col_start, col_end, step*2):
            points = []
            try:  # top point
                points.append(arr[row-step, col])
            except IndexError:
                pass
            try:  # left point
                points.append(arr[row, col-step])
            except IndexError:
                pass
            try:  # bottom point
                points.append(arr[row+step, col])
            except IndexError:
                pass
            try:  # right point
                points.append(arr[row, col+step])
            except IndexError:
                pass
            avg = sum(points) / len(points)
            rndm = get_random_offset(step)
            arr[row, col] = avg + rndm
        row_odd = not row_odd


def __diamond_step(arr):
    """
    perform diamond step on sub-array ``arr``
    """
    # compute average of four corner points
    avg = (arr[0, 0] + arr[0, -1] + arr[-1, 0] + arr[-1, -1]) / 4
    # set midpoint to average of corner points plus a random value
    idx_mid = arr.shape[0] // 2  # indices of midpoint
    rndm = get_random_offset(arr.shape[0])
    arr[idx_mid, idx_mid] = avg + rndm


def get_random_offset(step):
    """
    """
    return (random.random() - 0.5) * step


def normalize(arr, minvalue=0, maxvalue=1):
    """
    normalize array to range (0, 1)
    """
    zero_offset = arr.min()
    temp = arr - zero_offset
    ptp = temp.ptp()  # peak to peak
    arr_normalized = temp / ptp

    factor = abs(maxvalue - minvalue)
    arr_normalized *= factor

    arr_normalized += minvalue

    return arr_normalized

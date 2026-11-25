"""Interval transforms applied after calibration.

interval_fn3 implements Equations 3 and 4 of the paper, with l = omega (scale)
and b = beta (additive bias).
"""
import numpy as np



def interval_fn2(low,mid,high,l):
    lower = np.maximum(low - l,0)
    upper = np.minimum(high + l,1)
    return lower,mid, upper


def interval_fn3(low,mid,high,b,l):
    lower = np.maximum((low - mid)*l + mid - b,0)
    upper = np.minimum((high - mid)*l + mid + b,1)
    return lower,mid, upper

#!/usr/bin/env python3
"""
RDE O网格 - 源项
旋转爆轰源项（Numba 优化版本）。
"""

import numpy as np

try:
    from numba import jit, prange
except ImportError:
    prange = range
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, parallel=True)
def _source_term_ogrid_core(cell_centers, t, inner_r, outer_r, omega,
                            wave_pressure, tau, sigma, R):
    n_cells = cell_centers.shape[0]
    S = np.zeros(n_cells)
    theta0 = (omega * t) % (2.0 * np.pi)
    two_pi = 2.0 * np.pi
    for i in prange(n_cells):
        x = cell_centers[i, 0]
        y = cell_centers[i, 1]
        r = np.sqrt(x*x + y*y)
        if r >= inner_r and r <= outer_r:
            theta = np.arctan2(y, x)
            if theta < 0:
                theta += two_pi
            dtheta = theta - theta0
            if dtheta > np.pi:
                dtheta -= two_pi
            elif dtheta < -np.pi:
                dtheta += two_pi
            if dtheta >= 0:
                S[i] = (wave_pressure / tau *
                        np.exp(- (dtheta * R)**2 / (2.0 * sigma * sigma)))
    return S


def source_term_ogrid(cell_centers, t, inner_r, outer_r, omega, wave_pressure, tau, sigma, R):
    return _source_term_ogrid_core(cell_centers, t, inner_r, outer_r, omega,
                                    wave_pressure, tau, sigma, R)

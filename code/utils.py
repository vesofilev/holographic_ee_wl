# ASSERT_CONVENTION: metric (+,-,-,-,-), z_h=1, d=4, L=1
# ASSERT_CONVENTION: f(z)=1-z^4, epsilon=1e-4, float64 for area
# ASSERT_CONVENTION: area = V_2 * int (1/z^3) sqrt(1 + z'^2/f(z)) dx
"""
Shared utilities for RT surface computations in AdS5-Schwarzschild.

Provides: blackening factor, non-uniform grid construction, ODE benchmark loading.

References:
  - Ryu and Takayanagi, hep-th/0603001, hep-th/0605073
  - Filev, arXiv:2506.20115

Reproducibility:
  Python 3.11+, NumPy, PyTorch
  float64 throughout
"""

import numpy as np


# =============================================================================
# Physical constants and conventions
# =============================================================================
Z_H = 1.0        # Horizon position
EPSILON = 1e-4    # UV cutoff
D_BOUNDARY = 4    # d=4 boundary dimensions (AdS5/CFT4)


def blackening_factor(z, z_h=Z_H):
    """
    Blackening factor for AdS5-Schwarzschild: f(z) = 1 - (z/z_h)^4.

    Works with both numpy arrays and torch tensors.

    Parameters
    ----------
    z : array_like or torch.Tensor
        Radial coordinate(s).
    z_h : float
        Horizon position (default 1.0).

    Returns
    -------
    f : same type as z
        Blackening factor value(s).
    """
    return 1.0 - (z / z_h) ** 4


def make_nonuniform_grid(l, n_points=2000, uv_frac=0.8, delta_frac=0.1):
    """
    Construct a non-uniform x-grid on [-l/2, l/2] with dense UV boundaries.

    Allocates uv_frac of points to the regions within delta_frac of the endpoints,
    where the 1/z^3 singularity is most severe. This piecewise linear approach
    is more robust for the RT area functional than a standard Chebyshev grid.

    Parameters
    ----------
    l : float
        Strip width.
    n_points : int
        Total number of grid points (default 2000).
    uv_frac : float
        Fraction of points to allocate to the UV boundary layers (default 0.8).
    delta_frac : float
        Width of the UV boundary layers as a fraction of l/2 (default 0.1).

    Returns
    -------
    x_grid : np.ndarray, shape (n_points,)
        Sorted ascending x-values in [-l/2, l/2].
    """
    l_half = l / 2.0
    delta = delta_frac * l_half
    n_uv = int(n_points * uv_frac // 2)
    n_ir = n_points - 2 * n_uv

    # UV left: [-l/2, -l/2 + delta]
    x_uv_left = np.linspace(-l_half, -l_half + delta, n_uv, endpoint=False)
    # IR: [-l/2 + delta, l/2 - delta]
    x_ir = np.linspace(-l_half + delta, l_half - delta, n_ir, endpoint=False)
    # UV right: [l/2 - delta, l/2]
    x_uv_right = np.linspace(l_half - delta, l_half, n_uv)

    return np.concatenate([x_uv_left, x_ir, x_uv_right])


def make_half_grid(l, n_points=8000, x_s_frac=0.2):
    """
    Non-uniform grid on [0, l/2) for the half-strip with dense boundary.

    Allocates 30% of points to the interior [0, x_s] and 70% to the
    boundary region [x_s, l/2).  The endpoint x = l/2 is excluded
    (replaced by l/2*(1-1e-8)) to avoid the z' -> -inf singularity
    from the power-law BC encoding.

    Parameters
    ----------
    l : float
        Strip width.
    n_points : int
        Total number of grid points.
    x_s_frac : float
        Split fraction: x_s = x_s_frac * l.

    Returns
    -------
    x_grid : np.ndarray, shape (n_points,)
    """
    l_half = l / 2.0
    x_s = x_s_frac * l
    n_int = max(int(n_points * 0.3), 100)
    n_bdy = n_points - n_int

    x_int = np.linspace(0, x_s, n_int, endpoint=False)
    x_bdy = np.linspace(x_s, l_half * (1.0 - 1e-8), n_bdy)

    return np.concatenate([x_int, x_bdy])


def load_ode_benchmark(path="data/ode_benchmark.npz"):
    """
    Load the ODE benchmark data produced by Plan 01.

    Parameters
    ----------
    path : str
        Path to the .npz file.

    Returns
    -------
    data : dict
        Dictionary with keys: z_star_grid, l_of_zstar, A_reg_of_zstar,
        l_c, z_star_c, epsilon, and other metadata.
    """
    raw = np.load(path)
    data = {
        'z_star_grid': raw['z_star_grid'],
        'l_of_zstar': raw['l_of_zstar'],
        'A_reg_of_zstar': raw['A_reg_of_zstar'],
        'l_c': float(raw['l_c'][0]),
        'z_star_c': float(raw['z_star_c'][0]),
        'epsilon': float(raw['epsilon'][0]),
        'z_h': float(raw['z_h'][0]),
        'd_boundary': int(raw['d_boundary'][0]),
    }
    return data

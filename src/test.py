import numpy as np
import math

def quintic_func(q0, qf, T, qd0=0, qdf=0):
    # solve for the polynomial coefficients using least squares
    # fmt: off
    X = [
        [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
        [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
        [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
        [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
        [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
    ]

    # fmt: on
    coeffs, _, _, _ = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )

def mtraj(q0, qf, t):
    traj = []
    for i in range(len(q0)):
        # for each axis
        traj.append(quintic_func(q0[i], qf[i], t))

    for i in range(t):
        print(traj[0](t))
    """
    y = np.array([tg[0] for tg in traj]).T
    yd = np.array([tg[1] for tg in traj]).T
    ydd = np.array([tg[2] for tg in traj]).T
    """

    #return y, yd, ydd
    return None


mtraj(np.array([0.0, 0.0]), np.array([1.57, 0.30]), 10)
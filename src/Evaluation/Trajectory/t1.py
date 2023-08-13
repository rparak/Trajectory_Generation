#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from collections import namedtuple

try:  # pragma: no cover
    import sympy
    _sympy = True
except ImportError:
    _sympy = False

def getvector(v, dim=None, out='array', dtype=np.float64):
    if isinstance(v, (int, np.int64, float)) or (
            _sympy and isinstance(v, sympy.Expr)):  # handle scalar case
        v = [v]

    if isinstance(v, (list, tuple)):
        # list or tuple was passed in
        dt = dtype
        if _sympy:
            if any([isinstance(x, sympy.Expr) for x in v]):
                dt = None
            
        if dim is not None and v and len(v) != dim:
            raise ValueError("incorrect vector length")
        if out == 'sequence':
            return v
        elif out == 'array':
            return np.array(v, dtype=dt)
        elif out == 'row':
            return np.array(v, dtype=dt).reshape(1, -1)
        elif out == 'col':
            return np.array(v, dtype=dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")

    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim is not None:
            if not (s == (dim,) or s == (1, dim) or s == (dim, 1)):
                raise ValueError("incorrect vector length: expected {}, got {}".format(dim, s))

        v = v.flatten()

        if v.dtype.kind != 'O':
            dt = dtype

        if out == 'sequence':
            return list(v.flatten())
        elif out == 'array':
            return v.astype(dt)
        elif out == 'row':
            return v.astype(dt).reshape(1, -1)
        elif out == 'col':
            return v.astype(dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise TypeError("invalid input type")
    
def  jtraj(q0, q1, tv, qd0=None, qd1=None):
    if isinstance(tv, int):
        tscal = 1
        t = np.linspace(0, 1, tv) # normalized time from 0 -> 1
    else:
        tscal = max(tv)
        t = tv.flatten() / tscal

    q0 = getvector(q0)
    q1 = getvector(q1)
    assert len(q0) == len(q1), 'q0 and q1 must be same size'
    
    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        assert len(qd0) == len(q0), 'qd0 has wrong size'
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        assert len(qd1) == len(q0), 'qd1 has wrong size'

    # compute the polynomial coefficients
    A =   6 * (q1 - q0) - 3 * (qd1 + qd0) * tscal
    B = -15 * (q1 - q0) + (8 * qd0 + 7 * qd1) * tscal
    C =  10 * (q1 - q0) - (6 * qd0 + 4 * qd1) * tscal
    E =       qd0 * tscal #  as the t vector has been normalized
    F =       q0
    
    n = len(q0)
    
    tt = np.array([t**5, t**4, t**3, t**2, t, np.ones(t.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])
    
    # qt = tt @ coeffs
    qt = np.matmul(tt,coeffs)

    # compute  velocity
    c = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    # qdt = tt @ coeffs / tscal

    qdt = qt / tscal

    # compute  acceleration
    c = np.array([np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)])
    qddt = qt / tscal**2

    return namedtuple('jtraj', 't q qd qdd')(tt, qt, qdt, qddt)

def mstraj(viapoints, dt, tacc, qdmax=None, tsegment=None, q0=None, qd0=None, qdf=None, verbose=False):
    if q0 is None:
        q0 = viapoints[0,:]
        viapoints = viapoints[1:,:]
    else:
        assert viapoints.shape[1] == len(q0), 'WP and Q0 must have same number of columns'

    ns, nj = viapoints.shape
    Tacc = tacc
    
    assert not (qdmax is not None and tsegment is not None), 'cannot specify both qdmax and tsegment'
    if qdmax is None:
        assert tsegment is not None, 'tsegment must be given if qdmax is not'
        assert len(tsegment) == ns, 'Length of TSEG does not match number of viapoints'
    if tsegment is None:
        assert qdmax is not None, 'qdmax must be given if tsegment is not'
        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax = np.tile(qdmax, (nj,))
        else:
            assert len(qdmax) == nj, 'Length of QDMAX does not match number of axes'
    
    if isinstance(Tacc, (int, float)):
        Tacc = np.tile(Tacc, (ns,))
    else:
        assert len(Tacc) == ns, 'Tacc is wrong size'
    if qd0 is None:
        qd0 = np.zeros((nj,))
    else:
        assert len(qd0) == len(q0), 'qd0 is wrong size'
    if qdf is None:
        qdf = np.zeros((nj,))
    else:
        assert len(qdf) == len(q0), 'qdf is wrong size'

    # set the initial conditions
    q_prev = q0
    qd_prev = qd0

    clock = 0     # keep track of time
    arrive = np.zeros((ns,))   # record planned time of arrival at via points
    tg = np.zeros((0,nj))
    infolist = []
    info = namedtuple('mstraj_info', 'slowest segtime axtime clock')

    for seg in range(0, ns):
        if verbose:
            print('------------------- segment %d\n' % (seg,))

        # set the blend time, just half an interval for the first segment

        tacc = Tacc[seg]

        tacc = math.ceil(tacc / dt) * dt
        tacc2 = math.ceil(tacc / 2 / dt) * dt
        if seg == 0:
            taccx = tacc2
        else:
            taccx = tacc

        # estimate travel time
        #    could better estimate distance travelled during the blend
        q_next = viapoints[seg,:]    # current target
        dq = q_next - q_prev    # total distance to move this segment

        ## probably should iterate over the next section to get qb right...
        # while 1
        #   qd_next = (qnextnext - qnext)
        #   tb = abs(qd_next - qd) ./ qddmax;
        #   qb = f(tb, max acceleration)
        #   dq = q_next - q_prev - qb
        #   tl = abs(dq) ./ qdmax;

        if qdmax is not None:
            # qdmax is specified, compute slowest axis

            qb = taccx * qdmax / 2       # distance moved during blend
            tb = taccx

            # convert to time
            tl = abs(dq) / qdmax
            #tl = abs(dq - qb) / qdmax
            tl = np.ceil(tl / dt) * dt

            # find the total time and slowest axis
            tt = tb + tl
            slowest = np.argmax(tt)
            tseg = tt[slowest]

            infolist.append(info(slowest, tseg, tt, clock))

            # best if there is some linear motion component
            if tseg <= 2*tacc:
                tseg = 2 * tacc

        elif tsegment is not None:
            # segment time specified, use that
            tseg = tsegment[seg]
            slowest = math.nan

        # log the planned arrival time
        arrive[seg] = clock + tseg
        if seg > 0:
            arrive[seg] += tacc2

        if verbose:
            print('seg %d, slowest axis %d, time required %.4g\n' % (seg, slowest, tseg))

        ## create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        qb = jtraj(q0, q_prev + tacc2 * qd, np.arange(0, taccx, dt), qd0=qd_prev, qd1=qd).q
        tg = np.vstack([tg, qb[1:,:]])

        clock = clock + taccx     # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in np.arange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0 = (1 - s) * q_prev + s * q_next       # linear step
            tg = np.vstack([tg, q0])
            clock += dt

        q_prev = q_next    # next target becomes previous target
        qd_prev = qd

    # add the final blend
    qb = jtraj(q0, q_next, np.arange(0, tacc2, dt), qd0=qd_prev, qd1=qdf).q
    tg = np.vstack([tg, qb[1:,:]])

    #print(info)

    infolist.append(info(None, tseg, None, clock))
    
    return namedtuple('mstraj', 't q arrive info via')(dt * np.arange(0, tg.shape[0]), tg, arrive, infolist, viapoints)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    path = np.array([[10, 10], [10, 60], [80, 80], [50, 10]])
          
    out = mstraj(path, dt=0.1, tacc=10, qdmax=2.5) # extra=True)
    #print(out.q)
    
    plt.figure()
    plt.plot(out.t, out.q)
    plt.show() 
        
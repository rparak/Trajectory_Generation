import numpy as np
import matplotlib.pyplot as plt
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
import Lib.Trajectory.Utilities as Utilities

import math

def lerp(point0, v0, t0, t1, step=1):
    # Generate a series of timestep
    t = np.arange(t0, t1+step,step)#makit one column
    # Calculate velocity
    v = v0
    #time shift
    Ti = t0
    #equation
    s = point0 + v*(t-Ti)
    v = np.ones(t.size)*v
    a = np.zeros(t.size)
    return (t,s,v,a)

def parab(p0, v0, v1, t0, t1, step=1):
    # Generate a series of timestep
    t = np.arange(t0, t1+step,step)
    #calculate acceleration
    a = (v1-v0)/(t1-t0)
    #time shift
    Ti=t0
    # equation
    s = p0 + v0*(t-Ti) +0.5*a*(t-Ti)**2
    v = v0 + a*(t-Ti)
    a = np.ones(t.size)*a
    return (t,s,v,a)

def lspb(via, tb, qmax):

    v_seg = []; dur_n = []
    dt = 0.01; dur = 0.0
    for i in range(len(via) - 1):
        tacc = math.ceil(tb / dt) * dt
        tacc2 = math.ceil(tacc / 2 / dt) * dt

        if i == 0:
            taccx = tacc2
        else:
            taccx = tacc

        dq = via[i + 1] - via[i]

        tb = taccx

        tl = abs(dq) / qmax
        tl = np.ceil(tl / dt) * dt

        tt = tb + tl
        tseg = tt

        # best if there is some linear motion component
        if tseg <= 2*tacc:
            tseg = 2*tacc

        # linear velocity from qprev to qnext
        qd = dq / tseg

        dur += (via[1]-via[0])/qd

        dur_n.append(dur)
        v_seg.append(qd)
        
    print(v_seg)
    #=====CALCULATE-TIMING-EACH-VIA=====
    T_via=np.zeros(via.size)
    T_via[0]=0.0
    for i in range(1,len(via)-1):
        T_via[i]=T_via[i-1]+dur_n[i-1]
    T_via[-1]=T_via[-2]+dur_n[-1]
    print(T_via)

    #print(via[0] + qd * tb)
    # ...
    P_Cls = Utilities.Polynomial_Profile_Cls(0.01)
    (s, s_dot, s_ddot) = P_Cls.Generate(np.array([via[0], 0.0, 0.0]), np.array([via[0] + v_seg[0] * tb, v_seg[0], 0.0]), T_via[0]-tb, T_via[0]+tb)
    time    = P_Cls.t
    pos     = s
    speed   = s_dot

    (s, s_dot, s_ddot) = P_Cls.Generate(np.array([pos[-1], v_seg[0], 0.0]), np.array([pos[-1] + v_seg[0]*((T_via[1]-tb) - (T_via[0]+tb)), v_seg[0],  0.0]), T_via[0]+tb, T_via[1]-tb)
    time    = np.concatenate((time, P_Cls.t))
    pos     = np.concatenate((pos, s))
    speed   = np.concatenate((speed, s_dot))

    print(pos[-1], via[1] + v_seg[1] * tb)
    print(T_via[1]-tb, T_via[1]+tb)
    (s, s_dot, s_ddot) = P_Cls.Generate(np.array([pos[-1], v_seg[0], 0.0]), np.array([via[1] + v_seg[1] * tb, v_seg[1],  0.0]), T_via[1]-tb, T_via[1]+tb)
    time    = np.concatenate((time, P_Cls.t))
    pos     = np.concatenate((pos, s))
    speed   = np.concatenate((speed, s_dot))

    return(None,T_via,time,pos,speed)

via = np.asarray([10,60,80,10])
t_blend = 2.0
q_max = 5.0

res=lspb(via, t_blend, q_max)

plt.plot(res[1],via,'--')
plt.plot(res[2],res[3], '-')
plt.show()
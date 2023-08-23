import numpy as np
import matplotlib.pyplot as plt
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
import Lib.Trajectory.Utilities as Utilities

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

def lspb(via,dur,tb):
    #1. It must start and end at the first and last waypoint respectively with zero velocity
    #2. Note that during the linear phase acceleration is zero, velocity is constant and position is linear in time
    # if acc.min < 0 :
    #     print('acc must bigger than 0')
    #     return 0
    if ((via.size-1) != dur.size):
        print('duration must equal to number of segment which is via-1')
        return 0
    if (via.size <2):
        print('minimum of via is 2')
        return 0
    if (via.size != (tb.size)):
        print('acc must equal to number of via')
        return 0
    
    #=====CALCULATE-VELOCITY-EACH-SEGMENT=====
    v_seg=np.zeros(dur.size)
    for i in range(0,len(via)-1):
        # s = v * t -> t = s / v
        #print((via[i+1]-via[i])/2.0)
        #v_seg[i]=(via[i+1]-via[i])/(np.sum(dur) - 0.0 - 2*dur[i])
        v_seg[i]=(via[i+1]-via[i])/dur[i]

    print(v_seg)

    #=====CALCULATE-ACCELERATION-EACH-VIA=====
    a_via=np.zeros(via.size)
    a_via[0]=(v_seg[0]-0)/tb[0]
    for i in range(1,len(via)-1):
        a_via[i]=(v_seg[i]-v_seg[i-1])/tb[i]
    a_via[-1]=(0-v_seg[-1])/tb[-1]
    #print(a_via)

    #=====CALCULATE-TIMING-EACH-VIA=====
    T_via=np.zeros(via.size)
    T_via[0]=0.0
    for i in range(1,len(via)-1):
        T_via[i]=T_via[i-1]+dur[i-1]
    T_via[-1]=T_via[-2]+dur[-1]
    print(T_via)

    # s = v*t
    # 
    # t = s/v

    # ...
    P_Cls = Utilities.Polynomial_Profile_Cls(0.01)
    (s, s_dot, s_ddot) = P_Cls.Generate(via[0], via[0] + v_seg[0] * tb[0], 0.0, v_seg[0], 0.0, 0.0, T_via[0]-tb[0], T_via[0]+tb[0])
    time    = P_Cls.t
    pos     = s
    speed   = s_dot

    # linear
    t,s,v,_ = lerp(pos[-1], v_seg[0],T_via[0]+tb[1],T_via[1]-tb[2],0.01)
    time    = np.concatenate((time,t))
    pos     = np.concatenate((pos,s))
    speed   = np.concatenate((speed,v))

    (s, s_dot, s_ddot) = P_Cls.Generate(pos[-1], via[1] + v_seg[1] * tb[2], v_seg[0], v_seg[1], 0.0, 0.0, T_via[1]-tb[2], T_via[1]+tb[2])
    time    = np.concatenate((time, P_Cls.t))
    pos     = np.concatenate((pos, s))
    speed   = np.concatenate((speed, s_dot))

    # linear
    t,s,v,_ = lerp(pos[-1], v_seg[1],T_via[1]+tb[2],T_via[2]-tb[3],0.01)
    time    = np.concatenate((time,t))
    pos     = np.concatenate((pos,s))
    speed   = np.concatenate((speed,v))

    (s, s_dot, s_ddot) = P_Cls.Generate(pos[-1], via[2] + v_seg[2] * tb[3], v_seg[1], v_seg[2], 0.0, 0.0, T_via[2]-tb[3], T_via[2]+tb[3])
    time    = np.concatenate((time, P_Cls.t))
    pos     = np.concatenate((pos, s))
    speed   = np.concatenate((speed, s_dot))

    t,s,v,_ = lerp(pos[-1], v_seg[-1],T_via[-2]+tb[-2],T_via[-1]-tb[-1],0.01)
    time    = np.concatenate((time,t))
    pos     = np.concatenate((pos,s))
    speed   = np.concatenate((speed,v))

    print(T_via[-1]-tb[-1], T_via[-1]+tb[-1])
    # ...
    (s, s_dot, s_ddot) = P_Cls.Generate(pos[-1], via[-1], v_seg[-1], 0.0, 0.0, 0.0, T_via[-1]-tb[-1], T_via[-1]+tb[-1])
    time    = np.concatenate((time, P_Cls.t))
    pos     = np.concatenate((pos, s))
    speed   = np.concatenate((speed, s_dot))

    return(v_seg,a_via,T_via,time,pos,speed)

via = np.asarray([10,60,80,10])
dur = np.asarray([1.0,1.0,1.0])*5.0
tb = np.array([1.0, 1.0, 1.0, 1.0]) * 1.0

res=lspb(via,dur,tb)

plt.plot(res[2],via,'o--')
plt.plot(res[3],res[4], '-')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def generate_trapezoidal_profile(q0, qf, V=None):
    times = np.arange(0, 10)
    T = max(times)

    V = ((qf - q0) / T) * 1.5
    tb = (q0 - qf + V * T) / V
    a = V / tb

    p = []
    pd = []
    pdd = []
    for t in times:
        if t <= 0:
            pk = q0
            pdk = 0
            pddk = 0
        elif t <= tb:
            # initial blend
            pk = q0 + a / 2 * t**2
            pdk = a * t
            pddk = a
        elif t <= (T - tb):
            # linear motion
            pk = (qf + q0 - V * T) / 2 + V * t
            pdk = V
            pddk = 0
        elif t <= T:
            # final blend
            pk = qf - a / 2 * T**2 + a * T * t - a / 2 * t**2
            pdk = a * T - a * t
            pddk = -a
        else:
            #print(t)
            pk = qf
            pdk = 0
            pddk = 0
        p.append(pk)
        pd.append(pdk)
        pdd.append(pddk)
    
    return times, p, pd, pdd

times, positions, velocities, accelerations = generate_trapezoidal_profile(0.157, 1.57)

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(times, positions)
plt.title('Position Profile')
plt.xlabel('Time')
plt.ylabel('Position')

plt.subplot(3, 1, 2)
plt.plot(times, velocities)
plt.title('Velocity Profile')
plt.xlabel('Time')
plt.ylabel('Velocity')

plt.subplot(3, 1, 3)
plt.plot(times, accelerations)
plt.title('Acceleration Profile')
plt.xlabel('Time')
plt.ylabel('Acceleration')

plt.tight_layout()
plt.show()

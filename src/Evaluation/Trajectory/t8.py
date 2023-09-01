import numpy as np

P = np.asarray([10,60,80,10])
t = np.asarray([1.0,1.0,1.0])*5.0

v = np.zeros(t.size, dtype=np.float32)
for i, (P_i, P_ii, t_i) in enumerate(zip(P, P[1::], t)):
    v[i] = (P_ii - P_i)/t_i

print(v)
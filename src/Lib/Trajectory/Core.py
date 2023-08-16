# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Trajectory/Core
import Lib.Trajectory.Profile

class Multi_Point_Cls(object):
    def __init__(self, delta_time: float, t_blend: float, v_max: float) -> None:
        self.__dt = delta_time
        self.__t_blend = t_blend
        self.__v_max = v_max
        
    def Generate(self, P: tp.List[tp.List[float]]):
        """
        Description:
            ...

        Args:
            (1) P [Vector<float> mx1]: Input control points to be used for trajectory generation.
                                        Note:
                                        Where m is the number of points.
        """
        
        # ....
        n = P[1::].size
        T = np.array([self.__t_blend] * n, dtype=np.float32)

        # ...
        s_previous = P[0]; s_dot_previous = np.array([0.0], dtype=np.float32)
        s = np.empty((0,1), dtype=np.float32)
        for _, P_i in enumerate(P[1::]):
            pass

        return (np.linspace(Lib.Trajectory.Profile.CONST_T_0, Lib.Trajectory.Profile.CONST_T_1, s.size), s)
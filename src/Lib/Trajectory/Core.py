# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Time x ∈ [0: The starting value of the sequence, 
#           1: The end value of the sequence] {0.0 <= x <= 1.0}
CONST_T_0 = 0.0
CONST_T_1 = 1.0

class Polynomial_Profile_Cls(object):
    def __init__(self, s_0: tp.List[float], s_1: tp.List[float], N: int) -> None:
        # The value of the time must be within the interval: 
        #   0.0 <= t <= 1.0
        self.__t = np.linspace(CONST_T_0, CONST_T_1, N)

        self.__s_0 = s_0
        self.__s_1 = s_1

        # Initialization of other class parameters.
        self.__s = np.zeros((N, s_0.size), dtype=np.float32)
        self.__s_dot  = self.__s.copy()
        self.__s_ddot = self.__s.copy()

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Time.
                                                Note:
                                                    Where n is the number of points.
        """
                
        return self.__t
    
    @property
    def N(self) -> int:
        """
        Description:
           Get the number of time points of the trajectory.
        
        Returns:
            (1) parameter [int]: Number of time points.
        """
                
        return self.__t.shape[0]
    
    @property
    def s(self): 
        return self.__s
    
    @property
    def s_dot(self):
        return self.__s_dot
    
    @property
    def s_ddot(self):
        return self.__s_ddot
    
    def __Quintic_Polynomial(self, s_0, s_1):
        s_dot_0 = 0.0; s_dot_1 = 0.0

        A = 6.0 * (s_1 - s_0) - 3 * (s_dot_1 + s_dot_0)
        B = -15.0 * (s_1 - s_0) + (8 * s_dot_0 + 7 * s_dot_1)
        C = 10.0 * (s_1 - s_0) - (6 * s_dot_0 + 4 * s_dot_1)
        E = s_dot_0
        F = s_0

        # ...
        t = np.array([self.__t ** 5, self.__t ** 4, self.__t ** 3, 
                      self.__t ** 2, self.__t, np.ones(self.N)], dtype=np.float32).T
        
        return (t @ np.array([  A,       B,        C,      0.0,       E,   F], dtype=np.float32),
                t @ np.array([0.0, 5.0 * A,  4.0 * B,  3.0 * C,     0.0,   E], dtype=np.float32),
                t @ np.array([0.0,     0.0, 20.0 * A, 12.0 * B, 6.0 * C, 0.0], dtype=np.float32))

    def Generate(self):
        for i, (s_0_i, s_1_i) in enumerate(zip(self.__s_0, self.__s_1)):
            # ...
            (s, s_dot, s_ddot) = self.__Quintic_Polynomial(np.array(s_0_i, dtype=np.float32), 
                                                           np.array(s_1_i, dtype=np.float32))
            # ...
            self.__s[:, i] = s; self.__s_dot[:, i] = s_dot; self.__s_ddot[:, i] = s_ddot
    
class Trapezoidal_Profile_Cls(object):
    def __init__(self) -> None:
        pass

    @property
    def s(self):
        return None
    
    @property
    def s_dot(self):
        return None
    
    @property
    def s_ddot(self):
        return None
    
    def Generate(self):
        return True
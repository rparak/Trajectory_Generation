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

class Polynomial_Cls(object):
    """
    Description:
        The specific class for generating the polynomial trajectory of degree 5 (quintic) from input constraints.

        Note:
            A polynomial of degree 5 (quintic) was chosen to obtain the acceleration.

    Initialization of the Class:
        Args:
            (1) N [int]: The number of time points used to generate the polynomial trajectory.

        Example:
            Initialization:
                # Assignment of the variables.
                N = 100

                # Initialization of the class.
                Cls = Polynomial_Cls(Box)

            Features:
                # Properties of the class.
                Cls.t
                ...
                Cls.N

                # Functions of the class.
                Cls.Generate([0.0, 0.0, 0.0], [1.57, 0.0, 0.0])
    """
            
    def __init__(self, N: int) -> None:
        # The value of the time must be within the interval: 
        #   0.0 <= t <= 1.0
        self.__t = np.linspace(CONST_T_0, CONST_T_1, N)

        # Obtain the modified polynomial matrix of degree 5.
        self.__X = self.__Quintic_Polynomial()

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Normalized time.
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
    
    def __Quintic_Polynomial(self) -> tp.List[tp.List[float]]:
        """
        Descrtiption:
            Obtain the modified polynomial matrix of degree 5.

            A polynomial of degree 5 (quintic) is defined as follows:
                s(t) = c_{0} + c_{1}*t + c_{2}*t^2 + c_{3}*t^3 + c_{4}*t^4 + c_{5}*t^5

            The quintic polynomial can be expressed by a system of 6 equations. These equations 
            can be converted into a matrix, which we have called X.

                X = [[1.0, t_0,    t_0**2,       t_0**3,        t_0**4,        t_0**5],
                     [0.0, 1.0, 2.0 * t_0, 3.0 * t_0**2,  4.0 * t_0**3,  5.0 * t_0**4],
                     [0.0, 0.0,       2.0,    6.0 * t_0, 12.0 * t_0**2, 20.0 * t_0**3],
                     [1.0, t_f,    t_f**2,       t_f**3,        t_f**4,        t_f**5],
                     [0.0, 1.0, 2.0 * t_f, 3.0 * t_f**2,  4.0 * t_f**3,  5.0 * t_f**4],
                     [0.0, 0.0,       2.0,    6.0 * t_f, 12.0 * t_f**2, 20.0 * t_f**3]]

            The X matrix has been modified to make the calculation as fast as possible.
                Note:
                    t_0 is always equal to 0.0, and t_f is always equal to 1.0 because we use 
                    the normalized time, t.

        Returns:
            (1) parameter [Vector<float> 6x6]: Modified polynomial matrix of degree 5.
        """
        
        return np.array([[1.0, 0.0, 0.0, 0.0,  0.0,  0.0],
                         [0.0, 1.0, 0.0, 0.0,  0.0,  0.0],
                         [0.0, 0.0, 2.0, 0.0,  0.0,  0.0],
                         [1.0, 1.0, 1.0, 1.0,  1.0,  1.0],
                         [0.0, 1.0, 2.0, 3.0,  4.0,  5.0],
                         [0.0, 0.0, 2.0, 6.0, 12.0, 20.0]], dtype=np.float32)

    def Generate(self, s_0: tp.List[float], s_f: tp.List[float]) -> tp.Tuple[tp.List[float], tp.List[float], 
                                                                             tp.List[float]]:
        """
        Description:
            Generate position, velocity, and acceleration polynomial trajectories of degree 5.

        Args:
            (1, 2) s_0, s_f [Vector<float> 1x3]: Initial and final constraint configuration.
                                                 Note:
                                                    s_{..}[0] - Position.
                                                    s_{..}[1] - Velocity.
                                                    s_{..}[2] - Acceleration.

        Returns:
            (1 - 3) parameter [Vector<float> 1xn]: Position, velocity and acceleration polynomial trajectory of degree 5.
        """

        # Initialization of the output varliables.
        s = np.zeros(self.N, dtype=np.float32)
        s_dot = s.copy(); s_ddot = s.copy()

        # Find the coefficients c_{0 .. 5} from the equation below.
        #   Equation:
        #       [c_{0 .. 5}] = X^(-1) * [s_0, s_f, s_dot_0, s_dot_f, s_ddot_0, s_ddot_f]
        C = np.linalg.inv(self.__X) @ np.append(s_0, s_f)
        
        # Analytic expression (position):
        #   s(t) = c_{0} + c_{1}*t + c_{2}*t^2 + c_{3}*t^3 + c_{4}*t^4 + c_{5}*t^5
        for i, C_i in enumerate(C):
            s[:] += (self.__t ** i) * C_i
        
        # Analytic expression (velocity):
        #   s_dot(t) = c_{1} + 2*c_{2}*t + 3*c_{3}*t^2 + 4*c_{4}*t^3 + 5*c_{5}*t^4
        for i, (C_ii, var_i) in enumerate(zip(C[1:], np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)), 
                                             start=1):
            s_dot[:] += (self.__t ** (i - 1)) * C_ii * var_i

        # Analytic expression (acceleration):
        #   s_ddot(t) = 2*c_{2}*t + 6*c_{3}*t^2 + 12*c_{4}*t^3 + 20*c_{5}*t^4
        for i, (C_iii, var_i) in enumerate(zip(C[2:], np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)), 
                                              start=2):
            s_ddot[:] += (self.__t ** (i - 2)) * C_iii * var_i

        return (s, s_dot, s_ddot)
    
class Trapezoidal_Cls(object):   
    """
    Description:
        The specific class for generating the trapezoidal trajectory from input constraints.


    Initialization of the Class:
        Args:
            (1) N [int]: The number of time points used to generate the polynomial trajectory.

        Example:
            Initialization:
                # Assignment of the variables.
                N = 100

                # Initialization of the class.
                Cls = Trapezoidal_Cls(Box)

            Features:
                # Properties of the class.
                Cls.t
                ...
                Cls.N

                # Functions of the class.
                Cls.Generate(0.0, 1.57)
    """
             
    def __init__(self, N: int) -> None:
        # The value of the time must be within the interval: 
        #   0.0 <= t <= 1.0
        self.__t = np.linspace(CONST_T_0, CONST_T_1, N)

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Normalized time.
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

    def Generate(self, s_0: float, s_f: float) -> tp.Tuple[tp.List[float], tp.List[float], 
                                                           tp.List[float]]:
        """
        Description:
            Generate position, velocity, and acceleration trapezoidal trajectories.

        Args:
            (1, 2) s_0, s_f [float]: Initial and final constraint configuration.

        Returns:
            (1 - 3) parameter [Vector<float> 1xn]: Position, velocity and acceleration trapezoidal trajectory.
        """
        
        # https://bjpcjp.github.io/pdfs/robotics/MR_ch09_trajectory_generation.pdf
        # https://epub.jku.at/obvulihs/download/pdf/5841037?originalFilename=true

        # Initialization of the output varliables.
        s = np.zeros(self.N, dtype=np.float32)
        s_dot = s.copy(); s_ddot = s.copy()

        # ...
        T = self.t[-1] * self.N

        # Set the velocity ...
        #   Note:
        #       The velocity must be withint the interval ...
        #           v > abs(qf - q0) / T
        #           v < 2 * abs(qf - q0) / T
        v = ((s_f - s_0) / T) * 1.5
        
        # ...
        t_a = ((s_0 - s_f) + v * T) / v

        # Express the acceleration with a simple formula.
        a = v / t_a

        # ...
        for i, t_i in enumerate(self.__t * self.N):
            if t_i <= t_a:
                # Phase 1: Acceleration.
                s[i] = s_0 + 0.5 * a * t_i**2
                s_dot[i] = a * t_i
                s_ddot[i] = a
            elif t_i <= T - t_a:
                # Phase 2: Constant velocity.
                s[i] = (s_f + s_0 - v * T) * 0.5 + v * t_i
                s_dot[i] = v
                s_ddot[i] = 0.0
            elif t_i <= T:
                # Phase 3: Deceleration.
                s[i] = s_f - 0.5 * a * T**2 + a * T * t_i - 0.5*a * t_i**2
                s_dot[i] = a * (T - t_i)
                s_ddot[i] = -a

        return (s, s_dot, s_ddot)
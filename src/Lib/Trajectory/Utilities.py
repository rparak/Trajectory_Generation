"""
## =========================================================================== ## 
MIT License
Copyright (c) 2023 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: Profile.py
## =========================================================================== ## 
"""

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

class Polynomial_Profile_Cls(object):
    """
    Description:
        The specific class for generating the polynomial trajectory of degree 5 (quintic) from input constraints.

        Note:
            A polynomial of degree 5 (quintic) was chosen to obtain the acceleration.

    Initialization of the Class:
        Args:
            (1) delta_time [fluat]: The difference (spacing) between the time values.

        Example:
            Initialization:
                # Assignment of the variables.
                delta_time = 0.01

                # Initialization of the class.
                Cls = Polynomial_Profile_Cls(delta_time)

            Features:
                # Properties of the class.
                Cls.t
                ...
                Cls.N

                # Functions of the class.
                Cls.Generate([0.0, 0.0, 0.0], [1.57, 0.0, 0.0], 0.0, 1.0)
    """
            
    def __init__(self, delta_time: float) -> None:
        # The difference (spacing) between the time values.
        self.__delta_time = delta_time

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
            Get evenly spaced time values at the following interval, see below:
                t_0 <= t <= t_f

        Returns:
            (1) parameter [Vector<float> 1xn]: Time values.
                                                Note:
                                                    Where n is the number of time values.
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
                
        return self.__t.size
    
    def __Quintic_Polynomial(self, t_0: float, t_f: float) -> tp.List[tp.List[float]]:
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

        Args:
            (1, 2) t_0, t_f [float]: Initial and final time constant.

        Returns:
            (1) parameter [Vector<float> 6x6]: Modified polynomial matrix of degree 5.
        """
        
        return np.array([[1.0, t_0,    t_0**2,       t_0**3,        t_0**4,        t_0**5],
                         [0.0, 1.0, 2.0 * t_0, 3.0 * t_0**2,  4.0 * t_0**3,  5.0 * t_0**4],
                         [0.0, 0.0,       2.0,    6.0 * t_0, 12.0 * t_0**2, 20.0 * t_0**3],
                         [1.0, t_f,    t_f**2,       t_f**3,        t_f**4,        t_f**5],
                         [0.0, 1.0, 2.0 * t_f, 3.0 * t_f**2,  4.0 * t_f**3,  5.0 * t_f**4],
                         [0.0, 0.0,       2.0,    6.0 * t_f, 12.0 * t_f**2, 20.0 * t_f**3]], dtype=np.float32)
 
    def Generate(self, s_0: tp.List[float], s_f: tp.List[float], t_0, t_f) -> tp.Tuple[tp.List[float], tp.List[float], 
                                                                                       tp.List[float]]:
        """
        Description:
            A function to generate position, velocity, and acceleration polynomial trajectories of degree 5.

        Args:
            (1, 2) s_0, s_f [Vector<float> 1x3]: Initial and final constraint configuration.
                                                 Note:
                                                    s_{..}[0] - Position.
                                                    s_{..}[1] - Velocity.
                                                    s_{..}[2] - Acceleration.
            (3, 4) t_0, t_f [float]: Initial and final time constant.

        Returns:
            (1 - 3) parameter [Vector<float> 1xn]: Position, velocity and acceleration polynomial trajectory of degree 5.
        """

        # Get evenly distributed time values in a given interval.
        #   t_0 <= t <= t_f
        self.__t = np.arange(t_0, t_f + self.__delta_time, self.__delta_time)

        # Obtain the modified polynomial matrix of degree 5.
        self.__X = self.__Quintic_Polynomial(t_0, t_f)

        # Initialization of the output varliables.
        s = np.zeros(self.__t.size, dtype=np.float32)
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
    
class Trapezoidal_Profile_Cls(object):   
    """
    Description:
        The specific class for generating the trapezoidal trajectory from input constraints.

    Initialization of the Class:
        Args:
            (1) delta_time [fluat]: The difference (spacing) between the time values.

        Example:
            Initialization:
                # Assignment of the variables.
                delta_time = 0.01

                # Initialization of the class.
                Cls = Trapezoidal_Profile_Cls(delta_time)

            Features:
                # Properties of the class.
                Cls.t
                ...
                Cls.N

                # Functions of the class.
                Cls.Generate(0.0, 1.57)
    """
             
    def __init__(self, delta_time: float) -> None:
        # The difference (spacing) between the time values.
        self.__delta_time = delta_time

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
            Get evenly spaced time values at the following interval, see below:
                t_0 <= t <= t_f

        Returns:
            (1) parameter [Vector<float> 1xn]: Time values.
                                                Note:
                                                    Where n is the number of time values.
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
                
        return self.__t.size
        
    def Generate(self, s_0: float, s_f: float, t_0: float, t_f: float) -> tp.Tuple[tp.List[float], tp.List[float], 
                                                                                   tp.List[float]]:
        """
        Description:
            A function to generate position, velocity, and acceleration trapezoidal trajectories.

        Args:
            (1, 2) s_0, s_f [float]: Initial and final constraint configuration.
                                     Note:
                                        s_{..} - Position.
            (3, 4) t_0, t_f [float]: Initial and final time constant.

        Returns:
            (1 - 3) parameter [Vector<float> 1xn]: Position, velocity and acceleration trapezoidal trajectory.
        """
        
        # Get evenly distributed time values in a given interval.
        #   t_0 <= t <= t_f
        self.__t = np.arange(t_0, t_f + self.__delta_time, self.__delta_time)

        # Initialization of the output varliables.
        s = np.zeros(self.N, dtype=np.float32)
        s_dot = s.copy(); s_ddot = s.copy()

        # Get T as the maximum value of t.
        T = self.t[-1]

        # Calculate the velocity automatically.
        #   Note:
        #       The velocity must be within the interval, see below:
        #           (qf - q0) / T < v < 2 * ((qf - q0) / T)
        v = 1.5 * ((s_f - s_0) / T)
        
        # Time of constant acceleration phase.
        t_a = ((s_0 - s_f) + v * T) / v

        # Express the acceleration with a simple formula.
        a = v / t_a

        # Express the position (s), velocity (s_dot), and acceleration (s_ddot) of a trapezoidal 
        # trajectory.
        for i, t_i in enumerate(self.__t):
            if t_i <= t_a:
                # Phase 1: Acceleration.
                #s[i] = s_0 + 0.5 * a * t_i**2
                s_dot[i] = a * t_i
                s_ddot[i] = a
            elif t_i <= T - t_a:
                # Phase 2: Constant velocity.
                #s[i] = (s_f + s_0 - v * T) * 0.5 + v * t_i
                s_dot[i] = v
                s_ddot[i] = 0.0
            elif t_i <= T:
                # Phase 3: Deceleration.
                #s[i] = s_f - 0.5 * a * T**2 + a * T * t_i - 0.5*a * t_i**2
                s_dot[i] = a * (T - t_i)
                s_ddot[i] = -a

        return (s, s_dot, s_ddot)
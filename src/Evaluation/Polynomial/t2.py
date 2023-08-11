import numpy as np
import matplotlib.pyplot as plt

# Define waypoints (x, y)
waypoints = np.array([[0, 0], [1, 2], [3, 1], [4, 3]])

# Extract x and y values
x = waypoints[:, 0]
y = waypoints[:, 1]

# Calculate time intervals based on the number of waypoints
num_waypoints = waypoints.shape[0]
time_intervals = np.linspace(0, 1, num_waypoints)

# Solve for the coefficients of quintic polynomials for x and y
coeffs_x = np.polyfit(time_intervals, x, 5)
coeffs_y = np.polyfit(time_intervals, y, 5)

# Evaluate the quintic polynomials
num_points = 100
t_interp = np.linspace(0, 1, num_points)
x_interp = np.polyval(coeffs_x, t_interp)
y_interp = np.polyval(coeffs_y, t_interp)

# Plot the original waypoints and the interpolated trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'ro', label='Waypoints')
plt.plot(x_interp, y_interp, 'b-', label='Quintic Polynomial')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quintic Polynomial Trajectory Through Waypoints')
plt.legend()
plt.grid()
plt.show()
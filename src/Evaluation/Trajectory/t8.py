import numpy as np

def f(x):
    # Define your curve function here (e.g., f(x) = x^2)
    return x**2

def curve_length(f, a, b, n):
    # Calculate the length of the curve using the trapezoidal rule
    delta_x = (b - a) / n
    x_values = np.linspace(a, b, n + 1)
    y_values = f(x_values)
    
    length = 0
    for i in range(n):
        length += np.sqrt((x_values[i + 1] - x_values[i])**2 + (y_values[i + 1] - y_values[i])**2)
    
    return length

# Define the interval [a, b] and the number of subdivisions (n)
a = 0
b = 2
n = 100

# Calculate the length of the curve
curve_length_result = curve_length(f, a, b, n)

print(curve_length_result)
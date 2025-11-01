from scipy.integrate import trapezoid
import numpy as np


# Integrate the real and imaginary parts
def trapz_integral_real_imaginary(time_points, values):
    if len(time_points) != len(values):
        raise ValueError("Length of time_points and values must be equal")
    real_part = np.real(values)
    imag_part = np.imag(values)
    real_integral = trapezoid(real_part, time_points)
    imag_integral = trapezoid(imag_part, time_points)
    return real_integral, imag_integral

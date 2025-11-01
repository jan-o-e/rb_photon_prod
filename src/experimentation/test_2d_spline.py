import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import trapezoid


# Define a sample 2D function for testing
def sample_function(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# Define the coarse grid
t1_label = np.linspace(0, 1, 10)
t2_label = np.linspace(0, 1, 10)
t1_grid, t2_grid = np.meshgrid(t1_label, t2_label, indexing="ij")
real_values = sample_function(t1_grid, t2_grid)

# Oversample factor
oversample_factor = 10

# Define finer grids
t1_fine = np.linspace(t1_label[0], t1_label[-1], len(t1_label) * oversample_factor)
t2_fine = np.linspace(t2_label[0], t2_label[-1], len(t2_label) * oversample_factor)

# Interpolate using cubic spline
spline_real = interpolate.RectBivariateSpline(t1_label, t2_label, real_values)
smoothed_real_values_fine = spline_real(t1_fine, t2_fine)

# Compute integrals for the original grid
real_integral_t1 = trapezoid(real_values, x=t1_label, axis=1)
real_final_integral = trapezoid(real_integral_t1, x=t2_label, axis=0)

# Compute integrals for the oversampled grid
real_integral_t1_fine = trapezoid(smoothed_real_values_fine, x=t1_fine, axis=1)
real_final_integral_fine = trapezoid(real_integral_t1_fine, x=t2_fine, axis=0)

# Plotting
plt.figure(figsize=(12, 6))

# Original grid contour
plt.subplot(1, 2, 1)
plt.contourf(t1_grid, t2_grid, real_values, levels=20, cmap="viridis")
plt.title("Original Grid")
plt.xlabel("t1")
plt.ylabel("t2")
plt.colorbar(label="Function Value")

# Oversampled grid contour
plt.subplot(1, 2, 2)
t1_fine_grid, t2_fine_grid = np.meshgrid(t1_fine, t2_fine, indexing="ij")
plt.contourf(
    t1_fine_grid, t2_fine_grid, smoothed_real_values_fine, levels=20, cmap="viridis"
)
plt.title("Oversampled Grid")
plt.xlabel("t1")
plt.ylabel("t2")
plt.colorbar(label="Function Value")

plt.tight_layout()
plt.show()

# Print integral comparison
print(f"Original Grid Integral: {real_final_integral}")
print(f"Oversampled Grid Integral: {real_final_integral_fine}")

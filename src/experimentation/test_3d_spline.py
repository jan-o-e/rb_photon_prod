import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


# Define a test 3D function
def test_function(x, y, z):
    return np.sin(x) * np.sin(y) * np.exp(-z)


# Define the original grid
x = np.linspace(0, np.pi, 10)
y = np.linspace(0, np.pi, 10)
z = np.linspace(0, 1, 10)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Evaluate the test function on the original grid
original_values = test_function(X, Y, Z)

# Define the oversampling factor
oversample_factor = 10

# Create finer grids
x_fine = np.linspace(x[0], x[-1], len(x) * oversample_factor)
y_fine = np.linspace(y[0], y[-1], len(y) * oversample_factor)
z_fine = np.linspace(z[0], z[-1], len(z) * oversample_factor)

# Create a 3D cubic spline interpolator
spline = RegularGridInterpolator((x, y, z), original_values, method="cubic")

# Evaluate the spline on the finer grid
X_fine, Y_fine, Z_fine = np.meshgrid(x_fine, y_fine, z_fine, indexing="ij")
fine_points = np.array([X_fine.ravel(), Y_fine.ravel(), Z_fine.ravel()]).T
smoothed_values_fine = spline(fine_points).reshape(X_fine.shape)

# Compute trapezoidal integrals
# Original grid
original_integral_x = trapezoid(original_values, x, axis=0)
original_integral_xy = trapezoid(original_integral_x, y, axis=0)
original_integral_xyz = trapezoid(original_integral_xy, z, axis=0)

# Oversampled grid
smoothed_integral_x = trapezoid(smoothed_values_fine, x_fine, axis=0)
smoothed_integral_xy = trapezoid(smoothed_integral_x, y_fine, axis=0)
smoothed_integral_xyz = trapezoid(smoothed_integral_xy, z_fine, axis=0)

# Print results
print("Original integral:", original_integral_xyz)
print("Smoothed integral:", smoothed_integral_xyz)

# Plot a slice for comparison
slice_idx = len(z) // 2  # Middle slice along the z-axis

plt.figure(figsize=(12, 6))

# Original contours
plt.subplot(1, 2, 1)
plt.contourf(x, y, original_values[:, :, slice_idx], levels=20, cmap="viridis")
plt.title("Original Grid")
plt.colorbar()

# Oversampled contours
plt.subplot(1, 2, 2)
plt.contourf(
    x_fine,
    y_fine,
    smoothed_values_fine[:, :, slice_idx * oversample_factor],
    levels=20,
    cmap="viridis",
)
plt.title("Oversampled Grid")
plt.colorbar()

plt.tight_layout()
plt.show()

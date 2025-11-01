import numpy as np
import os
import matplotlib.pyplot as plt


def generate_time_correlator_eval(
    _n_off_diag_correlator_eval,
    frac_fine=2 / 3,
    fine_interval_frac=(0.1, 0.6),
    coarse_interval_frac=(0, 0.8),
    t_vst=0.5,
    t_rot=0.25,
):
    """
    Generate a list of time correlator evaluation points with specified sampling characteristics.

    Parameters:
    _n_off_diag_correlator_eval (int): Total number of points to generate.
    frac_fine (float): Fraction of points to be sampled finely in the fine_interval.
    fine_interval_frac (tuple): Fractions of t_vst for fine sampling (start, end).
    coarse_interval_frac (tuple): Fractions of t_vst + t_rot for coarse sampling (start, end).
    t_vst (float): Additional point to be included in the result.
    t_rot (float): Additional point to be included in the result.

    Returns:
    numpy.ndarray: Sorted array of evaluation points.

    Raises:
    ValueError: If any of the interval bounds exceed t_vst + t_rot.
    """

    # Compute the actual intervals based on fractions
    fine_interval = (fine_interval_frac[0] * t_vst, fine_interval_frac[1] * t_vst)
    # Don't sample all the way to the end of a time bin (there will be no correlation) and it causes a problem with list indexing
    coarse_interval = (
        coarse_interval_frac[0] * (t_vst + t_rot),
        coarse_interval_frac[1] * (t_vst + t_rot - 0.01 * t_rot),
    )

    # Check if any interval bounds exceed t_vst + t_rot
    if fine_interval[1] > t_vst + t_rot or coarse_interval[1] > t_vst + t_rot:
        raise ValueError("Interval bounds exceed t_vst + t_rot.")

    # Compute the number of points for each segment
    n_fine_points = int(frac_fine * _n_off_diag_correlator_eval)
    n_coarse_points = _n_off_diag_correlator_eval - n_fine_points

    # Generate finely sampled points in the fine_interval
    fine_points = np.linspace(fine_interval[0], fine_interval[1], n_fine_points)

    # Generate coarsely sampled points in the coarse_interval
    coarse_points = np.linspace(coarse_interval[0], coarse_interval[1], n_coarse_points)

    # Combine the points and ensure there are no duplicates
    all_points = np.unique(np.concatenate((fine_points, coarse_points)))

    # Include a single point at t=t_final, if not already included
    final_time = float(t_vst + t_rot)
    if final_time not in all_points:
        all_points = np.append(all_points, final_time)

    # Sort the final list of points
    t_correlator_eval = np.sort(all_points)
    return t_correlator_eval, final_time


def plot_density_matrix_correlations(
    H_sim_time_list,
    exp_values_diag_zero,
    exp_values_diag_one,
    t_correlator_eval,
    coherence_01,
    save_dir,
    n_start,
    length_stirap,
    bfield_split,
):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Plot for rho_00
    ax[0, 0].plot(H_sim_time_list[0], np.real(exp_values_diag_zero), label="Real")
    ax[0, 0].plot(H_sim_time_list[0], np.imag(exp_values_diag_zero), label="Imaginary")
    ax[0, 0].set_title("rho_00")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Correlation")
    ax[0, 0].legend()

    # Plot for rho_11
    ax[0, 1].plot(H_sim_time_list[2], np.real(exp_values_diag_one), label="Real")
    ax[0, 1].plot(H_sim_time_list[2], np.imag(exp_values_diag_one), label="Imaginary")
    ax[0, 1].set_title("rho_11")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Correlation")
    ax[0, 1].legend()

    # Plot for rho_10
    ax[1, 0].plot(t_correlator_eval, np.real(np.conj(coherence_01)), label="Real")
    ax[1, 0].plot(t_correlator_eval, np.imag(np.conj(coherence_01)), label="Imaginary")
    ax[1, 0].set_title("rho_10")
    ax[1, 0].set_xlabel("Time")
    ax[1, 0].set_ylabel("Correlation")
    ax[1, 0].legend()

    # Plot for rho_01
    ax[1, 1].plot(t_correlator_eval, np.real(coherence_01), label="Real")
    ax[1, 1].plot(t_correlator_eval, np.imag(coherence_01), label="Imaginary")
    ax[1, 1].set_title("rho_01")
    ax[1, 1].set_xlabel("Time")
    ax[1, 1].set_ylabel("Correlation")
    ax[1, 1].legend()

    plt.tight_layout()

    # Define the path where you want to save
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    save_path = os.path.join(src_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)

    plot_path = os.path.join(
        save_path,
        f"density_matrix_n1_correlations_n_start_{n_start}_vst{length_stirap}_b{bfield_split}.pdf",
    )
    plt.savefig(plot_path)
    plt.close(fig)


def plot_n2_density_matrix_correlation(
    t1_label,
    t2_label,
    real_values,
    imag_values,
    _element,
    save_dir,
    n_start,
    length_stirap,
    bfield_split,
    two_photon_det,
):
    X, Y = np.meshgrid(t1_label, t2_label)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot real part
    c1 = ax[0].contourf(X, Y, real_values, cmap="viridis")
    fig.colorbar(c1, ax=ax[0], label="Real Correlation Value")
    ax[0].set_title(f"Real Correlation Values for N=2 Matrix Element: {_element}")
    ax[0].set_xlabel("t1")
    ax[0].set_ylabel("t2")
    ax[0].set_xticks(np.linspace(min(t1_label), max(t1_label), num=5))
    ax[0].set_yticks(np.linspace(min(t2_label), max(t2_label), num=5))

    # Plot imaginary part
    c2 = ax[1].contourf(X, Y, imag_values, cmap="plasma")
    fig.colorbar(c2, ax=ax[1], label="Imaginary Correlation Value")
    ax[1].set_title(f"Imaginary Correlation Values for N=2 Matrix Element: {_element}")
    ax[1].set_xlabel("t1")
    ax[1].set_ylabel("t2")
    ax[1].set_xticks(np.linspace(min(t1_label), max(t1_label), num=5))
    ax[1].set_yticks(np.linspace(min(t2_label), max(t2_label), num=5))

    plt.tight_layout()

    # Define the path where you want to save
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    save_path = os.path.join(src_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)

    plot_path = os.path.join(
        save_path,
        f"n2_correlations_nstart{n_start}_vstlength{length_stirap}_b{bfield_split}_twophotdet"
        f"{np.round(two_photon_det[0],3)}_{np.round(two_photon_det[1],3)}{_element}.pdf",
    )
    plt.savefig(plot_path)
    plt.close(fig)

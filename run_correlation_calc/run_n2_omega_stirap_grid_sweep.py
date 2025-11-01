from run_correlation_calc.run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 32
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst=1
    omega_3_range = np.linspace(-100, +100, 20)+1139
    omega_4_range = np.linspace(-100, +100, 20)+1459

    for omega1 in omega_3_range:
        for omega2 in omega_4_range:
            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/n_1/omega_stirap_grid/",
                n_sim_steps=sim_steps,
                n_photons=2,
                b_field="0p07",
                _n_start=n_start,
                _len_stirap=t_vst,
                _rot_omega_3=omega1,
                _rot_omega_4=omega2,
                _two_photon_det_3=0,
                _two_photon_det_4=0,
                _plot=False)

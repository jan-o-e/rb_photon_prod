from run_correlation_calc.run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 32
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst=1
    omega_3_range = np.linspace(-300, +300, 24)+1400
    omega_4_range = np.linspace(-30, +50, 10)+50

    for omega1 in omega_3_range:
        for omega2 in omega_4_range:
            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/far_detuned/n_1/omega_rot_stirap_grid/",
                n_sim_steps=sim_steps,
                n_photons=1,
                b_field="0p07",
                _n_start=n_start,
                _len_stirap=t_vst,
                _omega_stirap_early=omega2,
                _omega_stirap_late=omega2,
                _vst_ramp_up=0.7,
                _vst_ramp_down=0.3,
                _shape_stirap=pulse_shape,
                _rot_omega_3=omega1,
                _two_photon_det_3=0,
                _two_photon_det_4=0,
                _plot=False)

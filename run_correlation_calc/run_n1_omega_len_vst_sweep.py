from run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 48
    pulse_shape = "flattop_blackman"
    n_start=2

    omega_range = np.linspace(10, 120, 12)
    t_list = np.linspace(0.2, 1.5, 12)
    for omega in omega_range:

        for t_vst in t_list:
            rise_time = 0.25*t_vst
            fall_time = 0.1*t_vst

            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/far_detuned/n_1/vst_length_omega_grid",
                n_sim_steps=sim_steps,
                n_photons=1,
                _n_start=n_start,
                _omega_stirap_early=omega,
                _omega_stirap_late=omega,
                b_field="0p07",
                _shape_stirap=pulse_shape,
                _vst_ramp_up=rise_time,
                _vst_ramp_down=fall_time,
                _len_stirap=t_vst,
                _two_photon_det_2=0,
                _two_photon_det_4=0,
                _plot=False,
                _spont_emission=True)

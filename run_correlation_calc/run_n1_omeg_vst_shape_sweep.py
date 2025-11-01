from run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 48
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst=1
    rise_time_range = np.linspace(0.05, 0.7, 20)
    fall_time_range = np.linspace(0.1, 0.3,3)

    for t_rise in rise_time_range:
        for t_fall in fall_time_range:
            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/far_detuned/n_1/omegavst_bm_risetime_grid/",
                n_sim_steps=sim_steps,
                n_photons=1,
                b_field="0p07",
                _n_start=n_start,
                _len_stirap=t_vst,
                _shape_stirap=pulse_shape,
                _vst_ramp_up=t_rise,
                _vst_ramp_down=t_fall,
                _omega_stirap_early=50,
                _omega_stirap_late=50,
                _two_photon_det_3=0,
                _two_photon_det_4=0,
                _plot=False)

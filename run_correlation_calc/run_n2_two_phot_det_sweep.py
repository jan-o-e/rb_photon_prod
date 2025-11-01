from run_correlation_calc.run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 32
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst =1
    det_1_range = np.linspace(-4, 4, 20)
    det_2_range = np.linspace(-4, 4, 20)

    for det_1 in det_1_range:
        for det_2 in det_2_range:
            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/n_1/det_grid/",
                n_sim_steps=sim_steps,
                n_photons=2,
                b_field="0p07",
                _n_start=n_start,
                _len_stirap=t_vst,
                _two_photon_det_3=det_1,
                _two_photon_det_4=det_2,
                _plot=False)

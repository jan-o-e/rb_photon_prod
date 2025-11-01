from run_correlation_calc import run_simulation
import numpy as np

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 48
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst=1
    b_field_range = ["0", "0p01", "0p02", "0p03", "0p04", "0p05", "0p06", "0p07", "0p08", "0p09", "0p1"]
    det_range = np.linspace(-np.pi, np.pi, 40)

    for b_field in b_field_range:
        for det in det_range:
            run_simulation(
                _save_dir=
                "saved_data_timebin/photon_correlations/far_detuned/n_1/det_b_field_grid/",
                n_sim_steps=sim_steps,
                n_photons=1,
                b_field=b_field,
                _n_start=n_start,
                _len_stirap=t_vst,
                _shape_stirap=pulse_shape,
                _vst_ramp_up=0.25,
                _vst_ramp_down=0.1,
                _two_photon_det_4=det,
                _two_photon_det_3=0,
                _plot=False)

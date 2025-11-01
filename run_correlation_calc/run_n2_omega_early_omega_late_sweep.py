from run_correlation_calc.run_correlation_calc import run_simulation
import numpy as np

omega_stirap_early = np.linspace(30, 80, 10) * 2 * np.pi
omega_stirap_late = np.linspace(30, 80, 10) * 2 * np.pi


for omega_early in omega_stirap_early:
    for omega_late in omega_stirap_late:
        run_simulation(_save_dir="saved_data_timebin/photon_correlations/n_2/omega_early_late_sweep/",n_sim_steps=32, n_photons=2, b_field="0p07", _n_start=1, _len_stirap=0.5, _two_photon_det_2=0, _two_photon_det_4=0, _plot=False, _omega_stirap_1=omega_early, _omega_stirap_2=omega_late)

from run_correlation_calc.run_correlation_calc import run_simulation

sim_steps = 16
b_field = "0p07"
spont_emission = True
n_start = 1
length_vst = 1
two_photon_det_2 = 0
two_photon_det_4 = 0
pulse_shape = "flattop_blackman"
rise_time = 0.1
fall_time = 0.1

run_simulation(_save_dir="saved_data_timebin/photon_correlations/n_3/",n_sim_steps=sim_steps,_omega_stirap_early=30, _omega_stirap_late=30, n_photons=3, b_field=b_field, _shape_stirap=pulse_shape, _n_start=n_start, _vst_ramp_up=rise_time, _vst_ramp_down=0.17, _len_stirap=length_vst, _two_photon_det_2=two_photon_det_2, _two_photon_det_4=two_photon_det_4, _spont_emission=spont_emission,_plot=False)


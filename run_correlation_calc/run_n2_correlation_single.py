from run_correlation_calc import run_simulation

sim_steps = 32
b_fields = ["0"]
spont_emissions = [True, False]
n_starts = [1,2]
length_vst = 1
omega_vsts=[50]
two_photon_det_2 = 0
two_photon_det_4 = 0
pulse_shape = "flattop_blackman"
rise_times = [0.7]
fall_time = 0.2

'run all combinations of the parameters'
for b_field in b_fields:
    for spont_emission in spont_emissions:
        for n_start in n_starts:
            for omega_vst in omega_vsts:
                for rise_time in rise_times:
                    run_simulation(_save_dir="saved_data_timebin/photon_correlations/far_detuned/n_2/",n_sim_steps=sim_steps,_omega_stirap_early=omega_vst, _omega_stirap_late=omega_vst, n_photons=2, b_field=b_field, _shape_stirap=pulse_shape, _n_start=n_start, _vst_ramp_up=rise_time, _vst_ramp_down=fall_time, _len_stirap=length_vst, _two_photon_det_2=two_photon_det_2, _two_photon_det_4=two_photon_det_4, _spont_emission=spont_emission,_plot=False)


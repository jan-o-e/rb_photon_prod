from run_correlation_calc import run_simulation

sim_steps = 64
b_field = "0p07"
spont_emission = True
two_photon_det_2 = 0
two_photon_det_4 = 0
start=1
shape="flattop_blackman"

if __name__ == "__main__":

    run_simulation(_save_dir="saved_data_timebin/photon_correlations/far_detuned/n_1/",_shape_stirap=shape,n_sim_steps=sim_steps,_omega_stirap_early=30,_vst_ramp_up=0.17,_omega_stirap_late=30, n_photons=1, b_field=b_field, _n_start=start, _len_stirap=1, _two_photon_det_2=two_photon_det_2, _two_photon_det_4=two_photon_det_4, _spont_emission=spont_emission,_plot=True)

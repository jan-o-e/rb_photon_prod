import sys
import os

# Add the parent directory to the path (relative to this script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)

from modules.photon_correlation_calc import TimeBinPhotonCorrelations
import numpy as np
import pickle
import os
import secrets


def run_simulation(_save_dir,
                   n_sim_steps: int = 20,
                   b_field: str = "0",
                   _n_start: int = 2,
                   n_photons=2,
                   _omega_stirap_early=30,
                   _omega_stirap_late=30,
                   _len_stirap: float = 1,
                   _shape_stirap: str = "sinsquared",
                   _vst_ramp_up: float = 0.17,
                     _vst_ramp_down: float = 0.13,
                   _rot_omega_1: float =573.4,
                    _rot_omega_2: float =754.1772283803055,
                    _rot_omega_3: float =1122.6363636363635,
                    _rot_omega_4: float =991.7272727272727,
                    _two_photon_det_1: float =0,
                    _two_photon_det_2: float =0,
                    _two_photon_det_3: float =0,
                    _two_photon_det_4: float =0,
                    _plot=False,
                   _spont_emission=True,
                   _far_detuned_raman=False,
                   _chirped=True,
                   reduced_x_manifold=False):

    # Initialize the class
    ground_states = {
        "g1M": 0,
        "g1": 1,
        "g1P": 2,  # F=1,mF=-1,0,+1 respectively
        "g2MM": 3,
        "g2M": 4,
        "g2": 5,
        "g2P": 6,
        "g2PP": 7  # F=2,mF=-2,..,+2 respectively
    }

    # List the excited levels to include in the simulation. the _d1 levels correspond to the D1 line levels, the other levels are by default the d2 levels

    if reduced_x_manifold:
        x_states = [
            'x2MM', 'x2M', 'x2', 'x2P', 'x2PP', 'x1M_d1', 'x1_d1', 'x1P_d1'
        ]
    else:
        x_states = [
            'x0', 'x1M', 'x1', 'x1P', 
            'x2MM', 'x2M', 'x2', 'x2P', 'x2PP',
           'x3MMM', 'x3MM', 'x3M', 'x3', 'x3P', 'x3PP', 'x3PPP', 
            'x1M_d1',
            'x1_d1', 'x1P_d1', 'x2MM_d1', 'x2M_d1', 'x2_d1', 'x2P_d1',
            'x2PP_d1'
        ]

    correlator_class = TimeBinPhotonCorrelations(
        save_dir=_save_dir,
        _x_states=x_states,
        _ground_states=ground_states,
        _omega_stirap_early=_omega_stirap_early * 2 * np.pi,
        _omega_stirap_late=_omega_stirap_late * 2 * np.pi,
        _length_stirap=_len_stirap,
        _n_steps_vst=int(2000 * _len_stirap),
        _vstirap_pulse_shape=_shape_stirap,
        _vst_ramp_up_time=_vst_ramp_up,
        _vst_ramp_down_time=_vst_ramp_down,
        _rot_omega_1=_rot_omega_1,
        _rot_omega_2=_rot_omega_2,
        _rot_omega_3=_rot_omega_3,
        _rot_omega_4=_rot_omega_4,
        _bfield_split=b_field,
        _n_start=_n_start,
        _two_photon_det_1=_two_photon_det_1,
        _two_photon_det_3=_two_photon_det_3,
        _two_photon_det_2=_two_photon_det_2,
        _two_photon_det_4=_two_photon_det_4,
        _spont_emission=_spont_emission,
        _far_detuned_raman=_far_detuned_raman,
        _chirped=_chirped)
    
    os.makedirs(_save_dir, exist_ok=True)

    if n_photons == 1:
        res = correlator_class.gen_n1_density_matrix(n_sim_steps, _plot=_plot , _parall=True)
        # Save the results as pkl
        print(res)

        random_hex = secrets.token_hex(8)  # Generate a random 8-byte (16-character) hexadecimal string

        
        filename = os.path.join(
        _save_dir,
        f"n1_results_nstart{_n_start}__spont{_spont_emission}vstlength{correlator_class.length_stirap}_b{b_field}_omega_sti{_omega_stirap_early}_{_omega_stirap_late}_two_phot_det_{_two_photon_det_1}_{_two_photon_det_2}_{_two_photon_det_3}_{_two_photon_det_4}_omega_rot_{_rot_omega_1}_{_rot_omega_2}_{_rot_omega_3}_{_rot_omega_4}_{random_hex}.pkl"
    )
    elif n_photons == 2:
        res = correlator_class.gen_n2_density_matrix(n_sim_steps,
                                                     _plot=_plot,
                                                     _parall=True,
                                                     only_diag=False)
        # Save the results as pkl
        print(res)
        random_hex = secrets.token_hex(8)  # Generate a random 8-byte (16-character) hexadecimal string

        filename = os.path.join(
            _save_dir,
            f"n2_results_nstart{_n_start}__spont{_spont_emission}vstlength{correlator_class.length_stirap}_b{b_field}_omega_sti{_omega_stirap_early}_{_omega_stirap_late}_{random_hex}.pkl"
        )

    elif n_photons == 3:
        res = correlator_class.gen_n3_density_matrix(n_sim_steps,
                                                     _parall=True,
                                                     only_diag=False,
                                                     only_off_diag=False)
        # Save the results as pkl
        print(res)
        random_hex = secrets.token_hex(8)  # Generate a random 8-byte (16-character) hexadecimal string
        filename = os.path.join(
            _save_dir,
            f"n3_results_nstart{_n_start}_spont{_spont_emission}_vstlength{correlator_class.length_stirap}_b{b_field}_omega_sti{correlator_class._omega_stirap_early}_{correlator_class._omega_stirap_late}_{random_hex}.pkl"
        )

    # Save the results as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(res, f)

    print(f"Saved results to {filename}")

    correlator_class.reset_simulation()

    return res

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps=6
    #file_path="rb_photon_prod_dev/saved_data_timebin/photon_correlations/n_2/b_field_sweep/"
    #b_fields = ["0p11", "0p13", "0p15" ]
    #for b in b_fields:
    #    run_simulation(_save_dir="saved_data_timebin/photon_correlations/n_2/b_field_sweep/",n_sim_steps=sim_steps, n_photons=2, b_field=b, _n_start=2, _len_stirap=0.5, _two_photon_det_2=0, _two_photon_det_4=-0.68, _plot=True)

    #file_path="rb_photon_prod_dev/saved_data_timebin/photon_correlations/n_2/two_photon_det_sweep/"

    #det_range = np.linspace(-2.2, -1.2, 10)
    det_range=[0]
    for det in det_range:
        run_simulation(_save_dir="saved_data_timebin/photon_correlations/n_1/",n_sim_steps=sim_steps, n_photons=1, b_field="0p07", _n_start=1, _len_stirap=1, _two_photon_det_2=0, _two_photon_det_4=0, _plot=True)

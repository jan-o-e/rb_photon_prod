from run_correlation_calc.run_correlation_calc import run_simulation

# Ensure the script runs correctly when executed
if __name__ == "__main__":
    sim_steps = 32
    pulse_shape = "flattop_blackman"
    n_start = 2
    t_vst = 1
    b_field = [
        "0", "0p01", "0p02", "0p03", "0p04", "0p05", "0p06", "0p07", "0p08",
        "0p09", "0p1", "0p11", "0p12", "0p13", "0p14", "0p15", "0p16", "0p17",
        "0p18", "0p19", "0p2"
    ]

    for b in b_field:
        run_simulation(
            _save_dir=
            "saved_data_timebin/photon_correlations/far_detuned/n_1/b_sweep_new/",
            n_sim_steps=sim_steps,
            n_photons=1,
            b_field="0p07",
            _n_start=n_start,
            _len_stirap=t_vst,
            _plot=False)

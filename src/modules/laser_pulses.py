import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, hbar

def rabi_to_laserpower(omega, d, cg, beam_waist):
    """
    Convert Rabi frequency to laser power
    Input agrs:
    omega: Rabi frequency in MHz
    d: dipole moment in C*m
    cg: angular CG dependence
    beam_waist: beam waist in micron"""
    efield=(hbar*(omega*10**6))/(d*cg)
    intensity=(efield**2*epsilon_0*c)/(2)
    return (intensity*np.pi*(beam_waist*10**(-6))**2)*10**(3) # in mW

def create_pulse(_time_array, _pulse_fct, _pulse_amp):
    """
    Creates an array of amplitudes for a laser pulse
    Args:
        _time_array: array of time points over which to calculate the laser amplitude
        _pulse_fct: function governing the laser amplitude at each time step
        _pulse_amp: float scaling the overall amplitude of the pulse
    Returns:
        pulse_array: array of laser amplitudes
    """
    pulse_array = np.zeros(len(_time_array))
    for i in range(len(_time_array)):
        pulse_array[i] = _pulse_amp * _pulse_fct(_time_array[i], _time_array[-1])

    return pulse_array


def create_blackman(time_array, delay_factor, pump_amp, stokes_amp):
    """
    Creates a pair of Blackman laser pulses
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        delay_factor: float governing the delay between the two pulses
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
    Returns:
        pump_pulse: array of first laser amplitudes
        stokes_pules: array of second laser amplitudes
    """
    duration = time_array[-1]

    length1 = length2 = duration / (1 + delay_factor)
    lengthDelay = delay_factor * length1
    t0_2 = duration / 2 - lengthDelay / 2 - length2 / 2
    t0_1 = duration / 2 + lengthDelay / 2 - length1 / 2

    def pump_fct(t, T):
        return np.piecewise(
            t,
            [t >= t0_1],
            [
                (
                    0.5
                    * (
                        1.0
                        - 0.16
                        - np.cos(2.0 * np.pi * (t - t0_1) / (duration - t0_1))
                        + 0.16 * np.cos(4.0 * np.pi * (t - t0_1) / ((duration - t0_1)))
                    )
                ),
                0,
            ],
        )

    def stokes_fct(t, T):
        return np.piecewise(
            t,
            [t <= t0_2 + length2],
            [
                (
                    0.5
                    * (
                        1.0
                        - 0.16
                        - np.cos(2.0 * np.pi * (t - 0) / ((t0_2 + length2) - 0))
                        + 0.16 * np.cos(4.0 * np.pi * (t - 0) / ((t0_2 + length2) - 0))
                    )
                ),
                0,
            ],
        )

    pump_pulse = create_pulse(time_array, pump_fct, pump_amp)
    stokes_pulse = create_pulse(time_array, stokes_fct, stokes_amp)

    return (pump_pulse, stokes_pulse)


def create_tophat(time_array, tophat_ratio, laser_amp):
    """
    Creates a single tophat pulse
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        tophat_ratio: float governing the ratio of the total time over which the tophat is "on"
        laser_amp: float governing the amplitude of the laser
    Returns:
        laser_pulse: array of laser amplitudes
    """
    duration = time_array[-1]
    tophat_length = tophat_ratio * duration
    pulse_start = (duration - tophat_length) / 2
    pulse_end = pulse_start + tophat_length

    def laser_fct(t, T):
        if t >= pulse_start and t <= pulse_end:
            return 1
        else:
            return 0

    laser_pulse = create_pulse(time_array, laser_fct, laser_amp)

    return laser_pulse


def create_fstirap(time_array, tau_ratio, pump_amp, stokes_amp):
    """
    Creates a pair of fstirap pulses
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        tau_ratio: float governing the time between the two pulses
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
    Returns:
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """
    duration = time_array[-1]
    T = duration / 9
    a = 1 / np.sqrt(2)

    def pump_fct(t, tot_t):
        return np.exp(-((t - 4.5 * T - tau_ratio * T) ** 2) / (T**2)) * a

    def stokes_fct(t, tot_t):
        return np.exp(-((t - 4.5 * T + tau_ratio * T) ** 2) / (T**2)) + (
            np.exp(-((t - 4.5 * T - tau_ratio * T) ** 2) / T**2) * a
        )

    pump_array = create_pulse(time_array, pump_fct, pump_amp)
    stokes_array = create_pulse(time_array, stokes_fct, stokes_amp)

    return (pump_array, stokes_array)


def create_masked(time_array, pump_amp, stokes_amp, a, n, c):
    """
    Creates a pair of optimal pulses overlaid with a Gaussian mask
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
        a: determines symmetry/asymmetry of pulses, optimise this parameter
        n: power that the exponent is raised to, makes the pulse more spiky
        c: standard deviation of gaussian to ensure pulse starts at zero
    Returns:
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """

    def pump_fct(t, T):
        return np.exp(-(((t - (T / 2)) / c) ** (2 * n))) * np.sin(
            np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T)))
        )

    def stokes_fct(t, T):
        return np.exp(-(((t - (T / 2)) / c) ** (2 * n))) * np.cos(
            np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T)))
        )

    pump_array = create_pulse(time_array, pump_fct, pump_amp)
    stokes_array = create_pulse(time_array, stokes_fct, stokes_amp)

    return (pump_array, stokes_array)


def create_optimal(time_array, pump_amp, stokes_amp, a):
    """
    Creates a pair of optimal pulses (best possible but doesn't start at zero)
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
        a: float that determines the exact shape of the pulse
    Returns:
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """

    def pump_fct(t, T):
        return np.sin(np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T))))

    def stokes_fct(t, T):
        return np.cos(np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T))))

    pump_array = create_pulse(time_array, pump_fct, pump_amp)
    stokes_array = create_pulse(time_array, stokes_fct, stokes_amp)

    return (pump_array, stokes_array)


def create_sinsquared(time_array, pump_amp, stokes_amp, delay_factor):
    """
    Creates a pair of sin squared pulses. Very simple but suboptimal
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
        a: float that determines the exact shape of the pulse
    Returns:
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """
    duration = time_array[-1]
    length1 = length2 = duration / (1 + delay_factor)
    lengthDelay = delay_factor * length1
    [w2, w1] = [1 * np.pi / length for length in [length2, length1]]
    t0_2 = duration / 2 - lengthDelay / 2 - length2 / 2
    t0_1 = duration / 2 + lengthDelay / 2 - length1 / 2

    def pump_fct(t, T):
        return np.piecewise(t, [t > t0_1], [np.sin(w1 * (t - t0_1)) ** 2, 0])

    def stokes_fct(t, T):
        return np.piecewise(t, [t < t0_2 + length2], [np.sin(w2 * (t - t0_2)) ** 2, 0])

    pump_array = create_pulse(time_array, pump_fct, pump_amp)
    stokes_array = create_pulse(time_array, stokes_fct, stokes_amp)

    return (pump_array, stokes_array)


def create_single_sinsquared(time_array, laser_amp):
    """
    Creates a single sin squared pulse. Useful for vstirap
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        laser_amp: amplitude of the laser
    Returns:
        laser_array: array of amplitudes
    """
    omega = np.pi / time_array[-1]

    def laser_fct(t, T):
        return np.sin(omega * t) ** 2

    laser_array = create_pulse(time_array, laser_fct, laser_amp)

    return laser_array


def create_single_blackman(time_array, laser_amp):
    """
    Creates a single Blackman pulse using the create_pulse function.
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        laser_amp: amplitude of the laser
    Returns:
        laser_array: array of amplitudes
    """

    def blackman_fct(t, T):
        return 0.42 - 0.5 * np.cos(2 * np.pi * t / T) + 0.08 * np.cos(4 * np.pi * t / T)

    # Use create_pulse to apply the function over the time array
    laser_array = create_pulse(time_array, blackman_fct, laser_amp)

    return laser_array


def create_flattop_gaussian(time_array, laser_amp, ramp_up_time, ramp_down_time):
    """
    Creates a single flattop pulse with a gaussian ramp up and down
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        laser_amp: amplitude of the laser
        ramp_time: time over which the ramp up and down occur
    Returns:
        laser_array: array of amplitudes
    """

    if ramp_up_time + ramp_down_time < time_array[-1]:
        AssertionError("Ramp up and down time is greater than total time")

    # Define the flattop function as a lambda function
    # picked a sigma of 4 to ensure the pulse starts at zero
    def flattop_fct(t, T):
        if t < ramp_up_time:
            return np.exp(-0.5 * ((t - ramp_up_time) / (ramp_up_time / 4)) ** 2)
        elif t > T - ramp_down_time:
            return np.exp(
                -0.5 * ((t - (T - ramp_down_time)) / (ramp_down_time / 4)) ** 2
            )
        else:
            return 1

    # Use create_pulse to apply the function over the time array
    laser_array = create_pulse(time_array, flattop_fct, laser_amp)

    return laser_array


def create_flattop_blackman(time_array, laser_amp, ramp_up_time, ramp_down_time):
    """
    Creates a single flattop pulse with a Blackman ramp up and down.
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        laser_amp: amplitude of the laser
        ramp_time: time over which the ramp up and down occur
    Returns:
        laser_array: array of amplitudes
    """
    # assert ramp time is less total time
    if ramp_up_time + ramp_down_time < time_array[-1]:
        AssertionError("Ramp up and down time is greater than total time")

    T = time_array[-1]  # Total duration of the pulse

    # Define the Blackman ramp-up and ramp-down function
    def flattop_fct(t, T):
        if t < ramp_up_time:  # Ramp up
            return (
                0.42
                - 0.5 * np.cos(2 * np.pi * t / (2 * ramp_up_time))
                + 0.08 * np.cos(4 * np.pi * t / (2 * ramp_up_time))
            )
        elif t > T - ramp_down_time:  # Ramp down
            t_ramp = T - t
            return (
                0.42
                - 0.5 * np.cos(2 * np.pi * t_ramp / (2 * ramp_down_time))
                + 0.08 * np.cos(4 * np.pi * t_ramp / (2 * ramp_down_time))
            )
        else:  # Flattop region
            return 1

    # Apply the flattop function to the time array
    laser_array = np.array([laser_amp * flattop_fct(t, T) for t in time_array])

    return laser_array


def create_fstirap_str(time_array, tau_ratio):
    """
    Creates a pair of fstirap pulses represented by strings and arrays.
    The amplitude is approx normalised to 1
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        tau_ratio: float governing the time between the two pulses
    Returns:
        pump_shape (str): shape of first laser
        stokes_shape (str): shape of second laser
        shape_args (dict): dictionary of parameters needed to describe the lasers
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """
    duration = time_array[-1]
    T = duration / 9
    a = 1 / np.sqrt(2)

    # create string representations
    pump_shape = "exp(-(t-4.5*T-tau*T)**2/(T**2))*a"
    stokes_shape = "exp(-(t-4.5*T+tau*T)**2/(T**2))+exp(-(t-4.5*T-tau*T)**2/T**2)*a"

    shape_args = {"a": a, "T": T, "tau": tau_ratio}

    # create arrays
    def pump_fct(t, tot_t):
        return np.exp(-((t - 4.5 * T - tau_ratio * T) ** 2) / (T**2)) * a

    def stokes_fct(t, tot_t):
        return (
            np.exp(-((t - 4.5 * T + tau_ratio * T) ** 2) / (T**2))
            + np.exp(-((t - 4.5 * T - tau_ratio * T) ** 2) / T**2) * a
        )

    pump_array = create_pulse(time_array, pump_fct, 1)
    stokes_array = create_pulse(time_array, stokes_fct, 1)

    return (pump_shape, stokes_shape, shape_args, pump_array, stokes_array)


def create_masked_str(time_array, a, n, c):
    """
    Creates a pair of optimal pulses overlaid with a Gaussian mask.
    Represented by strings and arrays.
    Amplitude approx normalised to 1
    Args:
        time_array: array of time points over which to calculate the laser amplitude
        pump_amp: amplitude of the first laser
        stokes_amp: amplitude of the second laser
        a: determines symmetry/asymmetry of pulses, optimise this parameter
        n: power that the exponent is raised to, makes the pulse more spiky
        c: standard deviation of gaussian to ensure pulse starts at zero
    Returns:
        pump_shape (str): shape of first laser
        stokes_shape (str): shape of second laser
        shape_args (dict): dictionary of parameters needed to describe the lasers
        pump_array: array of amplitudes of the first laser
        stokes_array: array of amplitudes of the second laser
    """

    T = time_array[-1]

    # create string representations
    pump_shape = "exp(-((t-(T/2))/c)**(2*n))*sin(pi/2*(1/(1+exp((-a*(t-T/2))/T))))"
    stokes_shape = (
        "exp(-((t - (T/2))/c)**(2*n))*cos(pi/2*(1/(1 + exp((-a*(t - T/2))/T))))"
    )

    shape_args = {"T": T, "a": a, "c": c, "n": n}

    def pump_fct(t, T):
        return np.exp(-(((t - (T / 2)) / c) ** (2 * n))) * np.sin(
            np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T)))
        )

    def stokes_fct(t, T):
        return np.exp(-(((t - (T / 2)) / c) ** (2 * n))) * np.cos(
            np.pi / 2 * (1 / (1 + np.exp((-a * (t - T / 2)) / T)))
        )

    pump_array = create_pulse(time_array, pump_fct, 1)
    stokes_array = create_pulse(time_array, stokes_fct, 1)

    return (pump_shape, stokes_shape, shape_args, pump_array, stokes_array)


def create_linear_ramp_up_down(t, t_total, t_up, t_down):
    """
    Generate a signal with a linear ramp up to 1, a constant value of 1,
    and then a linear ramp down from 1 to 0.

    Parameters:
    - t (array-like): Array of time values.
    -t_total (float): Total time of the signal.
    - t_up (float): Time for the linear ramp up to 1.
    - t_down (float): Time for the linear ramp down from 1 to 0.

    Returns:
    - array-like: Signal values corresponding to the time values.
    """
    ramp_up = np.where((t >= 0) & (t < t_up), t / t_up, 0)
    constant = np.where((t >= t_up) & (t < t_total - t_down), 1, 0)
    ramp_down = np.where(
        (t >= t_total - t_down) & (t <= t_total),
        1 - (t - (t_total - t_down)) / t_down,
        0,
    )

    return ramp_up + constant + ramp_down


if __name__ == "__main__":
    # Define time array and amplitude
    time_array = np.linspace(0, 1, 500)  # 0 to 1 seconds with 500 points
    laser_amp = 1.0  # Maximum amplitude of the pulse

    # Generate Blackman pulse using create_blackman_pulse
    laser_array = create_single_blackman(time_array, laser_amp)

    # Plot the result
    plt.figure(figsize=(8, 4))
    plt.plot(time_array, laser_array, label="Blackman Pulse", color="blue")
    plt.title("Blackman Pulse")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    laser_array = create_flattop_gaussian(time_array, laser_amp, 0.2, 0.1)

    # Plot the result

    plt.figure(figsize=(8, 4))
    plt.plot(time_array, laser_array, label="Flattop Gaussian Pulse", color="blue")
    plt.title("Flattop Gaussian Pulse")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    laser_array = create_flattop_blackman(time_array, laser_amp, 0.3, 0.1)

    # Plot the result
    plt.figure(figsize=(8, 4))
    plt.plot(time_array, laser_array, label="Flattop Blackman Pulse", color="blue")
    plt.title("Flattop Blackman Pulse")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

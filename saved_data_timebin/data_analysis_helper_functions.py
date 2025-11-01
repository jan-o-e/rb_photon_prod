import numpy as np
from scipy.linalg import sqrtm
import pickle
import glob
import re
import os

# Function to load pickle file
def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to compute the complex phase of the off-diagonal element 10
def compute_phase(real_part, imag_part):
    return np.arctan2(imag_part, real_part)

one_photon_ghz = 1 / 2 * np.array([[1, 1], [1, 1]])

# Function to calculate fidelity
def calculate_fidelity(rho, sigma):
    """
    Compute the fidelity between two density matrices.

    Parameters:
    rho (numpy.ndarray): The first density matrix (must be a square, Hermitian, positive semi-definite matrix).
    sigma (numpy.ndarray): The second density matrix (must be a square, Hermitian, positive semi-definite matrix).

    Returns:
    float: The fidelity between the two density matrices.
    """
    # Compute the square root of rho
    sqrt_rho = sqrtm(rho)
    
    # Compute the product sqrt(rho) * sigma * sqrt(rho)
    product = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
    
    # Compute the square root of the product
    sqrt_product = sqrtm(product)
    
    # Take the trace and square the result
    trace_value = np.trace(sqrt_product)
    
    # Return the fidelity
    return np.real(trace_value)**2

def process_bm_grid_time(file_path_pattern):
    fall_times = []
    rise_times = []
    fidelities = []
    diagonals = []

    # Store fall_time and rise_time combinations for later reshaping
    fall_rise_pairs = []

    # Loop over all pickle files in the specified directory
    file_paths = sorted(glob.glob(file_path_pattern))
    if not file_paths:
        print("No files found matching the pattern.")
        return fall_times, rise_times, fidelities, diagonals

    for file_path in file_paths:
        #print(f"Processing file: {file_path}")

        # Load the pickle data
        try:
            data = load_pkl(file_path)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            continue

        # Extract real and imaginary parts of the density matrix elements
        try:
            int_diag_00_re = data['int_diag_00_re']
            int_diag_00_im = data['int_diag_00_im']
            int_diag_11_re = data['int_diag_11_re']
            int_diag_11_im = data['int_diag_11_im']
            int_off_diag_10_re = data['int_off_diag_10_re']
            int_off_diag_10_im = data['int_off_diag_10_im']
            int_off_diag_01_re = data['int_off_diag_01_re']
            int_off_diag_01_im = data['int_off_diag_01_im']
        except KeyError as e:
            print(f"Error extracting matrix elements from {file_path}: {e}")
            continue

        # Construct the density matrix from the components
        density_matrix = np.array([[
            int_diag_00_re,
            np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        ],
                                   [
                                       np.abs(int_off_diag_10_re +
                                              1j * int_off_diag_10_im),
                                       int_diag_11_re
                                   ]])

        # Check the condition for the off-diagonal terms
        off_diag_magnitude = np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        max_diag = max(int_diag_00_re, int_diag_11_re)
        min_diag = min(int_diag_00_re, int_diag_11_re)

        if off_diag_magnitude > min_diag:
            density_matrix[0, 1] = density_matrix[1, 0] = min_diag

        # Normalize the density matrix
        sum_real_diag = int_diag_00_re + int_diag_11_re
        if sum_real_diag != 0:
            density_matrix /= sum_real_diag

        rise_time = float(data["vst_rise_time"])  # Convert to float
        fall_time = float(data["vst_fall_time"])  # Convert to float
        fidelity = calculate_fidelity(density_matrix, one_photon_ghz)

        # Store fall_time and rise_time values and their corresponding fidelity/diagonal
        fall_times.append(fall_time)
        rise_times.append(rise_time)
        fidelities.append(fidelity)
        diagonals.append(sum_real_diag)
        fall_rise_pairs.append((fall_time, rise_time))

    if not fall_times:
        print("No data processed. Please check file paths or formats.")

    # Convert to numpy arrays for easier manipulation
    fall_times = np.array(fall_times)
    rise_times = np.array(rise_times)
    # Get unique fall_time and rise_time values
    unique_fall_times = np.unique(fall_times)
    unique_rise_times = np.unique(rise_times)

    # Create 2D grid arrays for fidelity and diagonals
    fidelity_grid = np.zeros((len(unique_fall_times), len(unique_rise_times)))
    diagonal_grid = np.zeros((len(unique_fall_times), len(unique_rise_times)))

    # Fill the 2D grids with values from the processed data
    for i, fall_time in enumerate(unique_fall_times):
        for j, rise_time in enumerate(unique_rise_times):
            # Find the corresponding fidelity and diagonal for each fall_time-rise_time pair
            idx = np.where((fall_times == fall_time) & (rise_times == rise_time))[0]
            if len(idx) > 0:
                fidelity_grid[i, j] = fidelities[idx[0]]
                diagonal_grid[i, j] = diagonals[idx[0]]

    return unique_fall_times, unique_rise_times, fidelity_grid, diagonal_grid
def process_omega_grid(file_path_pattern):
    omegas = []
    lengths = []
    fidelities = []
    diagonals = []

    # Store omega and length combinations for later reshaping
    omega_length_pairs = []

    # Loop over all pickle files in the specified directory
    file_paths = sorted(glob.glob(file_path_pattern))
    if not file_paths:
        print("No files found matching the pattern.")
        return omegas, lengths, fidelities, diagonals

    for file_path in file_paths:
        #print(f"Processing file: {file_path}")

        # Load the pickle data
        try:
            data = load_pkl(file_path)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            continue

        # Extract real and imaginary parts of the density matrix elements
        try:
            int_diag_00_re = data['int_diag_00_re']
            int_diag_00_im = data['int_diag_00_im']
            int_diag_11_re = data['int_diag_11_re']
            int_diag_11_im = data['int_diag_11_im']
            int_off_diag_10_re = data['int_off_diag_10_re']
            int_off_diag_10_im = data['int_off_diag_10_im']
            int_off_diag_01_re = data['int_off_diag_01_re']
            int_off_diag_01_im = data['int_off_diag_01_im']
        except KeyError as e:
            print(f"Error extracting matrix elements from {file_path}: {e}")
            continue

        # Construct the density matrix from the components
        density_matrix = np.array([[
            int_diag_00_re,
            np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        ],
                                   [
                                       np.abs(int_off_diag_10_re +
                                              1j * int_off_diag_10_im),
                                       int_diag_11_re
                                   ]])

        # Check the condition for the off-diagonal terms
        off_diag_magnitude = np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        max_diag = max(int_diag_00_re, int_diag_11_re)
        min_diag = min(int_diag_00_re, int_diag_11_re)

        if off_diag_magnitude > min_diag:
            density_matrix[0, 1] = density_matrix[1, 0] = min_diag

        # Normalize the density matrix
        sum_real_diag = int_diag_00_re + int_diag_11_re
        if sum_real_diag != 0:
            density_matrix /= sum_real_diag
        


        # Extract the omega and length values from the filename using regex
        file_name = file_path.split('/')[-1]  # Get just the filename

        # Use regex to extract the length and omega values
        pattern = r'vstlength([\d\.]+)_.*?omega_sti([\d\.]+)'

        match = re.search(pattern, file_name)       
        if match:
            length_str = match.group(1)
            omega_str = match.group(2)
            #print(f"Extracted length: {length_str}, omega: {omega_str}")
        else:
            print(f"Filename format not recognized: {file_name}")
            continue

        try:
            length = float(length_str)  # Convert to float
            omega = float(omega_str)  # Convert to float
            if omega==100:
                print(length)
                print(density_matrix)
            fidelity = calculate_fidelity(density_matrix, one_photon_ghz)

            # Store omega and length values and their corresponding fidelity/diagonal
            omegas.append(omega)
            lengths.append(length)
            fidelities.append(fidelity)
            diagonals.append(sum_real_diag)
            omega_length_pairs.append((omega, length))

        except ValueError as e:
            print(f"Error parsing omega/length from {file_path}: {e}")

    if not omegas:
        print("No data processed. Please check file paths or formats.")

    # Convert to numpy arrays for easier manipulation
    omegas = np.array(omegas)
    lengths = np.array(lengths)
    # Get unique omega and length values
    unique_omegas = np.unique(omegas)
    unique_lengths = np.unique(lengths)

    # Create 2D grid arrays for fidelity and diagonals
    fidelity_grid = np.zeros((len(unique_omegas), len(unique_lengths)))
    diagonal_grid = np.zeros((len(unique_omegas), len(unique_lengths)))

    # Fill the 2D grids with values from the processed data
    for i, omega in enumerate(unique_omegas):
        for j, length in enumerate(unique_lengths):
            # Find the corresponding fidelity and diagonal for each omega-length pair
            idx = np.where((omegas == omega) & (lengths == length))[0]
            if len(idx) > 0:
                fidelity_grid[i, j] = fidelities[idx[0]]
                diagonal_grid[i, j] = diagonals[idx[0]]

    return unique_omegas, unique_lengths, fidelity_grid, diagonal_grid

def process_detuning_bfield(directory_path):
    detunings = []
    b_fields = []
    fidelities = []
    diagonals = []

    # Loop over all pickle files in the specified directory
    file_paths = sorted(glob.glob(os.path.join(directory_path, "*.pkl")))
    if not file_paths:
        print("No .pkl files found in the directory.")

    for file_path in file_paths:
        # Load the pickle data
        try:
            data = load_pkl(file_path)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            continue

        # Extract b_field and two_photon_det
        try:
            b_field_str = data['bfield_split']
            detuning = data['two_photon_det'][3]

            # Convert b_field string (e.g., "0p01") to float
            detuning = float(detuning)
            b_field = float(b_field_str.replace('p', '.'))
        except KeyError as e:
            print(f"Key not found in data for file {file_path}: {e}")
            continue
        except ValueError as e:
            print(f"Error converting b_field or detuning for file {file_path}: {e}")
            continue

        # Extract real and imaginary parts of the density matrix elements
        try:
            int_diag_00_re = data['int_diag_00_re']
            int_diag_11_re = data['int_diag_11_re']
            int_off_diag_10_re = data['int_off_diag_10_re']
            int_off_diag_10_im = data['int_off_diag_10_im']
            int_off_diag_01_re = data['int_off_diag_01_re']
            int_off_diag_01_im = data['int_off_diag_01_im']
        except KeyError as e:
            print(f"Error extracting matrix elements from {file_path}: {e}")
            continue

        # Construct the density matrix from the components
        density_matrix = np.array([[
            int_diag_00_re,
            int_off_diag_01_re + 1j * int_off_diag_01_im
        ],
                                   [
                                       int_off_diag_10_re +
                                              1j * int_off_diag_10_im,
                                       int_diag_11_re
                                   ]])
        
        # Check the condition for the off-diagonal terms
        off_diag_magnitude = np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        min_diag = min(int_diag_00_re, int_diag_11_re)

        if off_diag_magnitude > min_diag:
            density_matrix[0, 1] = density_matrix[1, 0] = min_diag

        # Normalize the density matrix
        sum_real_diag = int_diag_00_re + int_diag_11_re
        if sum_real_diag != 0:
            density_matrix /= sum_real_diag

        try:
            fidelity = calculate_fidelity(density_matrix, one_photon_ghz)

            # Store b-field and detuning values and their corresponding fidelity/diagonal
            detunings.append(detuning)
            b_fields.append(b_field)
            fidelities.append(fidelity)
            diagonals.append(sum_real_diag)
        except Exception as e:
            print(f"Error calculating fidelity for file {file_path}: {e}")

    if not detunings:
        print("No data processed. Please check file contents or formats.")

    # Convert to numpy arrays for easier manipulation
    detunings = np.array(detunings)
    b_fields = np.array(b_fields)
    # Get unique detuning and b-field values
    unique_detunings = np.unique(detunings)
    unique_b_fields = np.unique(b_fields)

    # Create 2D grid arrays for fidelity
    fidelity_grid = np.zeros((len(unique_detunings), len(unique_b_fields)))

    # Fill the 2D grids with values from the processed data
    for i, detuning in enumerate(unique_detunings):
        for j, b_field in enumerate(unique_b_fields):
            # Find the corresponding fidelity for each detuning-b_field pair
            idx = np.where((detunings == detuning) & (b_fields == b_field))[0]
            if len(idx) > 0:
                fidelity_grid[i, j] = fidelities[idx[0]]
    
    return unique_detunings, unique_b_fields, fidelity_grid


def process_omega_omega(directory_path):
    omega_3s = []
    omega_4s = []
    fidelities = []
    diagonals = []

    # Loop over all pickle files in the specified directory
    file_paths = sorted(glob.glob(os.path.join(directory_path, "*.pkl")))
    if not file_paths:
        print("No .pkl files found in the directory.")

    for file_path in file_paths:
        # Load the pickle data
        try:
            data = load_pkl(file_path)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            continue

        # Extract b_field and two_photon_det
        try:
            omega_3 = data['omega_rot_stirap'][2]
            omega_4 = data['omega_stirap'][0]

            # Convert b_field string (e.g., "0p01") to float
            omega_3 = float(omega_3)
            omega_4 = float(omega_4)
        except KeyError as e:
            print(f"Key not found in data for file {file_path}: {e}")
            continue
        except ValueError as e:
            print(f"Error converting b_field or detuning for file {file_path}: {e}")
            continue

        # Extract real and imaginary parts of the density matrix elements
        try:
            int_diag_00_re = data['int_diag_00_re']
            int_diag_11_re = data['int_diag_11_re']
            int_off_diag_10_re = data['int_off_diag_10_re']
            int_off_diag_10_im = data['int_off_diag_10_im']
            int_off_diag_01_re = data['int_off_diag_01_re']
            int_off_diag_01_im = data['int_off_diag_01_im']
        except KeyError as e:
            print(f"Error extracting matrix elements from {file_path}: {e}")
            continue

        # Construct the density matrix from the components
        density_matrix = np.array([[
            int_diag_00_re,
            np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        ],
                                   [
                                       np.abs(int_off_diag_10_re +
                                              1j * int_off_diag_10_im),
                                       int_diag_11_re
                                   ]])
        
        # Check the condition for the off-diagonal terms
        off_diag_magnitude = np.abs(int_off_diag_01_re + 1j * int_off_diag_01_im)
        min_diag = min(int_diag_00_re, int_diag_11_re)

        if off_diag_magnitude > min_diag:
            density_matrix[0, 1] = density_matrix[1, 0] = min_diag

        # Normalize the density matrix
        sum_real_diag = int_diag_00_re + int_diag_11_re
        if sum_real_diag != 0:
            density_matrix /= sum_real_diag

        try:
            fidelity = calculate_fidelity(density_matrix, one_photon_ghz)

            # Store b-field and detuning values and their corresponding fidelity/diagonal
            omega_3s.append(omega_3)
            omega_4s.append(omega_4)
            fidelities.append(fidelity)
            diagonals.append(sum_real_diag)
        except Exception as e:
            print(f"Error calculating fidelity for file {file_path}: {e}")

    if not omega_3s:
        print("No data processed. Please check file contents or formats.")

    # Convert to numpy arrays for easier manipulation
    omega_3s = np.array(omega_3s)
    omega_4s = np.array(omega_4s)
    # Get unique detuning and b-field values
    unique_omega_3s = np.unique(omega_3s)
    unique_omega_4s = np.unique(omega_4s)

    # Create 2D grid arrays for fidelity
    fidelity_grid = np.zeros((len(unique_omega_3s), len(unique_omega_4s)))
    print(fidelity_grid.shape)

    # Fill the 2D grids with values from the processed data
    for i, detuning in enumerate(unique_omega_3s):
        for j, b_field in enumerate(unique_omega_4s):
            # Find the corresponding fidelity for each detuning-b_field pair
            idx = np.where((omega_3s == detuning) & (omega_4s == b_field))[0]
            if len(idx) > 0:
                fidelity_grid[i, j] = fidelities[idx[0]]
    
    return unique_omega_3s, unique_omega_4s, fidelity_grid


def process_phase_phase(file_path_pattern):
    detunings = []
    detunings_2 = []
    fidelities = []
    diagonals = []

    # Loop over all pickle files in the specified directory
    file_paths = sorted(glob.glob(file_path_pattern))
    if not file_paths:
        print("No .pkl files found in the directory.")

    for file_path in file_paths:
        #print(f"Processing file: {file_path}")

        # Load the pickle data
        try:
            data = load_pkl(file_path)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            continue
        # Extract detunings
        try:
            detuning = float(data['two_photon_det'][3])
            detuning_2 = float(data['two_photon_det'][2])
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error extracting detuning values from file {file_path}: {e}")
            continue

        # Extract real and imaginary parts of the density matrix elements
        try:
            int_diag_00_re = data['int_diag_00_re']
            int_diag_11_re = data['int_diag_11_re']
            int_off_diag_10_re = data['int_off_diag_10_re']
            int_off_diag_10_im = data['int_off_diag_10_im']
            int_off_diag_01_re = data['int_off_diag_01_re']
            int_off_diag_01_im = data['int_off_diag_01_im']
        except KeyError as e:
            print(f"Error extracting matrix elements from {file_path}: {e}")
            continue

        # Construct the density matrix from the components
        density_matrix = np.array([
            [int_diag_00_re, int_off_diag_01_re + 1j * int_off_diag_01_im],
            [int_off_diag_10_re + 1j * int_off_diag_10_im, int_diag_11_re]
        ])

        # Check the condition for the off-diagonal terms
        off_diag_magnitude = np.abs(density_matrix[0, 1])
        min_diag = min(int_diag_00_re, int_diag_11_re)

        if off_diag_magnitude > min_diag:
            density_matrix[0, 1] = density_matrix[1, 0] = min_diag

        # Normalize the density matrix
        sum_real_diag = int_diag_00_re + int_diag_11_re
        if sum_real_diag != 0:
            density_matrix /= sum_real_diag

        try:
            fidelity = calculate_fidelity(density_matrix, one_photon_ghz)

            # Store detunings and fidelity
            detunings.append(detuning)
            detunings_2.append(detuning_2)
            fidelities.append(fidelity)
            diagonals.append(sum_real_diag)
        except Exception as e:
            print(f"Error calculating fidelity for file {file_path}: {e}")

    if not detunings:
        print("No data processed. Please check file contents or formats.")

    # Convert to numpy arrays
    detunings = np.array(detunings)
    detunings_2 = np.array(detunings_2)
    unique_detunings = np.unique(detunings)
    unique_detunings_2 = np.unique(detunings_2)

    # Create 2D grid for fidelities
    fidelity_grid = np.zeros((len(unique_detunings), len(unique_detunings_2)))

    for i, d1 in enumerate(unique_detunings):
        for j, d2 in enumerate(unique_detunings_2):
            idx = np.where((detunings == d1) & (detunings_2 == d2))[0]
            if len(idx) > 0:
                fidelity_grid[i, j] = fidelities[idx[0]]

    return unique_detunings, unique_detunings_2, fidelity_grid

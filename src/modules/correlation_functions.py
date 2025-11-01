import numpy as np
from qutip import Options, isket, ket2dm, mesolve, expect
import time


def _find_closest_value_index(flattened_t_list, t_inp):
    # Calculate absolute differences
    abs_diff = np.abs(flattened_t_list - t_inp)
    # Find index of the minimum difference
    idx = np.argmin(abs_diff)
    return idx


def _find_next_largest_index(_time_bounds, t):
    for i, _time in enumerate(_time_bounds):
        if _time >= t:
            return i
    return AssertionError("Invalid time specified")


def _find_next_smallest_index(_time_bounds, t):
    # Create a copy of the input array to avoid modifying the original list
    time_bounds_copy = _time_bounds.copy()

    # Ensure the copy starts with 0
    if time_bounds_copy[0] != 0:
        time_bounds_copy.insert(0, 0)

    # Find the next smallest index
    # print(f"time_bounds_copy: {time_bounds_copy}")
    for i, _time in enumerate(time_bounds_copy):
        # if index is at the second to last el for comparison then allow geq condition for max time
        if i == len(time_bounds_copy) - 1:
            if t >= _time and t <= time_bounds_copy[i + 1]:
                return i
        else:
            if t >= _time and t < time_bounds_copy[i + 1]:
                return i

    raise AssertionError("Invalid time specified")


def exp_eval_floating_start_finish(
    h_list,
    t_sim_lists,
    state0,
    t_start,
    t_ret_list,
    c_op_lists,
    a_op,
    b_op,
    e_op,
    args={},
    options=Options(store_final_state=True),
    debug=False,
):
    """
    Internal function for calculating the evolution of a density matrix with some operators
    <A(t) B(t)>
    using a master equation solver and returns the expectation value of an operator at the final time
    input args:
    H_list: list of Hamiltonians for each t_sim_lists
    t_sim_lists: list of simulation times for each Hamiltonian
    state0: initial state at start time
    t_start: start_time
    t_ret_list: times at which to evaluate the expectation value in ascending order
    c_ops: list of collapse operators
    a_op: operator which to evolve from left
    b_op: operator which to evolve from right
    e_op: operator for expectation value
    args: arguments for the solver
    """
    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    # create list of boundaries for each time list define hamiltonian simulation times
    time_h_bounds = []
    t_bound = 0
    for sim_time in t_sim_lists:
        t_bound += sim_time[-1]
        if t_bound > 0:
            time_h_bounds.append(t_bound)

    # time_h_bounds = []
    # t_bound = 0

    # for sim_time in t_sim_lists:
    #    t_bound += sim_time[-1]

    #    if t_bound>0:
    #        time_h_bounds.append(t_bound)

    # if debug:
    # print(f"sim_time[-1]: {sim_time[-1]}")

    if t_ret_list[-1] > max(time_h_bounds):
        raise ValueError("t_finish must not exceed simulation time list max")

    # find the indices of the Hamiltonians that correspond to the start and finish times
    max_time = max(t_ret_list)
    max_ham_index = _find_next_largest_index(time_h_bounds, max_time)
    min_ham_index = _find_next_smallest_index(time_h_bounds, t_start)

    if debug:
        print(max_time)
        print(f"Min ham index: {min_ham_index}")
        print(f"Max ham index: {max_ham_index}")
        print(f"Time bounds: {time_h_bounds}")

    # slice the time lists and Hamiltonians to the start and finish times
    if min_ham_index == 0:
        min_ham_time_idx = _find_closest_value_index(
            t_sim_lists[min_ham_index], t_start
        )
    else:
        min_ham_time_idx = _find_closest_value_index(
            t_sim_lists[min_ham_index], t_start - time_h_bounds[min_ham_index - 1]
        )
    min_ham_sliced_time_list = t_sim_lists[min_ham_index][
        min_ham_time_idx : len(t_sim_lists[min_ham_index])
    ]
    if debug:
        print(f"Min ham time idx: {min_ham_time_idx}")
    H_min_sliced = []
    for elem in h_list[min_ham_index]:
        H_min_sliced.append(
            [elem[0], elem[1][min_ham_time_idx : len(t_sim_lists[min_ham_index])]]
        )

    max_ham_time_idx = _find_closest_value_index(
        t_sim_lists[max_ham_index], max_time - time_h_bounds[max_ham_index - 1]
    )
    max_ham_sliced_time_list = t_sim_lists[max_ham_index][0:max_ham_time_idx]
    if debug:
        print("Max ham time idx: ", max_ham_time_idx)
    H_max_sliced = []
    for elem in h_list[max_ham_index]:
        H_max_sliced.append([elem[0], elem[1][0:max_ham_time_idx]])
    assert len(H_min_sliced[0][1]) == len(min_ham_sliced_time_list)
    assert len(H_max_sliced[0][1]) == len(max_ham_sliced_time_list)

    # create list of indices for each time list to evaluate the expectation value
    exp_return_indices = []
    for i in range(len(h_list)):
        exp_return_indices.append([])

    if debug:
        print("t_ret_list: ", t_ret_list)
        print("time_h_bounds: ", time_h_bounds)

    for t in t_ret_list:
        for i, time_bound in enumerate(time_h_bounds):
            if t <= time_bound:
                if i == 0:
                    ham_index = _find_closest_value_index(t_sim_lists[i], t)
                else:
                    ham_index = _find_closest_value_index(
                        t_sim_lists[i], t - time_h_bounds[i - 1]
                    )
                exp_return_indices[i].append(ham_index)
                break

    if debug:
        print(f"Min ham time list initial: {min_ham_sliced_time_list[0]}")
        print(f"Max ham time list final: {max_ham_sliced_time_list[-1]}")
        print(f"Exp Return Indices: {exp_return_indices}")

    exp_return = []

    time_sim_start = time.time()

    for i, H in enumerate(h_list):
        if i == min_ham_index:
            # otherwise the solver will go to SS
            if len(min_ham_sliced_time_list) > 1:
                rho_first_evolution = mesolve(
                    H_min_sliced,
                    a_op * rho0 * b_op,
                    min_ham_sliced_time_list,
                    c_op_lists[i],
                    [],
                    args=args,
                    options=options,
                ).states
                rho_next = rho_first_evolution[-1]

            else:
                rho_first_evolution = rho0
                rho_next = rho0

            if exp_return_indices[i]:
                for idx in exp_return_indices[i]:
                    if idx == 0:
                        if e_op is None:
                            exp_return.append(a_op * rho_next * b_op)
                        else:
                            exp_return.append(expect(e_op, a_op * rho0 * b_op))
                    else:
                        # since rho_first_evo is smaller than rho_first evo by one we need to return the i-1st index
                        if debug:
                            print("return index:", idx)
                            print("len(rho_first_evo..)", len(rho_first_evolution))
                        # make sure we are returning the correct index with min ham time idx subtracted
                        if e_op is None:
                            exp_return.append(
                                rho_first_evolution[idx - 1 - min_ham_time_idx]
                            )
                        else:
                            exp_return.append(
                                expect(
                                    e_op,
                                    rho_first_evolution[idx - 1 - min_ham_time_idx],
                                )
                            )

        elif i < max_ham_index and i > min_ham_index:
            rho_m = mesolve(
                H,
                rho_next,
                t_sim_lists[i],
                c_op_lists[i],
                [],
                args=args,
                options=options,
            ).states

            rho_next = rho_m[-1]

            if exp_return_indices[i]:
                for idx in exp_return_indices[i]:
                    if idx == 0:
                        if e_op is None:
                            exp_return.append(rho_next)
                        else:
                            exp_return.append(expect(e_op, rho_next))
                    else:
                        if e_op is None:
                            exp_return.append(rho_m[idx - 1])
                        else:
                            # since rho_first_evo is smaller than rho_first evo by one we need to return the i-1st index
                            exp_return.append(expect(e_op, rho_m[idx - 1]))

        elif i == max_ham_index:
            # solver will go to SS otherwise
            if len(max_ham_sliced_time_list) > 1:
                rho_f = mesolve(
                    H_max_sliced,
                    rho_next,
                    max_ham_sliced_time_list,
                    c_op_lists[i],
                    [],
                    args=args,
                    options=options,
                ).states
            else:
                rho_f = rho_next

            # max_return_idx = max(exp_return_indices[i])
            if exp_return_indices[i]:
                for idx in exp_return_indices[i]:
                    if idx == 0:
                        if e_op is None:
                            exp_return.append(rho_next)
                        else:
                            exp_return.append(expect(e_op, rho_next))
                    else:
                        if e_op is None:
                            exp_return.append(rho_f[idx - 1])
                        else:
                            # since rho_first_evo is smaller than rho_first evo by one we need to return the i-1st index
                            exp_return.append(expect(e_op, rho_f[idx - 1]))

    print(
        f"Time exp_eval_floating_start_finish() simulation took: {time.time()-time_sim_start}"
    )

    if e_op is None:
        return exp_return
    else:
        return np.array(exp_return)


def rho_evo_floating_start_finish(
    h_list,
    t_sim_lists,
    state0,
    t_start,
    t_finish,
    c_ops_list,
    a_op,
    b_op,
    args={},
    options=Options(),
    debug=False,
):
    """
    Internal function for calculating the evolution of a density matrix with some operators
    <A(t) B(t)>
    using a master equation solver, returns the density matrix at the final time
    input args:
    H_list: list of Hamiltonians for each t_sim_lists
    t_sim_lists: list of simulation timesteps for each Hamiltonian
    state0: initial state at start time
    t_start: start_time
    t_finish: finish_time
    c_ops: list of collapse operators
    a_op: operator which to evolve from left
    b_op: operator which to evolve from right
    args: arguments for the solver
    """
    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    # create list of boundaries for each time list define hamiltonian simulation times
    time_h_bounds = []
    t_bound = 0
    for sim_time in t_sim_lists:
        t_bound += sim_time[-1]
        time_h_bounds.append(t_bound)

    if t_finish > max(time_h_bounds):
        raise ValueError("t_finish must not exceed simulation time list max")

    max_ham_index = _find_next_largest_index(time_h_bounds, t_finish)

    min_ham_index = _find_next_smallest_index(time_h_bounds, t_start)

    if debug:
        print(f"Time bounds: {time_h_bounds}")
        print(f"t_start: {t_start}")
        print(f"t_finish: {t_finish}")
        print(f"Min ham index: {min_ham_index}")
        print(f"Max ham index: {max_ham_index}")

    if min_ham_index == 0:
        min_ham_time_idx = _find_closest_value_index(
            t_sim_lists[min_ham_index], t_start
        )
    else:
        min_ham_time_idx = _find_closest_value_index(
            t_sim_lists[min_ham_index], t_start - time_h_bounds[min_ham_index - 1]
        )
    min_ham_sliced_time_list = t_sim_lists[min_ham_index][
        min_ham_time_idx : len(t_sim_lists[min_ham_index])
    ]

    H_min_sliced = []
    for elem in h_list[min_ham_index]:
        H_min_sliced.append(
            [elem[0], elem[1][min_ham_time_idx : len(t_sim_lists[min_ham_index])]]
        )

    # print("t_sim_lists[max_ham_index]", t_sim_lists[max_ham_index])
    # print("t_finish- time_h_bounds[max_ham_index]", t_finish- time_h_bounds[max_ham_index])
    # print("t_finish", t_finish)
    # print("time_h_bounds[max_ham_index]", time_h_bounds[max_ham_index])
    max_ham_time_idx = _find_closest_value_index(
        t_sim_lists[max_ham_index], t_finish - time_h_bounds[max_ham_index - 1]
    )
    max_ham_sliced_time_list = t_sim_lists[max_ham_index][0:max_ham_time_idx]
    if debug:
        print(f"Min ham time idx: {min_ham_time_idx}")
        print(f"Max ham time idx: {max_ham_time_idx}")
    H_max_sliced = []
    for elem in h_list[max_ham_index]:
        H_max_sliced.append([elem[0], elem[1][0:max_ham_time_idx]])
    assert len(H_min_sliced[0][1]) == len(min_ham_sliced_time_list)
    assert len(H_max_sliced[0][1]) == len(max_ham_sliced_time_list)

    time_sim_start = time.time()
    for i, H in enumerate(h_list):
        if i == min_ham_index:
            # otherwise the solver will go to SS
            if len(min_ham_sliced_time_list) > 1:
                first_evolution = mesolve(
                    H_min_sliced,
                    a_op * rho0 * b_op,
                    min_ham_sliced_time_list,
                    c_ops_list[i],
                    [],
                    args=args,
                    options=options,
                ).states
                rho0 = first_evolution[-1]
                rho_return = rho0
            else:
                rho_return = rho0

        elif i < max_ham_index and i > min_ham_index:
            rho = mesolve(
                H, rho0, t_sim_lists[i], c_ops_list[i], [], args=args, options=options
            ).states
            rho0 = rho[-1]
            # print(f"rho0: {rho0}")
        elif i == max_ham_index:
            # solver will go to SS otherwise
            if len(max_ham_sliced_time_list) > 1:
                rho_f = mesolve(
                    H_max_sliced,
                    rho0,
                    max_ham_sliced_time_list,
                    c_ops_list[i],
                    [],
                    args=args,
                    options=options,
                ).states
                rho_return = rho_f[-1]
            else:
                rho_return = rho0

    print(
        f"Time rho_evo_floating_start_finish() sim took: {time.time()-time_sim_start}"
    )

    return rho_return


def exp_eval_fixed_start(
    h_list,
    t_sim_lists,
    state0,
    tlist,
    c_ops_list,
    a_op,
    b_op,
    e_op,
    args={},
    options=Options(store_final_state=True),
    debug=False,
):
    """
    Internal function for calculating the expectation value of an operator
    <E(t)> whilst following some evolution of the state <A rho B >
    using a master equation solver.
    input args:
    H_list: list of Hamiltonians for each t_sim_lists
    t_sim_lists: list of simulation time steps for each Hamiltonian
    state0: initial state
    tlist: list of times to evaluate the expectation value
    c_ops: list of collapse operators
    a_op: operator which to evolve from left
    b_op: operator which to evolve from right
    e_op: operator for expectation value
    args: arguments for the solver
    """

    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    # create list of boundaries for each time list define hamiltonian simulation times
    time_h_bounds = []
    t_bound = 0
    for sim_time in t_sim_lists:
        t_bound += sim_time[-1]
        time_h_bounds.append(t_bound)

    if tlist[-1] > max(time_h_bounds):
        raise ValueError("tlist must not exceed simulation time list max")
    else:
        max_time = max(tlist)
        max_ham_index = _find_next_largest_index(time_h_bounds, max_time)
        exp_return_indices = []
        for i in range(len(h_list)):
            exp_return_indices.append([])

    for t in tlist:
        # cannot be smaller than 0
        for i, time_bound in enumerate(time_h_bounds):
            if t <= time_bound:
                if i == 0:
                    ham_index = _find_closest_value_index(t_sim_lists[i], t)
                else:
                    ham_index = _find_closest_value_index(
                        t_sim_lists[i], t - time_h_bounds[i - 1]
                    )

                exp_return_indices[i].append(ham_index)
                break

    if debug:
        print(f"Max ham index: {max_ham_index}")
        print(f"Exp Return Indices: {exp_return_indices}")
        print(f"t_h_bounds: {time_h_bounds}")

    exp_return = []

    t_start = time.time()

    for i, H in enumerate(h_list):
        if i == 0:
            # TODO only run a single master equation solver for final states and then evaluate expectation values separately with expect
            # first_evolution_exp= mesolve(H, a_op*rho0*b_op, t_sim_lists[i], c_ops, [e_op], args=args, options=options).expect[0]
            first_evolution_rho = mesolve(
                H,
                a_op * rho0 * b_op,
                t_sim_lists[i],
                c_ops_list[i],
                [],
                args=args,
                options=options,
            ).states

            if len(first_evolution_rho) > 1:
                rho0 = first_evolution_rho[-1]
            else:
                rho0 = first_evolution_rho[0]

            # set exp output
            for idx in exp_return_indices[i]:
                exp_return.append(expect(first_evolution_rho[idx], e_op))
        elif i <= max_ham_index:
            # TODO only run a single master equation solver for final states and then evaluate expectation values separately with expect
            # rho_exp= mesolve(H, rho0, t_sim_lists[i], c_ops, [e_op], args=args, options=options).expect[0]
            rho_second_evo = mesolve(
                H, rho0, t_sim_lists[i], c_ops_list[i], [], args=args, options=options
            ).states
            rho0 = rho_second_evo[-1]
            # set rho output
            for idx in exp_return_indices[i]:
                exp_return.append(expect(rho_second_evo[idx], e_op))

    print(f"Time exp_eval_fixed_start() sim took: {time.time()-t_start}")

    return np.array(exp_return)


def rho_evo_fixed_start(
    h_list,
    t_sim_lists,
    state0,
    tlist,
    c_ops_list,
    a_op,
    b_op,
    args={},
    options=Options(),
    debug=False,
):
    """
    Internal function for calculating the evolution of a density matrix with some operators
    <A(t) B(t)>
    using a master equation solver.
    input args:
    H_list: list of Hamiltonians for each t_sim_lists
    t_sim_lists: list of simulation times for each Hamiltonian
    state0: initial state at T=0
    tlist: list of times to evaluate rho
    t_sim_list: list of simulation timesteps for each Hamiltonian
    c_ops: list of collapse operators
    a_op: operator which to evolve from left
    b_op: operator which to evolve from right
    args: arguments for the solver
    """

    # the solvers only work for positive time differences and the correlators
    # require positive tau
    if isket(state0):
        rho0 = ket2dm(state0)
    else:
        rho0 = state0

    # create list of boundaries for each time list define hamiltonian simulation times
    time_h_bounds = []
    t_bound = 0
    for sim_time in t_sim_lists:
        t_bound += sim_time[-1]
        time_h_bounds.append(t_bound)

    if debug:
        print(f"Time bounds: {time_h_bounds}")

    if tlist[-1] > max(time_h_bounds):
        raise ValueError("tlist must not exceed simulation time list max")
    else:
        max_time = max(tlist)
        max_ham_index = _find_next_largest_index(time_h_bounds, max_time)
        rho_return_indices = []
        for i in range(len(h_list)):
            rho_return_indices.append([])

    for t in tlist:
        for i, time_bound in enumerate(time_h_bounds):
            if t <= time_bound:
                if i == 0:
                    ham_index = _find_closest_value_index(t_sim_lists[i], t)
                else:
                    ham_index = _find_closest_value_index(
                        t_sim_lists[i], t - time_h_bounds[i - 1]
                    )
                rho_return_indices[i].append(ham_index)
                break
    if debug:
        print(f"Rho return indices {rho_return_indices}")
        print(f"Max ham index: {max_ham_index}")

    rho_return = []

    t_start = time.time()
    for i, H in enumerate(h_list):
        if i == 0:
            first_evolution = mesolve(
                H,
                a_op * rho0 * b_op,
                t_sim_lists[i],
                c_ops_list[i],
                [],
                args=args,
                options=options,
            ).states
            rho0 = first_evolution[-1]
            # set rho output
            for idx in rho_return_indices[i]:
                if idx:
                    if idx == 0:
                        rho_return.append(a_op * rho0 * b_op)
                    else:
                        rho_return.append(first_evolution[idx - 1])
        elif i <= max_ham_index:
            rho = mesolve(
                H, rho0, t_sim_lists[i], c_ops_list[i], [], args=args, options=options
            ).states
            rho0 = rho[-1]
            # set rho output
            for idx in rho_return_indices[i]:
                if idx:
                    if idx == 0:
                        rho_return.append(rho0)
                    else:
                        rho_return.append(rho[idx - 1])

    print(f"Time rho_evo_fixed_start() sim took: {time.time()-t_start}")

    return rho_return

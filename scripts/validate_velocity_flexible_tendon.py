from functools import partial
import timeit
from typing import Callable

from casadi import MX
from matplotlib import pyplot as plt
from musculotendon_ocp import (
    RigidbodyModelWithMuscles,
    RigidbodyModels,
    MuscleHillModels,
    ComputeForceDampingMethods,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
)
import numpy as np
from scipy.integrate import solve_ivp


def compute_muscle_lengths(model: RigidbodyModelWithMuscles, all_muscle_fiber_lengths: np.ndarray) -> list[np.ndarray]:
    # Dispatch so the outer list is the muscles and the inner list is the time points (opposite of the current structure)
    out = [None] * model.nb_muscles
    for i in range(model.nb_muscles):
        out[i] = np.array(all_muscle_fiber_lengths[i, :])
    return out


def compute_finitediff(array: np.ndarray, t: np.ndarray) -> np.ndarray:
    finitediff = np.zeros(len(t))
    finitediff[1:-1] = (array[2:] - array[:-2]) / (t[2] - t[0])
    return finitediff


def compute_muscle_fiber_velocities(
    model: RigidbodyModelWithMuscles,
    activations: np.ndarray,
    all_muscle_lengths: np.ndarray,
    all_q: np.ndarray,
    all_qdot: np.ndarray,
) -> np.ndarray:
    lmdot = [np.ndarray(len(all_q.T)) for _ in range(model.nb_muscles)]

    muscle_fiber_lengths_dot_func = model.to_casadi_function(
        model.muscle_fiber_velocities, "activations", "q", "qdot", "muscle_fiber_lengths"
    )

    for i, (lengths, q, qdot) in enumerate(zip(all_muscle_lengths.T, all_q.T, all_qdot.T)):
        lmdot_all_muscles = muscle_fiber_lengths_dot_func(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=lengths,
        )["output"].__array__()
        for m, vel_muscle in enumerate(lmdot_all_muscles):
            lmdot[m][i] = vel_muscle

    return lmdot


last_computed_lmdot = [np.array([0])]


def dynamics(
    _,
    x,
    dynamics_functions: list[Callable],
    model: RigidbodyModelWithMuscles,
    activations: np.ndarray,
    is_linearized: bool,
) -> np.ndarray:
    muscle_fiber_lmdot_func, forward_dynamics_func = dynamics_functions

    fiber_lengths = x[: model.nb_muscles]
    q = x[model.nb_muscles : model.nb_muscles + model.nb_q]
    qdot = x[model.nb_muscles + model.nb_q :]

    fiber_lengths_dot = muscle_fiber_lmdot_func(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=fiber_lengths,
        muscle_fiber_velocities=0,  # if is_linearized else last_computed_lmdot[-1],
    )["output"].__array__()[:, 0]
    last_computed_lmdot.append(fiber_lengths_dot)

    qddot = forward_dynamics_func(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=fiber_lengths,
        muscle_fiber_velocities=fiber_lengths_dot,
    )["output"].__array__()[:, 0]

    return np.concatenate((fiber_lengths_dot, qdot, qddot))


def prepare_muscle_fiber_velocities(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    muscle_fiber_velocities = model.muscle_fiber_velocities(
        activations=activations, q=q, qdot=qdot, muscle_fiber_lengths=model.muscle_fiber_lengths_mx
    )
    return muscle_fiber_velocities


def prepare_forward_dynamics(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    tau = model.muscle_joint_torque(activations, q, qdot, muscle_fiber_lengths=model.muscle_fiber_lengths_mx)
    qddot = model.forward_dynamics(q, qdot, tau)
    return qddot


def main(compute_muscle_fiber_velocity_method: ComputeMuscleFiberVelocityMethods):
    model = RigidbodyModels.WithMuscles(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium(),
                compute_muscle_fiber_velocity=compute_muscle_fiber_velocity_method(),
            ),
        ],
    )

    dt_steps = 0.01
    n_substeps = 10

    all_t_span = (0, 0.5)
    t = np.linspace(*all_t_span, 10000)
    q = np.ones(model.nb_q) * -0.24
    qdot = np.zeros(model.nb_qdot)
    activations = np.ones(model.nb_muscles) * 0.6
    initial_muscle_fiber_length = np.array(
        model.function_to_dm(model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot)
    )[:, 0]

    # Request the integration of the equations of motion
    fiber_velocity_func = model.to_casadi_function(
        partial(prepare_muscle_fiber_velocities, model=model), "activations", "q", "qdot"
    )
    forward_dynamics_func = model.to_casadi_function(
        partial(prepare_forward_dynamics, model=model), "activations", "q", "qdot"
    )
    dynamics_functions = partial(
        dynamics,
        dynamics_functions=(fiber_velocity_func, forward_dynamics_func),
        model=model,
        activations=activations,
        is_linearized=compute_muscle_fiber_velocity_method
        == ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized,
    )

    time_vector = [np.array([0])]
    integrated_values = [np.concatenate((initial_muscle_fiber_length, q, qdot)).reshape(-1, 1)]
    n_steps = int(all_t_span[1] / dt_steps)
    if n_steps != all_t_span[1] / dt_steps:
        raise ValueError("The final time should be a multiple of the time step")

    for i in range(n_steps):
        t_span = (i * dt_steps, (i + 1) * dt_steps)

        time_vector.append(np.linspace(*t_span, n_substeps + 1)[1:])
        integrated_values.append(
            solve_ivp(
                dynamics_functions,
                t_span,
                np.concatenate((initial_muscle_fiber_length, q, qdot)),
                t_eval=time_vector[-1],
            ).y
        )

        if integrated_values[-1].shape[1] != n_substeps:
            raise ValueError("Integration failed")

        initial_muscle_fiber_length = integrated_values[-1][: model.nb_muscles, -1]
        q = integrated_values[-1][model.nb_muscles : model.nb_muscles + model.nb_q, -1]
        qdot = integrated_values[-1][model.nb_muscles + model.nb_q :, -1]

    # Concatenate the results
    t = np.concatenate(time_vector)
    integrated = np.concatenate(integrated_values, axis=1)

    muscle_fiber_lengths_int = integrated[: model.nb_muscles, :]
    q_int = integrated[model.nb_muscles : model.nb_muscles + model.nb_q, :]
    qdot_int = integrated[model.nb_muscles + model.nb_q :, :]

    # Compute muscle velocities from finite difference as benchmark
    muscle_lengths = compute_muscle_lengths(model, muscle_fiber_lengths_int)
    muscle_fiber_velocities_finitediff = [compute_finitediff(length, t) for length in muscle_lengths]

    # Compute muscle velocities from jacobian
    muscle_fiber_velocities_computed = compute_muscle_fiber_velocities(
        model, activations, muscle_fiber_lengths_int, q_int, qdot_int
    )

    # If the two methods are equivalent, the plot should be on top of each other
    plt.figure("Generalized coordinates")
    for q in q_int:
        plt.plot(t, q)
    plt.xlabel("Time (s)")
    plt.ylabel("Generalized coordinate (rad)")

    plt.figure("Generalized velocities")
    for qdot in qdot_int:
        plt.plot(t, qdot)
    plt.xlabel("Time (s)")
    plt.ylabel("Generalized velocity (rad/s)")

    plt.figure("Generalized accelerations")
    qddot_finitediff = [compute_finitediff(qdot, t) for qdot in qdot_int]
    for qddot in qddot_finitediff:
        plt.plot(t, qddot)
    plt.xlabel("Time (s)")
    plt.ylabel("Generalized acceleration (rad/s^2)")

    plt.figure("Muscle lengths")
    for length in muscle_lengths:
        plt.plot(t, length)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle length (m)")

    plt.figure("Muscle velocities")
    for finitediff, computed in zip(muscle_fiber_velocities_finitediff, muscle_fiber_velocities_computed):
        plt.plot(t, finitediff)
        plt.plot(t, computed)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle velocity (m/s)")
    plt.legend([f"Finite diff", f"Computed"])


if __name__ == "__main__":
    # Time the running of the script

    repeat = 50
    print(f"Timing the script with {repeat} repeats")

    print("Timing the script using FlexibleTendonImplicit")
    implicit_timing = (
        timeit.timeit(
            "main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit)",
            setup="from __main__ import main, ComputeMuscleFiberVelocityMethods",
            number=repeat,
        )
        / repeat
    )

    print("Timing the script using FlexibleTendonExplicit")
    explicit_timing = (
        timeit.timeit(
            "main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit)",
            setup="from __main__ import main, ComputeMuscleFiberVelocityMethods",
            number=repeat,
        )
        / repeat
    )

    print("Timing the script using FlexibleTendonLinearized")
    linearized_timing = (
        timeit.timeit(
            "main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized)",
            setup="from __main__ import main, ComputeMuscleFiberVelocityMethods",
            number=repeat,
        )
        / repeat
    )
    print("Timings is over")

    # Implicit fails, so it needs "error_on_fail" to be set to False
    print("Showing the plots")
    main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit)
    main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit)
    main(compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized)

    print(f"Implicit: {implicit_timing:.3f} s")
    print(f"Explicit: {explicit_timing:.3f} s")
    print(f"Linearized: {linearized_timing:.3f} s")

    plt.show()

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
from musculotendon_ocp.math import compute_finitediff, precise_rk45, precise_rk4
import numpy as np


def compute_muscle_lengths(model: RigidbodyModelWithMuscles, all_muscle_fiber_lengths: np.ndarray) -> list[np.ndarray]:
    # Dispatch so the outer list is the muscles and the inner list is the time points (opposite of the current structure)
    out = [None] * model.nb_muscles
    for i in range(model.nb_muscles):
        out[i] = np.array(all_muscle_fiber_lengths[i, :])
    return out


def compute_muscle_fiber_velocities(
    model: RigidbodyModelWithMuscles,
    activations: np.ndarray,
    all_muscle_lengths: np.ndarray,
    all_q: np.ndarray,
    all_qdot: np.ndarray,
) -> np.ndarray:
    lmdot = [np.ndarray(len(all_q.T)) for _ in range(model.nb_muscles)]

    muscle_fiber_lengths_dot_func = model.to_casadi_function(
        model.muscle_fiber_velocities,
        "activations",
        "q",
        "qdot",
        "muscle_fiber_lengths",
        "muscle_fiber_velocity_initial_guesses",
    )

    lmdot_all_muscles = np.zeros((model.nb_muscles,))
    for i, (lengths, q, qdot) in enumerate(zip(all_muscle_lengths.T, all_q.T, all_qdot.T)):
        lmdot_all_muscles = muscle_fiber_lengths_dot_func(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=lengths,
            muscle_fiber_velocity_initial_guesses=lmdot_all_muscles,
        )["output"].__array__()
        for m, vel_muscle in enumerate(lmdot_all_muscles):
            lmdot[m][i] = vel_muscle[0]

    return lmdot


last_computed_lmdot = [np.array([0])]


def dynamics(
    _,
    x,
    dynamics_functions: list[Callable],
    model: RigidbodyModelWithMuscles,
    activations: np.ndarray,
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
        muscle_fiber_velocity_initial_guesses=last_computed_lmdot[-1],
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


def prepare_fiber_lmdot(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    fiber_lmdot = model.muscle_fiber_velocities(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=model.muscle_fiber_lengths_mx,
        muscle_fiber_velocity_initial_guesses=model.muscle_fiber_velocity_initial_guesses_mx,
    )
    return fiber_lmdot


def prepare_forward_dynamics(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    tau = model.muscle_joint_torque(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=model.muscle_fiber_lengths_mx,
        muscle_fiber_velocities=model.muscle_fiber_velocities_mx,
    )
    qddot = model.forward_dynamics(q, qdot, tau)
    return qddot


def main(
    compute_muscle_fiber_velocity_method: ComputeMuscleFiberVelocityMethods,
    integration_method: Callable = precise_rk45,
    skip_graphs: bool = False,
    color: str = "k",
) -> None:
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
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=compute_muscle_fiber_velocity_method,
            ),
        ],
    )

    dt = 0.001 if integration_method == precise_rk4 else 0.005
    t_span = (0, 0.5)
    activations = np.ones(model.nb_muscles) * 0.2
    q = np.ones(model.nb_q) * -0.24
    qdot = np.zeros(model.nb_qdot)
    initial_muscle_fiber_length = np.array(
        model.function_to_dm(model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot)
    )[:, 0]
    y0 = np.concatenate((initial_muscle_fiber_length, q, qdot))

    # Request the integration of the equations of motion
    fiber_lmdot_func = model.to_casadi_function(partial(prepare_fiber_lmdot, model=model), "activations", "q", "qdot")
    forward_dynamics_func = model.to_casadi_function(
        partial(prepare_forward_dynamics, model=model), "activations", "q", "qdot"
    )
    dynamics_functions = partial(
        dynamics,
        dynamics_functions=(fiber_lmdot_func, forward_dynamics_func),
        model=model,
        activations=activations,
    )
    t, integrated = integration_method(dynamics_functions, y0, t_span, dt)

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
    if not skip_graphs:
        plt.figure("Generalized coordinates")
        for q in q_int:
            plt.plot(t, q, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Generalized coordinate (rad)")

        plt.figure("Generalized velocities")
        for qdot in qdot_int:
            plt.plot(t, qdot, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Generalized velocity (rad/s)")

        plt.figure("Generalized accelerations")
        qddot_finitediff = [compute_finitediff(qdot, t) for qdot in qdot_int]
        for qddot in qddot_finitediff:
            plt.plot(t, qddot, color=color, linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Generalized acceleration (rad/s^2)")

        plt.figure("Muscle lengths")
        for length in muscle_lengths:
            plt.plot(t, length, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle length (m)")

        plt.figure("Muscle velocities")
        for finitediff, computed in zip(muscle_fiber_velocities_finitediff, muscle_fiber_velocities_computed):
            plt.plot(t, finitediff, color=color, linestyle="--")
            plt.plot(t, computed, color=color)
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle velocity (m/s)")
        plt.legend([f"Finite diff", f"Computed"])


def time_main(methods: list[ComputeMuscleFiberVelocityMethods], repeat: int) -> list[float]:
    """
    Time the main function with different methods

    Parameters
    ----------
    methods: list[ComputeMuscleFiberVelocityMethods]
        The methods to time
    repeat: int
        The number of repeats

    Returns
    -------
    list[float]
        The timings for each method
    """
    timings = {}
    for method in methods:
        print(f"Timing the script using {method}")
        timings[method] = (
            timeit.timeit(
                f"main(compute_muscle_fiber_velocity_method={method}, skip_graphs=True, integration_method=precise_rk4)",
                setup="from __main__ import main, ComputeMuscleFiberVelocityMethods, precise_rk4",
                number=repeat,
            )
            / repeat
        )
    print("Timings is over")

    return timings


if __name__ == "__main__":
    print("Preparing the plots")

    main(
        compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
        color="r",
    )
    main(
        compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit(),  # TODO ForceDefect
        color="g",
    )
    main(
        compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit(),  # TODO VelocityDefect
        color="b",
    )
    main(
        compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit(),
        color="#00A5FF",
        integration_method=precise_rk4,
    )
    main(
        compute_muscle_fiber_velocity_method=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
        color="k",
    )

    repeat = 20
    print(f"Timing the script, each method will be repeated {repeat} times. This may take a while")
    timings = time_main(
        [
            "ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit()",
            "ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit()",
            "ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized()",
            "ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic()",
        ],
        repeat=repeat,
    )
    for timing in timings:
        print(f"{timing} took {timings[timing]} seconds")

    plt.show()

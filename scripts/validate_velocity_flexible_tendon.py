from functools import partial
from typing import Callable

from casadi import MX, vertcat
from matplotlib import pyplot as plt
from musculotendon_ocp import (
    MuscleBiorbdModel,
    MuscleModelHillFlexibleTendon,
    ComputeMuscleFiberLengthAsVariable,
    ComputeMuscleFiberVelocityFlexibleTendon,
    ForceDampingLinear,
)
import numpy as np
from scipy.integrate import solve_ivp


def compute_muscle_lengths(model: MuscleBiorbdModel, all_muscle_fiber_lengths: np.ndarray) -> list[np.ndarray]:
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
    model: MuscleBiorbdModel,
    activations: np.ndarray,
    all_muscle_lengths: np.ndarray,
    all_q: np.ndarray,
    all_qdot: np.ndarray,
) -> np.ndarray:
    velocities = [np.ndarray(len(all_q.T)) for _ in range(model.nb_muscles)]

    muscle_fiber_velocities_func = model.to_casadi_function(model.muscle_fiber_velocities, "activations", "q", "qdot")

    for i, (lengths, q, qdot) in enumerate(zip(all_muscle_lengths.T, all_q.T, all_qdot.T)):
        vel_all_muscles = muscle_fiber_velocities_func(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_lengths=lengths,
        )["output"].__array__()
        for m, vel_muscle in enumerate(vel_all_muscles):
            velocities[m][i] = vel_muscle

    return velocities


# TODO ipuch should we use this
last_computed_velocity = [np.array([0])]


def dynamics(_, x, dynamics_functions: list[Callable], model: MuscleBiorbdModel, activations: np.ndarray) -> np.ndarray:
    # TODO Find a way to initialize muscle_velocities?
    muscle_fiber_velocity_func, forward_dynamics_func = dynamics_functions

    fiber_lengths = x[: model.nb_muscles]
    q = x[model.nb_muscles : model.nb_muscles + model.nb_q]
    qdot = x[model.nb_muscles + model.nb_q :]

    fiber_lengths_dot = muscle_fiber_velocity_func(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=fiber_lengths,
        # muscle_fiber_velocities=last_computed_velocity[-1],
    )["output"].__array__()[:, 0]
    last_computed_velocity.append(fiber_lengths_dot)

    qddot = forward_dynamics_func(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=fiber_lengths,
        muscle_fiber_velocities=fiber_lengths_dot,
    )["output"].__array__()[:, 0]

    return np.concatenate((fiber_lengths_dot, qdot, qddot))


def prepare_muscle_fiber_velocities(model: MuscleBiorbdModel, activations: MX, q: MX, qdot: MX) -> MX:
    muscle_fiber_velocities = model.muscle_fiber_velocities(activations=activations, q=q, qdot=qdot)
    return muscle_fiber_velocities


def prepare_forward_dynamics(model: MuscleBiorbdModel, activations: MX, q: MX, qdot: MX) -> MX:
    tau = model.muscle_joint_torque(activations, q, qdot)
    qddot = model.forward_dynamics(q, qdot, tau)
    return qddot


def main():
    model = MuscleBiorbdModel(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleModelHillFlexibleTendon(
                name="Mus1",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                force_damping=ForceDampingLinear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthAsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityFlexibleTendon(),
            ),
        ],
    )

    t_span = (0, 1.5)
    t = np.linspace(*t_span, 10000)
    q = np.ones(model.nb_q) * -0.24
    qdot = np.zeros(model.nb_qdot)
    activations = np.ones(model.nb_muscles) * 0.1
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
    integrated = solve_ivp(
        partial(
            dynamics,
            dynamics_functions=(fiber_velocity_func, forward_dynamics_func),
            model=model,
            activations=activations,
        ),
        t_span,
        np.concatenate((initial_muscle_fiber_length, q, qdot)),
        t_eval=t,
    ).y
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
    plt.show()


if __name__ == "__main__":
    main()

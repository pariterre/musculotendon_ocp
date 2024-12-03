from functools import partial
from typing import Callable

from casadi import MX
from matplotlib import pyplot as plt
from musculotendon_ocp import RigidbodyModelWithMuscles, RigidbodyModels, MuscleHillModels
from musculotendon_ocp.math import compute_finitediff, precise_rk45
import numpy as np


def compute_muscle_lengths(model: RigidbodyModelWithMuscles, all_q: np.ndarray) -> list[np.ndarray]:
    func = model.to_casadi_function(model.muscle_fiber_lengths, "activations", "q", "qdot")
    values = [func(q=q)["output"] for q in all_q.T]

    # Dispatch so the outer list is the muscles and the inner list is the time points (opposite of the current structure)
    out = [None] * model.nb_muscles
    for i in range(model.nb_muscles):
        out[i] = np.array([float(value[i][0]) for value in values])
    return out


def compute_muscle_fiber_velocities(
    model: RigidbodyModelWithMuscles, all_q: np.ndarray, all_qdot: np.ndarray
) -> np.ndarray:
    velocities = [np.ndarray(len(all_q.T)) for _ in range(model.nb_muscles)]

    muscle_tendon_length_jacobian_func = model.to_casadi_function(model.muscle_tendon_length_jacobian, "q")

    for i, (q, qdot) in enumerate(zip(all_q.T, all_qdot.T)):
        jac = muscle_tendon_length_jacobian_func(q=q)["output"]
        vel_all_muscles = np.array(jac @ qdot)
        for m, vel_muscle in enumerate(vel_all_muscles):
            velocities[m][i] = vel_muscle[0]

    return velocities


def dynamics(_, x, dynamics_func: Callable, model: RigidbodyModelWithMuscles, activations: np.ndarray) -> np.ndarray:
    q = x[: model.nb_q]
    qdot = x[model.nb_q :]

    qddot = dynamics_func(activations=activations, q=q, qdot=qdot)["output"].__array__()[:, 0]

    return np.concatenate((qdot, qddot))


def qddot_from_muscles(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    muscle_fiber_lengths = model.muscle_fiber_lengths(activations=activations, q=q, qdot=qdot)
    muscle_fiber_velocities = model.muscle_fiber_velocities(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=muscle_fiber_lengths,
        muscle_fiber_velocity_initial_guesses=np.array([0.0] * model.nb_muscles),
    )
    tau = model.muscle_joint_torque(
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=muscle_fiber_lengths,
        muscle_fiber_velocities=muscle_fiber_velocities,
    )
    return model.forward_dynamics(q, qdot, tau)


def main():
    model = RigidbodyModels.WithMuscles(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
        ],
    )

    dt = 0.005
    t_span = (0, 2.5)
    q = np.ones(model.nb_q) * -0.2
    qdot = np.zeros(model.nb_qdot)
    activations = np.ones(model.nb_muscles) * 1.0

    # Request the integration of the equations of motion
    dynamics_func = model.to_casadi_function(partial(qddot_from_muscles, model=model), "activations", "q", "qdot")
    t, integrated = precise_rk45(
        partial(dynamics, dynamics_func=dynamics_func, model=model, activations=activations),
        y0=np.concatenate((q, qdot)),
        t_span=t_span,
        dt=dt,
    )
    q_int = integrated[: model.nb_q, :]
    qdot_int = integrated[model.nb_q :, :]

    # Compute muscle velocities from finite difference as benchmark
    muscle_lengths = compute_muscle_lengths(model, q_int)
    muscle_fiber_velocities_finitediff = [compute_finitediff(length, t) for length in muscle_lengths]

    # Compute muscle velocities from jacobian
    muscle_fiber_velocities_jacobian = compute_muscle_fiber_velocities(model, q_int, qdot_int)

    # If the two methods are equivalent, the plot should be on top of each other
    plt.figure("Muscle lengths")
    for length in muscle_lengths:
        plt.plot(t, length)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle length (m)")

    plt.figure("Muscle velocities")
    for finitediff, jacobian in zip(muscle_fiber_velocities_finitediff, muscle_fiber_velocities_jacobian):
        plt.plot(t, finitediff)
        plt.plot(t, jacobian)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle velocity (m/s)")
    plt.legend([f"Finite diff", f"Jacobian"])
    plt.show()


if __name__ == "__main__":
    main()

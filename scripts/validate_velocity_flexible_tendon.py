from functools import partial
from typing import Callable

from casadi import MX, vertcat
from matplotlib import pyplot as plt
from musculotendon_ocp import MuscleBiorbdModel, MuscleModelHillFlexibleTendon, ComputeMuscleFiberLengthAsVariable
import numpy as np
from scipy.integrate import solve_ivp


def compute_muscle_lengths(model: MuscleBiorbdModel, all_q: np.ndarray) -> list[np.ndarray]:
    func = model.to_casadi_function(model.muscle_fiber_lengths, "q")
    values = [model.evaluate_function(func, q=q) for q in all_q.T]

    # Dispatch so the outer list is the muscles and the inner list is the time points (opposite of the current structure)
    out = [None] * model.nb_muscles
    for i in range(model.nb_muscles):
        out[i] = np.array([float(value[i][0]) for value in values])
    return out


def muscle_fiber_velocity_from_finitediff(lengths: np.ndarray, t: np.ndarray) -> np.ndarray:
    finitediff = np.zeros(len(t))
    finitediff[1:-1] = (lengths[2:] - lengths[:-2]) / (t[2] - t[0])
    return finitediff


def compute_muscle_fiber_velocities(model: MuscleBiorbdModel, all_q: np.ndarray, all_qdot: np.ndarray) -> np.ndarray:
    velocities = [np.ndarray(len(all_q.T)) for _ in range(model.nb_muscles)]

    muscle_tendon_length_jacobian_func = model.to_casadi_function(model.muscle_tendon_length_jacobian, "q")

    for i, (q, qdot) in enumerate(zip(all_q.T, all_qdot.T)):
        jac = model.evaluate_function(muscle_tendon_length_jacobian_func, q=q)
        vel_all_muscles = np.array(jac @ qdot)
        for m, vel_muscle in enumerate(vel_all_muscles):
            velocities[m][i] = vel_muscle

    return velocities


def dynamics(_, x, dynamics_func: Callable, model: MuscleBiorbdModel, activations: np.ndarray) -> np.ndarray:
    fiber_lengths = x[: model.nb_muscles]
    q = x[model.nb_muscles : model.nb_muscles + model.nb_q]
    qdot = x[model.nb_muscles + model.nb_q :]

    fiber_lengths_dot, qddot = model.evaluate_function(
        dynamics_func, activations=activations, q=q, qdot=qdot, muscle_lengths=fiber_lengths
    ).__array__()[:, 0]

    return np.concatenate((fiber_lengths_dot, qdot, qddot))


def dynamics_as_functions(model: MuscleBiorbdModel, activations: MX, q: MX, qdot: MX) -> MX:
    # TODO RENDU ICI!!! Compléter après l'autre TODO
    fiber_lengths_dot = model.muscle_fiber_velocities(
        model.muscle_fiber_len, activation=activations, muscle_fiber_length=fiber_lengths
    )

    tau = model.muscle_joint_torque(activations, q, qdot)
    qddot = model.forward_dynamics(q, qdot, tau)
    return fiber_lengths_dot, qddot


def main():
    model = MuscleBiorbdModel(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleModelHillFlexibleTendon(
                name="Mus1",
                maximal_force=500,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthAsVariable(),
            ),
        ],
    )

    t_span = (0, 2.5)
    t = np.linspace(*t_span, 1000)
    q = np.ones(model.nb_q) * -0.2
    qdot = np.zeros(model.nb_qdot)
    activations = np.ones(model.nb_muscles) * 1.0
    initial_muscle_fiber_length = np.array(
        model.function_to_dm(model.muscle_tendon_lengths, q=q)
        - vertcat(*[mus.tendon_slack_length for mus in model.muscles])
    )[:, 0]

    # Request the integration of the equations of motion
    dynamics_func = model.to_casadi_function(partial(dynamics_as_functions, model=model), "activations", "q", "qdot")
    integrated = solve_ivp(
        partial(dynamics, dynamics_func=dynamics_func, model=model, activations=activations),
        t_span,
        np.concatenate((initial_muscle_fiber_length, q, qdot)),
        t_eval=t,
    ).y
    q_int = integrated[: model.nb_q, :]
    qdot_int = integrated[model.nb_q :, :]

    # Compute muscle velocities from finite difference as benchmark
    muscle_lengths = compute_muscle_lengths(model, q_int)
    muscle_fiber_velocities_finitediff = [muscle_fiber_velocity_from_finitediff(length, t) for length in muscle_lengths]

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

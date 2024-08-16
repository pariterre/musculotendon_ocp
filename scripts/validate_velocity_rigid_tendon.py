from functools import partial

from matplotlib import pyplot as plt
from musculotendon_ocp import MuscleBiorbdModel, MuscleModelHillRigidTendon
import numpy as np
from scipy.integrate import solve_ivp


def compute_muscle_lengths(model: MuscleBiorbdModel, all_q: np.ndarray) -> list[np.ndarray]:
    lengths = [np.ndarray(len(all_q.T)) for _ in range(model.nbMuscles())]

    for i, q in enumerate(all_q.T):
        model.updateMuscles(q)
        for m in range(model.nbMuscles()):
            mus = model.muscle(m)
            lengths[m][i] = mus.length(model, q)

    return lengths


def muscle_velocity_from_finitediff(lengths: np.ndarray, t: np.ndarray) -> np.ndarray:
    finitediff = np.zeros(len(t))
    finitediff[1:-1] = (lengths[2:] - lengths[:-2]) / (t[2] - t[0])
    return finitediff


def compute_muscle_velocities(model: MuscleBiorbdModel, all_q: np.ndarray, all_qdot: np.ndarray) -> np.ndarray:
    velocities = [np.ndarray(len(all_q.T)) for _ in range(model.nbMuscles())]

    for i, (q, qdot) in enumerate(zip(all_q.T, all_qdot.T)):
        jac = model.musclesLengthJacobian(q).to_array()
        vel_all_muscles = jac @ qdot
        for m, vel_muscle in enumerate(vel_all_muscles):
            velocities[m][i] = vel_muscle

    return velocities


def dynamics(_, x, model: MuscleBiorbdModel, activations: np.ndarray) -> np.ndarray:
    q = x[: model.nb_q]
    qdot = x[model.nb_q :]
    tau = model.muscle_joint_torque(activations, q, qdot)

    qddot = model.evaluate_mx(model.forward_dynamics(q, qdot, tau), q=q, qdot=qdot).__array__()[:, 0]

    return np.concatenate((qdot, qddot))


def main():
    model = MuscleBiorbdModel(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1)],
    )

    t_span = (0, 2)
    t = np.linspace(*t_span, 100)
    q = np.ones(model.nb_q) * -0.25
    qdot = np.zeros(model.nb_qdot)
    activations = np.ones(model.nb_muscles) * 0.2

    # Request the integration of the equations of motion
    integrated = solve_ivp(
        partial(dynamics, model=model, activations=activations),
        t_span,
        np.concatenate((q, qdot)),
        t_eval=t,
    ).y
    q_int = integrated[: model.nb_q, :]
    qdot_int = integrated[model.nb_q :, :]

    # Compute muscle velocities from finite difference as benchmark
    muscle_lengths = compute_muscle_lengths(model, q_int)
    muscle_velocities_finitediff = [muscle_velocity_from_finitediff(length, t) for length in muscle_lengths]

    # Compute muscle velocities from jacobian
    muscle_velocities_jacobian = compute_muscle_velocities(model, q_int, qdot_int)

    # If the two methods are equivalent, the plot should be on top of each other
    plt.figure("Muscle velocities")
    for finitediff, jacobian in zip(muscle_velocities_finitediff, muscle_velocities_jacobian):
        plt.plot(t, finitediff)
        plt.plot(t, jacobian)
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle velocity (m/s)")
    plt.legend([f"Finite diff", f"Jacobian"])
    plt.show()


if __name__ == "__main__":
    main()

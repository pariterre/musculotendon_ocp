from functools import partial
from typing import Callable

from casadi import DM
from matplotlib import pyplot as plt
from musculotendon_ocp import (
    MuscleHillModels,
    RigidbodyModels,
    ComputeForceDampingMethods,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
)
from musculotendon_ocp.math import precise_rk4
import numpy as np

# TODO : Prendre le graphique v en fonction du temps et faire un delta v en fonction du temps.
#        De ce graphique, on peut générer le graphqiue de delta v en fonction du temps, et en faire un par delta t
# TODO : Activations [0; 1] by increments of 0.25


def muscle_fiber_length_dynamics(_, x, fn_to_dm: Callable, activations: np.ndarray, q: np.ndarray):
    muscle_fiber_lengths = x
    fiber_lengths_dot = np.array(
        fn_to_dm(
            q=q,
            qdot=np.zeros_like(q),
            activations=activations,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocity_initial_guesses=initial_velocity_guesses[-1],
        )["output"]
    ).squeeze()
    return fiber_lengths_dot


def update_initial_velocity_guesses(_, x, fn_to_dm: Callable, activations: np.ndarray, q: np.ndarray):
    muscle_fiber_lengths = x
    initial_velocity_guesses.append(
        np.array(
            fn_to_dm(
                q=q,
                qdot=np.zeros_like(q),
                activations=activations,
                muscle_fiber_lengths=muscle_fiber_lengths,
                muscle_fiber_velocity_initial_guesses=initial_velocity_guesses[-1],
            )["output"]
        ).squeeze()
    )


initial_velocity_guesses = []


def main() -> None:
    colors = ["b", "g", "y", "m", "c"]
    model = RigidbodyModels.WithMuscles(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            # MuscleHillModels.RigidTendon(
            #     name="Mus1",
            #     maximal_force=1000,
            #     optimal_length=0.1,
            #     tendon_slack_length=0.16,
            #     compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
            #     maximal_velocity=5.0,
            # ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Force defects",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects(),
            ),
            # MuscleHillModels.FlexibleTendon(
            #     name="Mus1",
            #     maximal_force=1000,
            #     optimal_length=0.1,
            #     tendon_slack_length=0.16,
            #     compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
            #     maximal_velocity=5.0,
            #     compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
            #     compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects(),
            # ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Linearized",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
            ),
            # MuscleHillModels.FlexibleTendon(
            #     name="Mus1",
            #     maximal_force=1000,
            #     optimal_length=0.1,
            #     tendon_slack_length=0.16,
            #     compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
            #     maximal_velocity=5.0,
            #     compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
            #     compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
            # ),
        ],
    )
    muscle_count = len(model.muscles)

    activations = np.array([0.0] * muscle_count)
    q = np.array([0.290])
    qdot = np.array([0.0])
    initial_muscles_fiber_length = np.array(
        model.function_to_dm(model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot)
    )[:, 0]
    initial_muscles_fiber_velocity = np.array([0.0] * muscle_count)

    # TODO 0->100 and 100 -> 0
    # TODO Change dt
    activations = np.array([1.0] * muscle_count)
    muscles_fiber_length_dynamics_fn = partial(
        muscle_fiber_length_dynamics,
        fn_to_dm=model.to_casadi_function(
            model.muscle_fiber_velocities,
            "activations",
            "q",
            "qdot",
            "muscle_fiber_lengths",
            "muscle_fiber_velocity_initial_guesses",
        ),
        activations=activations,
        q=q,
    )
    inter_integration_step_fn = partial(
        update_initial_velocity_guesses,
        fn_to_dm=model.to_casadi_function(
            model.muscle_fiber_velocities,
            "activations",
            "q",
            "qdot",
            "muscle_fiber_lengths",
            "muscle_fiber_velocity_initial_guesses",
        ),
        activations=activations,
        q=q,
    )
    muscles_force_fn = model.to_casadi_function(
        model.muscle_forces,
        "activations",
        "q",
        "qdot",
        "muscle_fiber_lengths",
        "muscle_fiber_velocities",
    )

    # Reset the previous muscle fiber velocities container
    initial_velocity_guesses.clear()
    initial_velocity_guesses.append(initial_muscles_fiber_velocity)

    # Integrate the muscle fiber length
    t, muscles_fiber_length = precise_rk4(
        muscles_fiber_length_dynamics_fn,
        y0=initial_muscles_fiber_length,
        t_span=[0, 0.05],
        dt=0.001,
        inter_step_callback=inter_integration_step_fn,
    )

    # Dispatch the results and compute the resulting muscle forces
    muscles_fiber_length: np.ndarray = muscles_fiber_length.T
    muscles_fiber_velocity: np.ndarray = np.array(initial_velocity_guesses)
    muscles_force = np.array(
        [
            muscles_force_fn(
                activations=activations,
                q=q,
                qdot=qdot,
                muscle_fiber_lengths=muscles_fiber_length[i, :],
                muscle_fiber_velocities=muscles_fiber_velocity[i, :],
            )["output"]
            for i in range(muscles_fiber_length.shape[0])
        ]
    ).squeeze()

    # Plot muscle velocities
    plt.figure()
    for m in range(len(model.muscles)):
        plt.plot(t, muscles_fiber_velocity[:, m], label=model.muscles[m].label, color=colors[m], marker="o")
    plt.title("Muscle fiber velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle fiber velocity (m/s)")
    plt.grid(visible=True)
    plt.legend()

    # Plot muscle forces
    # TODO Generalize this
    plt.figure()
    diff_force = muscles_force[:, 1] - muscles_force[:, 0]
    impulse = np.zeros_like(diff_force)
    for i in range(1, len(diff_force)):
        impulse[i] = (diff_force[i] - diff_force[i - 1]) * (t[i] - t[i - 1])
    plt.plot(t, impulse, label=model.muscles[m].label, color=colors[m], marker="o")
    plt.title("Impulse")
    plt.xlabel("Time (s)")
    plt.ylabel("Impulse (N * s)")
    plt.grid(visible=True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

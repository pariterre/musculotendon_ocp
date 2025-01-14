from functools import partial
from typing import Callable

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

# NOTE Changing the dt to 0.01 will make the Force and Velocity defects to fail. At 0.005, it kind of work for them, but
# the results are wrong, the linearized fails. At 0.001, the results are correct for all of them.


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
    reference_name = "Force defects"
    activations_patterns = ((0.01, 1.0), (1.0, 0.01))
    model = RigidbodyModels.WithMuscles(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1",
                label="Rigid",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.1,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Force defects",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.1,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects(),
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Velocity defects",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.1,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects(),
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Linearized",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.1,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                label="Quadratic",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.1,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
            ),
        ],
    )
    muscle_count = len(model.muscles)
    reference_index = [m.label for m in model.muscles].index(reference_name)

    for activation_at_start, activation_at_end in activations_patterns:
        activations_at_start = np.array([activation_at_start] * muscle_count)
        q = np.array([0.210])
        qdot = np.array([0.0])
        initial_muscles_fiber_length = np.array(
            model.function_to_dm(
                model.muscle_fiber_lengths_equilibrated, activations=activations_at_start, q=q, qdot=qdot
            )
        )[:, 0]
        initial_muscles_fiber_velocity = np.array([0.0] * muscle_count)

        activations_at_end = np.array([activation_at_end] * muscle_count)
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
            activations=activations_at_end,
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
            activations=activations_at_end,
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
                    activations=activations_at_end,
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
        plt.title(f"Muscle fiber velocity (act {activation_at_start} -> {activation_at_end})")
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle fiber velocity (m/s)")
        plt.grid(visible=True)
        plt.legend()

        # Plot muscle forces
        plt.figure()
        for m in range(len(model.muscles)):
            plt.plot(t, muscles_force[:, m], label=model.muscles[m].label, color=colors[m], marker="o")
        plt.title(f"Impulse (act {activation_at_start} -> {activation_at_end})")
        plt.xlabel("Time (s)")
        plt.ylabel("Impulse (N * s)")
        plt.grid(visible=True)
        plt.legend()

        # Plot muscle forces
        plt.figure()
        for m in range(len(model.muscles)):
            diff_force = muscles_force[:, m] - muscles_force[:, reference_index]
            impulse = np.zeros_like(diff_force)
            # TODO Cumsum instead of by time
            impulse[1:-1] = (diff_force[2:] + diff_force[:-2]) / 2 * (t[2:] - t[:-2])
            plt.plot(t, impulse, label=model.muscles[m].label, color=colors[m], marker="o")
        plt.title(f"Difference of impulse (act {activation_at_start} -> {activation_at_end})")
        plt.xlabel("Time (s)")
        plt.ylabel("Difference of impulse (N * s)")
        plt.grid(visible=True)
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

from functools import partial
from typing import Callable

from matplotlib import pyplot as plt
from musculotendon_ocp import (
    MuscleHillModels,
    RigidbodyModels,
    RigidbodyModelWithMuscles,
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


def _compute_equilibrated_lengths(
    model: RigidbodyModelWithMuscles, activations: np.ndarray, q: np.ndarray, qdot: np.ndarray
) -> np.ndarray:
    return np.array(
        model.function_to_dm(model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot)
    )[:, 0]


def _compute_muscle_forces(
    model: RigidbodyModelWithMuscles, activations: np.ndarray, q: np.ndarray, qdot: np.ndarray
) -> np.ndarray:

    equilibrated_lengths = _compute_equilibrated_lengths(model, activations, q, qdot)
    muscle_velocities = np.array([0.0] * model.nb_muscles)
    muscle_forces = np.array(
        model.function_to_dm(
            model.muscle_forces,
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=equilibrated_lengths,
            muscle_fiber_velocities=muscle_velocities,
        )
    )[:, 0]

    return muscle_forces


def optimize_for_tendon_to_optimal_length_ratio(
    model: RigidbodyModelWithMuscles,
    target_ratio: float,
    target_force: float,
    q_initial_guess: np.ndarray,
    reference_muscle_index: int,
) -> tuple[RigidbodyModelWithMuscles, np.ndarray]:
    # TODO Test this

    if len(model.name_dof) != 1 or model.name_dof[0] != "Cube_TransZ":
        raise ValueError(
            "This function is only implemented for the model one_muscle_holding_a_cube.bioMod with one degree of freedom Cube_TransZ"
        )
    if model.model.nbMuscles() != 1:
        raise ValueError("This function is only implemented for models with one muscle in the bioMod file")

    optimized_model = model.copy

    qdot = np.array([0.0] * model.nb_qdot)
    activations = np.array([1.0] * model.nb_muscles)

    delta = []
    for optimized_muscle, muscle in zip(optimized_model.muscles, model.muscles):
        optimized_muscle.tendon_slack_length = (
            muscle.tendon_slack_length / muscle.optimal_length * target_ratio * muscle.tendon_slack_length
        )
        delta.append(optimized_muscle.tendon_slack_length - muscle.tendon_slack_length)

    # All the delta are supposed to be the same, make sure of this, then use the referenced one
    if not all([d == delta[reference_muscle_index] for d in delta]):
        raise ValueError(
            "All the tendon_slack_length / optimal_length ratios are supposed to be the same. This means "
            "your model is not correctly defined."
        )
    delta = delta[reference_muscle_index]

    # Optimize the value of q so the resulting muscle forces are the target_forces
    from scipy.optimize import minimize

    def objective(q: np.ndarray) -> float:
        muscle_forces = _compute_muscle_forces(optimized_model, activations, q, qdot)
        value = muscle_forces[reference_muscle_index] - target_force
        return value**2

    result = minimize(objective, q_initial_guess + delta, method="Nelder-Mead")
    optimized_q = result.x

    # Validate that the optimization actually converged
    muscle_forces = _compute_muscle_forces(optimized_model, activations, optimized_q, qdot)
    if np.abs(muscle_forces[reference_muscle_index] - target_force) > 0.1:  # Tolerance of 1N
        raise ValueError(
            f"The optimization did not converge. The muscle force is {muscle_forces[reference_muscle_index]:.1f} "
            f"instead of the target force {target_force}."
        )

    return optimized_model, optimized_q


def main() -> None:
    target_force = 500.0
    ratios = [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0]  # This necessitates dt=0.0001
    # ratios = [0.5, 1.0, 5.0, 10.0]
    dt = 0.0001
    colors = ["b", "g", "y", "m", "c"]
    activation_at_start = 0.01
    activation_at_end = 1.0
    reference_muscle_label = "Force defects"
    model = RigidbodyModels.WithMuscles(
        "../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            # MuscleHillModels.RigidTendon(
            #     name="Mus1",
            #     label="Rigid",
            #     maximal_force=1000,
            #     optimal_length=0.1,
            #     tendon_slack_length=0.1,
            #     compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
            #     maximal_velocity=5.0,
            # ),
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
    reference_index = [m.label for m in model.muscles].index(reference_muscle_label)

    for ratio in ratios:
        resized_model, optimized_q = optimize_for_tendon_to_optimal_length_ratio(
            model,
            target_ratio=ratio,
            target_force=target_force,
            q_initial_guess=np.array([0.185]),
            reference_muscle_index=reference_index,
        )

        activations_at_start = np.array([activation_at_start] * muscle_count)
        qdot = np.array([0.0])
        initial_muscles_fiber_length = np.array(
            resized_model.function_to_dm(
                resized_model.muscle_fiber_lengths_equilibrated,
                activations=activations_at_start,
                q=optimized_q,
                qdot=qdot,
            )
        )[:, 0]
        initial_muscles_fiber_velocity = np.array([0.0] * muscle_count)

        activations_at_end = np.array([activation_at_end] * muscle_count)
        muscles_fiber_length_dynamics_fn = partial(
            muscle_fiber_length_dynamics,
            fn_to_dm=resized_model.to_casadi_function(
                resized_model.muscle_fiber_velocities,
                "activations",
                "q",
                "qdot",
                "muscle_fiber_lengths",
                "muscle_fiber_velocity_initial_guesses",
            ),
            activations=activations_at_end,
            q=optimized_q,
        )
        inter_integration_step_fn = partial(
            update_initial_velocity_guesses,
            fn_to_dm=resized_model.to_casadi_function(
                resized_model.muscle_fiber_velocities,
                "activations",
                "q",
                "qdot",
                "muscle_fiber_lengths",
                "muscle_fiber_velocity_initial_guesses",
            ),
            activations=activations_at_end,
            q=optimized_q,
        )
        muscles_force_fn = resized_model.to_casadi_function(
            resized_model.muscle_forces,
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
            dt=dt,
            inter_step_callback=inter_integration_step_fn,
        )

        # Dispatch the results and compute the resulting muscle forces
        muscles_fiber_length: np.ndarray = muscles_fiber_length.T
        muscles_fiber_velocity: np.ndarray = np.array(initial_velocity_guesses)
        muscles_force = np.array(
            [
                muscles_force_fn(
                    activations=activations_at_end,
                    q=optimized_q,
                    qdot=qdot,
                    muscle_fiber_lengths=muscles_fiber_length[i, :],
                    muscle_fiber_velocities=muscles_fiber_velocity[i, :],
                )["output"]
                for i in range(muscles_fiber_length.shape[0])
            ]
        ).squeeze()

        plt.figure(f"Muscle fiber velocity and force for a ratio of {ratio}")

        # Plot muscle velocities
        plt.subplot(3, 1, 1)
        for m in range(len(resized_model.muscles)):
            plt.plot(t, muscles_fiber_velocity[:, m], label=resized_model.muscles[m].label, color=colors[m], marker="o")
        plt.title(f"Muscle fiber velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle fiber velocity (m/s)")
        plt.grid(visible=True)
        plt.legend()

        # Plot muscle forces
        plt.subplot(3, 1, 2)
        for m in range(len(resized_model.muscles)):
            plt.plot(t, muscles_force[:, m], label=resized_model.muscles[m].label, color=colors[m], marker="o")
        plt.title(f"Muscle force")
        plt.xlabel("Time (s)")
        plt.ylabel("Muscle force (N)")
        plt.grid(visible=True)
        plt.legend()

        # Plot the integrated impulse difference
        plt.subplot(3, 1, 3)
        for m in range(len(resized_model.muscles)):
            cum_diff_force = np.cumsum(muscles_force[:, m] - muscles_force[:, reference_index])
            impulse = np.zeros_like(muscles_force[:, m])
            impulse[1:] = (cum_diff_force[1:] + cum_diff_force[:-1]) * (t[1:] - t[:-1]) / 2
            plt.plot(t, impulse, label=model.muscles[m].label, color=colors[m], marker="o")
        plt.title(f"Integrated impulse difference")
        plt.xlabel("Time (s)")
        plt.ylabel("Integrated impulse\ndifference (N*s)")
        plt.grid(visible=True)
        plt.legend()

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

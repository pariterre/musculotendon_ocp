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
    reference_muscle_label: str,
) -> RigidbodyModelWithMuscles:
    # TODO Test this

    if len(model.name_dof) != 1 or model.name_dof[0] != "Cube_TransZ":
        raise ValueError(
            "This function is only implemented for the model one_muscle_holding_a_cube.bioMod with one degree of freedom Cube_TransZ"
        )
    if model.model.nbMuscles() != 1:
        raise ValueError("This function is only implemented for models with one muscle in the bioMod file")

    optimized_model = model.copy
    reference_index = [m.label for m in model.muscles].index(reference_muscle_label)

    qdot = np.array([0.0] * model.nb_qdot)
    activations = np.array([1.0] * model.nb_muscles)

    delta = []
    for optimized_muscle, muscle in zip(optimized_model.muscles, model.muscles):
        optimized_muscle.tendon_slack_length = (
            muscle.tendon_slack_length / muscle.optimal_length * target_ratio * muscle.tendon_slack_length
        )
        delta.append(optimized_muscle.tendon_slack_length - muscle.tendon_slack_length)

    # All the delta are supposed to be the same, make sure of this, then use the referenced one
    if not all([d == delta[reference_index] for d in delta]):
        raise ValueError(
            "All the tendon_slack_length / optimal_length ratios are supposed to be the same. This means "
            "your model is not correctly defined."
        )
    delta = delta[reference_index]

    # Optimize the value of q so the resulting muscle forces are the target_forces
    from scipy.optimize import minimize

    def objective(q: np.ndarray) -> float:
        muscle_forces = _compute_muscle_forces(optimized_model, activations, q, qdot)
        value = muscle_forces[reference_index] - target_force
        return value**2

    result = minimize(objective, q_initial_guess + delta, method="Nelder-Mead")
    q = result.x

    # Validate that the optimization actually converged
    muscle_forces = _compute_muscle_forces(optimized_model, activations, q, qdot)
    if np.abs(muscle_forces[reference_index] - target_force) > 0.1:  # Tolerance of 1N
        raise ValueError(
            f"The optimization did not converge. The muscle force is {muscle_forces[reference_index]:.1f} "
            f"instead of the target force {target_force}."
        )

    return optimized_model


def main() -> None:
    target_force = 500.0
    ratios = [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0]
    colors = ["b", "g", "y", "m", "c"]
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

    for ratio in ratios:
        resized_model = optimize_for_tendon_to_optimal_length_ratio(
            model,
            target_ratio=ratio,
            target_force=target_force,
            q_initial_guess=np.array([0.185]),
            reference_muscle_label="Force defects",
        )

        resized_model


if __name__ == "__main__":
    main()

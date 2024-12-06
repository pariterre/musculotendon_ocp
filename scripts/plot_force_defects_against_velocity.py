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
from musculotendon_ocp.math import precise_rk1, precise_rk4, precise_rk45
import numpy as np


def muscle_fiber_length_dynamics(_, x, fn_to_dm: Callable, activations: np.ndarray, q: np.ndarray):
    muscle_fiber_lengths = x
    fiber_lengths_dot = np.array(
        fn_to_dm(
            q=q,
            qdot=np.zeros_like(q),
            activations=activations,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocity_initial_guesses=previous_lmdot[-1],  # TODO : Only use zeros as initial guess?
        )["output"]
    ).squeeze()
    previous_lmdot.append(fiber_lengths_dot)
    return fiber_lengths_dot


previous_lmdot = []


def main() -> None:
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
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects(),
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
                maximal_velocity=5.0,
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
            ),
        ],
    )
    muscle_count = len(model.muscles)

    activations = np.array([0.1] * muscle_count)
    q = np.array([0.25])
    qdot = np.array([0.0])
    initial_muscles_fiber_length = np.array([0.1] * muscle_count)
    initial_muscles_fiber_velocity = np.array([-1.7] * muscle_count)

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
    muscles_force_fn = model.to_casadi_function(
        model.muscle_forces,
        "activations",
        "q",
        "qdot",
        "muscle_fiber_lengths",
        "muscle_fiber_velocities",
    )

    # Compute the muscle fiber length using different integration methods
    results = {}
    for key, method, color in zip(["RK1", "RK4", "RK45"], [precise_rk1, precise_rk4, precise_rk45], ["r", "g", "b"]):
        # Reset the previous muscle fiber velocities container
        previous_lmdot.clear()
        previous_lmdot.append(initial_muscles_fiber_velocity)

        # Integrate the muscle fiber length
        t, muscles_fiber_length = method(
            muscles_fiber_length_dynamics_fn, y0=initial_muscles_fiber_length, t_span=[0, 1], dt=0.001
        )

        # Dispatch the results and compute the resulting muscle forces
        muscles_fiber_length: np.ndarray = muscles_fiber_length.T
        muscles_fiber_velocity = np.array(previous_lmdot)
        velocity_length_ratio = (muscles_fiber_velocity.shape[0] - 1) / (muscles_fiber_length.shape[0] - 1)
        if velocity_length_ratio != int(velocity_length_ratio):
            # TODO Once we decided what to do with the muscle fiber velocity during the integration, we should remove this warning
            Warning("The muscle fiber velocity and length do not have a compatible shape")
            muscles_fiber_velocity = np.zeros_like(muscles_fiber_length) * np.nan
        else:
            muscles_fiber_velocity = muscles_fiber_velocity[:: int(velocity_length_ratio)]
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

        results[key] = {
            "t": t,
            "muscles_fiber_length": muscles_fiber_length,
            "muscles_fiber_velocity": muscles_fiber_velocity,
            "muscles_force": muscles_force,
            "color": color,
        }

    plt.figure()
    for key, result in results.items():
        plt.plot(result["t"], result["muscles_fiber_length"], label=key, color=result["color"])
    plt.title("Muscles fiber length")
    plt.xlabel("Time (s)")
    plt.ylabel("Muscles fiber length")
    plt.grid(visible=True)
    plt.legend()

    plt.figure()
    for key, result in results.items():
        plt.plot(result["t"], result["muscles_fiber_velocity"], label=key, color=result["color"])
    plt.title("Muscle fiber velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle fiber velocity")
    plt.grid(visible=True)
    plt.legend()

    plt.figure()
    for key, result in results.items():
        plt.plot(result["t"], result["muscles_force"], label=key, color=result["color"])
    plt.title("Muscle force")
    plt.xlabel("Time (s)")
    plt.ylabel("Muscle force")
    plt.grid(visible=True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

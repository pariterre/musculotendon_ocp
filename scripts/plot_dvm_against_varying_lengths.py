import json
from functools import partial
import os
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
    ratios = [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0]
    dts = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    # ratios = [5.0, 10.0]
    # dts = [0.001, 0.005]

    load_results = False
    save_path = "results/results.json"
    plot_graphs = True
    target_force = 500.0
    velocity_threshold = 1e-2
    colors = ["b", "g", "y", "m", "c"]
    activation_at_start = 0.01
    activation_at_end = 1.0
    reference_muscle_label = "Force defects"
    model = RigidbodyModels.WithMuscles(
        "../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
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
    reference_index = [m.label for m in model.muscles].index(reference_muscle_label)
    meta_data = {
        "ratios": ratios,
        "dts": dts,
        "target_force": target_force,
        "velocity_threshold": velocity_threshold,
        "activation_at_start": activation_at_start,
        "activation_at_end": activation_at_end,
        "reference_muscle_label": reference_muscle_label,
        "model_name": model.name,
        "muscle_models": [muscle.serialize() for muscle in model.muscles],
    }

    if load_results and os.path.exists(save_path):
        data = json.load(open(save_path, "r"))
        if meta_data != data["meta_data"]:
            raise ValueError(
                "The meta data does not match the saved data, change your parameters or set load_results to False"
            )
        results = data["results"]
    else:
        results = {}
        for ratio in ratios:
            results[str(ratio)] = {}
            for dt in dts:
                print(f"Processing ratio {ratio} for dt={dt}")
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
                try:
                    t, muscles_fiber_length = precise_rk4(
                        muscles_fiber_length_dynamics_fn,
                        y0=initial_muscles_fiber_length,
                        t_span=[0, 0.05],
                        dt=dt,
                        inter_step_callback=inter_integration_step_fn,
                    )
                except:
                    print(f"Failed for ratio {ratio} and dt {dt}")
                    continue

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

                equilibrated_t_indices = []
                # Find where the t index where the velocity is negligeable (velocity_threshold)
                for m in range(len(resized_model.muscles)):
                    normalized_velocity = model.muscles[reference_index].normalize_muscle_fiber_velocity(
                        muscles_fiber_velocity[:, m]
                    )
                    index = np.where(np.abs(normalized_velocity) < velocity_threshold)[0]
                    # By design the index 0 is 0, so skip it
                    equilibrated_t_indices.append(int(index[1]) if len(index) > 1 else None)

                results[str(ratio)][str(dt)] = {
                    "t": t.tolist(),
                    "muscles_fiber_velocity": muscles_fiber_velocity.tolist(),
                    "muscles_force": muscles_force.tolist(),
                    "equilibrated_t_indices": equilibrated_t_indices,
                }

        # Save as json file
        if not os.path.exists("results"):
            os.makedirs("results")
        json.dump(
            {
                "meta_data": meta_data,
                "results": results,
            },
            open(save_path, "w"),
        )

    # Build the latex table from the results
    header_title = r"\multicolumn{" + str(len(dts)) + r"}{c}{Time to equilibrium (s)}"
    header_dt = " & ".join([f"dt = {dt:.4f}" for dt in dts])

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{>{\raggedright}p{3.5cm}" + "c" * len(dts) + r"}")
    print(r"\multirow{2}{3.5cm}{Ratio slack length to optimal length} & " + header_title + r"\\")
    print(r" & " + header_dt + r"\\")
    print(r"\hline")

    for ratio in ratios:
        row = f"{ratio:.1f}"
        for dt in dts:
            data = results[str(ratio)][str(dt)]
            equilibrated_t_index = data["equilibrated_t_indices"][reference_index]
            row += " & -" if equilibrated_t_index is None else f" & {data['t'][equilibrated_t_index]:.4f}"
        row += r"\\"
        print(row)

    print(r"\end{tabular}")
    print(
        r"\caption{Time to reach an equilibrated muscle fiber velocity for different tendon to optimal length ratios and time steps}"
    )
    print(r"\label{tab:equilibrated_muscle_fiber_velocity}")
    print(r"\end{table}")

    if plot_graphs:
        for ratio in ratios:
            for dt in dts:
                data = results[str(ratio)][str(dt)]

                t = np.array(data["t"])
                muscles_fiber_velocity = np.array(data["muscles_fiber_velocity"])
                muscles_force = np.array(data["muscles_force"])
                equilibrated_t_indices = np.array(data["equilibrated_t_indices"])

                plt.figure(f"Muscle fiber velocity and force for a ratio of {ratio} at dt = {dt}")

                # Plot muscle velocities
                plt.subplot(3, 1, 1)
                for m in range(len(model.muscles)):
                    plt.plot(
                        t,
                        muscles_fiber_velocity[:, m],
                        label=model.muscles[m].label,
                        color=colors[m],
                        marker="o",
                    )
                    if equilibrated_t_indices[m] is not None:
                        plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
                plt.title(f"Muscle fiber velocity")
                plt.xlabel("Time (s)")
                plt.ylabel("Muscle fiber velocity (m/s)")
                plt.grid(visible=True)
                plt.legend()

                # Plot muscle forces
                plt.subplot(3, 1, 2)
                for m in range(len(model.muscles)):
                    plt.plot(t, muscles_force[:, m], label=model.muscles[m].label, color=colors[m], marker="o")
                    if equilibrated_t_indices[m] is not None:
                        plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
                plt.title(f"Muscle force")
                plt.xlabel("Time (s)")
                plt.ylabel("Muscle force (N)")
                plt.grid(visible=True)
                plt.legend()

                # Plot the integrated impulse difference
                plt.subplot(3, 1, 3)
                for m in range(len(model.muscles)):
                    cum_diff_force = np.cumsum(muscles_force[:, m] - muscles_force[:, reference_index])
                    impulse = np.zeros_like(muscles_force[:, m])
                    impulse[1:] = (cum_diff_force[1:] + cum_diff_force[:-1]) * (t[1:] - t[:-1]) / 2
                    plt.plot(t, impulse, label=model.muscles[m].label, color=colors[m], marker="o")
                    if equilibrated_t_indices[m] is not None:
                        plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
                plt.title(f"Integrated impulse difference")
                plt.xlabel("Time (s)")
                plt.ylabel("Integrated impulse\ndifference (N*s)")
                plt.grid(visible=True)
                plt.legend()

                plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

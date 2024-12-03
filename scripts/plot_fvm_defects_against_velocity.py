from matplotlib import pyplot as plt
from musculotendon_ocp import MuscleHillModels
from musculotendon_ocp.math import compute_finitediff, precise_rk45, precise_rk4
import numpy as np


def main() -> None:
    muscle = MuscleHillModels.FlexibleTendon(
        name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
    )

    initial_vm = muscle.normalize_muscle_fiber_velocity(np.linspace(-5, 5, 100))
    colors = ["r", "g", "b", "k", "m"]
    for i, delta_vm in enumerate([0.001, 0.01, 0.1]):
        # True
        true_force_velocity = muscle.compute_force_velocity(initial_vm + delta_vm)

        # Linearized
        first_derivative = muscle.compute_force_velocity.first_derivative(initial_vm)
        linearized_force_velocity = muscle.compute_force_velocity(initial_vm) + first_derivative * delta_vm

        # Quadratic
        second_derivative = muscle.compute_force_velocity.second_derivative(initial_vm)
        quadratic_force_velocity = linearized_force_velocity + second_derivative / 2 * delta_vm**2

        # Plot
        plt.plot(
            initial_vm + delta_vm,
            np.abs(linearized_force_velocity - true_force_velocity) * 100,
            color=colors[i],
            linestyle="-",
            label=r"Linearized (\Delta_{vm} = " + f"{delta_vm})",
        )
        plt.plot(
            initial_vm + delta_vm,
            np.abs(quadratic_force_velocity - true_force_velocity) * 100,
            color=colors[i],
            linestyle="--",
            label=r"Quadratic (\Delta_{vm} = " + f"{delta_vm})",
        )

    plt.title("Error of the force-velocity relationship")
    plt.xlabel("Normalized muscle fiber velocity")
    plt.ylabel("Error of fv (%)")
    plt.grid(visible=True)
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

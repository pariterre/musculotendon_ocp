from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    ConstraintList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    InitialGuessList,
    ControlType,
    Node,
    PenaltyController,
    MultinodeConstraintList,
)
from casadi import MX
import numpy as np
from musculotendon_ocp import (
    RigidbodyModels,
    RigidbodyModelWithMuscles,
    MuscleHillModels,
    ComputeForceDampingMethods,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
    PlotHelpers,
    DynamicsHelpers,
)


# TODO add pennation angle
# TODO OCP tracking a trajectory
# TODO Test a standardized panel of muscles
# TODO COLLOCATION


# NOTES:
#
# Constraint of no muscle velocity on the first node (instead of bound) so it adjust to the activations:
#     This turned out to be a bad idea (at least for DMS) since solving the equilibrium instantaneously implies using
#     the rootfinder method, making it impossible to use SX variables (which is too high of a price, compared to the
#     benefits of this constraint)
#
# OCP for rigid tendon:
#     Keeping the extra useless variables of the muscle fiber length does not seem to negatively impact the convergence
#     time, nor the final solution, even though it doubles the amount of constraintes.
#     It is therefore a good idea to keep them since it allows to easily switch between rigid and flexible tendons.


def fiber_lmdot_equals_velocities(controllers: list[PenaltyController]) -> MX:
    return controllers[0].controls["muscles_fiber_velocities"].cx - (
        (controllers[1].states["muscles_fiber_lengths"].cx - controllers[0].states["muscles_fiber_lengths"].cx)
        / controllers[0].dt.cx
    )


def fiber_lmdot_equals_velocities_end(controller: PenaltyController) -> MX:
    return controller.controls["muscles_fiber_velocities"].cx


def prepare_ocp(
    model: RigidbodyModelWithMuscles,
    final_time: float,
    q0: np.ndarray,
    qf: np.ndarray,
    ode_solver: OdeSolverBase,
    use_sx: bool,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    model: RigidbodyModelWithMuscles
        The model to be used
    final_time: float
        The time at the final node, this also determines the number of shooting nodes. Based on prior tests, we know
        that RK4 steps should not be larger that 0.001, since RK4 performs 5 substeps
    q0: np.ndarray
        The initial position
    qf: np.ndarray
        The final position
    ode_solver: OdeSolverBase
        The ode solver to be used
    control_type: ControlType
        The type of control to be used
    use_sx: bool
        If the problem should be solved with SX or MX

    Returns
    -------
    The ocp
    """

    # Declare some useful variables
    minimum_integration_time = 0.001 * ode_solver.n_integration_steps
    shooting_count = int(final_time / minimum_integration_time)
    if shooting_count != final_time / minimum_integration_time:
        raise ValueError("The final time should be a multiple of (0.001 * ode_solver.n_integration_steps)")

    nb_muscles = model.nb_muscles
    # Without damping, there is a singularity if the activation is 0. So it needs to be > (0 + eps) where eps is a
    # neighborhood of 0 and depends on the muscle, it can be very small or not
    activation_min, activation_max = 0.05, 1.0

    activations_min = np.array([activation_min] * nb_muscles)
    activations_max = np.array([activation_max] * nb_muscles)
    activations_init = np.mean([activations_min, activations_max], axis=0)
    equilibrated_muscle_lengths = np.array(
        model.function_to_dm(
            model.muscle_fiber_lengths_equilibrated,
            activations=activations_init,
            q=q0,
            qdot=np.zeros(model.nb_q),
        )
    )[:, 0]

    # Declare bioptim variables
    dynamics = DynamicsList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    multinode_constraints = MultinodeConstraintList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()

    # Declare the dynamics
    dynamics.add(DynamicsHelpers.configure, expand_dynamics=False)

    # Minimize the muscle activation
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=50)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="muscles", node=Node.END, weight=50 * minimum_integration_time
    )

    # Start and end at a specific position, at rest, and make sure the model is constraint the requested bounds
    x_bounds.add("q", model.bounds_from_ranges("q"))
    x_bounds.add("qdot", model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = q0
    x_bounds["q"][:, -1] = qf
    x_bounds["qdot"][0, [0, -1]] = 0
    x_init["q"] = q0

    # Muscle lengths are stricly positive and start with muscle fiber lengths at equilibrium
    x_bounds["muscles_fiber_lengths"] = [0] * nb_muscles, [np.inf] * nb_muscles
    x_bounds["muscles_fiber_lengths"][:, 0] = equilibrated_muscle_lengths
    x_init["muscles_fiber_lengths"] = equilibrated_muscle_lengths

    # Muscle activations are between activation_min and activation_max
    u_bounds["muscles"] = activations_min, activations_max
    u_init["muscles"] = activations_init

    # Make sure the virtual muscle fiber velocities (as controls) are the same as start of next node of the real (as states).
    # That allows us (at convergence) to access the integrated value as initial guess
    for i in range(shooting_count):
        multinode_constraints.add(fiber_lmdot_equals_velocities, nodes_phase=(0, 0), nodes=(i, i + 1))
    constraints.add(fiber_lmdot_equals_velocities_end, node=Node.END)

    ocp = OptimalControlProgram(
        model,
        dynamics,
        shooting_count,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        use_sx=use_sx,
        ode_solver=ode_solver,
        control_type=ControlType.LINEAR_CONTINUOUS,
    )

    # Add the tendon forces to the plot
    PlotHelpers.add_tendon_forces_plot_to_ocp(ocp=ocp, model=model)
    PlotHelpers.add_muscle_forces_plot_to_ocp(ocp=ocp, model=model)

    model_flexible_from_velocity_defects = model.copy_with_all_flexible_tendons()
    PlotHelpers.add_tendon_forces_plot_to_ocp(ocp=ocp, model=model_flexible_from_velocity_defects)
    PlotHelpers.add_muscle_forces_plot_to_ocp(ocp=ocp, model=model_flexible_from_velocity_defects)

    return ocp


def main():
    model = RigidbodyModels.WithMuscles(
        "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod",
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1",
                maximal_force=1000,
                optimal_length=0.1,
                tendon_slack_length=0.16,
                maximal_velocity=5.0,
                compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
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
        ],
    )

    ocp = prepare_ocp(
        model=model,
        final_time=0.5,
        q0=np.array([-0.22]),
        qf=np.array([-0.30]),
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate(show_gravity_vector=False)


if __name__ == "__main__":
    main()

from functools import partial

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    ObjectiveFcn,
    ObjectiveList,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    InitialGuessList,
    ControlType,
    DynamicsFunctions,
    Node,
)
from casadi import SX, MX, vertcat
import numpy as np
from musculotendon_ocp import (
    RigidbodyModels,
    RigidbodyModelWithMuscles,
    MuscleHillModels,
    ComputeForceDampingMethods,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
)


# TODO add a constraint at node zero for the muscle_length (instead of bound) so it adjust to the activations?
# TODO Implement the second order approximation


def prepare_muscle_fiber_velocities(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    muscle_fiber_velocities = model.muscle_fiber_velocities(
        activations=activations, q=q, qdot=qdot, muscle_fiber_lengths=model.muscle_fiber_lengths_mx
    )
    return muscle_fiber_velocities


def prepare_forward_dynamics(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
    tau = model.muscle_joint_torque(activations, q, qdot, muscle_fiber_lengths=model.muscle_fiber_lengths_mx)
    qddot = model.forward_dynamics(q, qdot, tau)
    return qddot


def custom_dynamics(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    algebraic_states: MX | SX
        The algebraic states of the system
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    model: RigidbodyModelWithMuscles = nlp.model

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    muscle_fiber_lengths = DynamicsFunctions.get(nlp.states["muscles"], states)
    muscle_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    muscle_fiber_lengths_dot = model.to_casadi_function(
        partial(prepare_muscle_fiber_velocities, model=model), "activations", "q", "qdot"
    )(
        activations=muscle_activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=muscle_fiber_lengths,
        muscle_fiber_velocities=0,  # TODO Check if it is possible to use a better initial guess
    )[
        "output"
    ]

    qddot = model.to_casadi_function(partial(prepare_forward_dynamics, model=model), "activations", "q", "qdot")(
        activations=muscle_activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=muscle_fiber_lengths,
        muscle_fiber_velocities=muscle_fiber_lengths_dot,
    )["output"]

    return DynamicsEvaluation(dxdt=vertcat(qdot, qddot, muscle_fiber_lengths_dot), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    # States
    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=True, as_controls=False)

    # Control
    ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)

    # Dynamics
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_ocp(
    model: RigidbodyModelWithMuscles,
    final_time: float,
    q0: np.ndarray,
    qf: np.ndarray,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    control_type: ControlType = ControlType.LINEAR_CONTINUOUS,
    use_sx: bool = True,
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
    activation_min, activation_max = 0.2, 1.0

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
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()

    # Declare the dynamics
    dynamics.add(custom_configure, dynamic_function=custom_dynamics, expand_dynamics=False)

    # Minimize the muscle activation
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=50)
    if control_type == ControlType.LINEAR_CONTINUOUS:
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_CONTROL, key="muscles", weight=50, node=Node.END)

    # Start and end at a specific position, at rest, and make sure the model is constraint the requested bounds
    x_bounds.add("q", model.bounds_from_ranges("q"))
    x_bounds.add("qdot", model.bounds_from_ranges("qdot"))
    x_bounds["q"][:, 0] = q0
    x_bounds["q"][:, -1] = qf
    x_bounds["qdot"][0, [0, -1]] = 0
    x_init["q"] = q0

    # Muscle lengths are stricly positive and start with muscle fiber lengths at equilibrium
    x_bounds["muscles"] = [0] * nb_muscles, [np.inf] * nb_muscles
    x_bounds["muscles"][:, 0] = equilibrated_muscle_lengths
    x_init["muscles"] = equilibrated_muscle_lengths

    # Muscle activations are between activation_min and activation_max
    u_bounds["muscles"] = activations_min, activations_max
    u_init["muscles"] = activations_init

    return OptimalControlProgram(
        model,
        dynamics,
        shooting_count,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        use_sx=use_sx,
        ode_solver=ode_solver,
        control_type=control_type,
    )


def main():
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
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
            ),
        ],
    )

    ocp = prepare_ocp(
        model=model,
        final_time=0.5,
        q0=np.array([-0.22]),
        qf=np.array([-0.26]),
        ode_solver=OdeSolver.RK4(),
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate(show_gravity_vector=False)


if __name__ == "__main__":
    main()

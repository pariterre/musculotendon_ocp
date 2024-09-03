from functools import partial

from bioptim import OptimalControlProgram, NonLinearProgram, ConfigureProblem, DynamicsEvaluation, DynamicsFunctions
from casadi import MX, SX, vertcat, Function

from .casadi_helpers import CasadiHelpers
from ..rigidbody_models import RigidbodyModelWithMuscles


class DynamicsHelpers:
    @staticmethod
    def configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
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
        ConfigureProblem.configure_new_variable(
            "muscles_fiber_lengths", nlp.model.muscle_names, ocp, nlp, as_states=True, as_controls=False
        )

        # Control
        ConfigureProblem.configure_muscles(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_new_variable(
            "muscles_fiber_velocities", nlp.model.muscle_names, ocp, nlp, as_states=False, as_controls=True
        )

        # Dynamics
        model: RigidbodyModelWithMuscles = nlp.model
        muscle_fiber_length_dot_func = model.to_casadi_function(
            partial(CasadiHelpers.prepare_fiber_lmdot_mx, model=model), "activations", "q", "qdot"
        )
        qddot_func = model.to_casadi_function(
            partial(CasadiHelpers.prepare_forward_dynamics_mx, model=model), "activations", "q", "qdot"
        )
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            partial(
                DynamicsHelpers.dynamics,
                muscle_fiber_length_dot_func=muscle_fiber_length_dot_func,
                qddot_func=qddot_func,
            ),
        )

    @staticmethod
    def dynamics(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_fiber_length_dot_func: Function,
        qddot_func: Function,
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

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        muscle_fiber_lengths = DynamicsFunctions.get(nlp.states["muscles_fiber_lengths"], states)
        muscle_fiber_velocities = DynamicsFunctions.get(nlp.controls["muscles_fiber_velocities"], controls)
        muscle_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

        muscle_fiber_lengths_dot = muscle_fiber_length_dot_func(
            activations=muscle_activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocity_initial_guesses=muscle_fiber_velocities,
        )["output"]

        qddot = qddot_func(
            activations=muscle_activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocities=muscle_fiber_lengths_dot,
        )["output"]

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot, muscle_fiber_lengths_dot), defects=None)

from typing import Iterable, Self

from bioptim import BiorbdModel, OptimalControlProgram, NonLinearProgram, ConfigureProblem
from casadi import MX, DM, Function

from ..muscle_models.muscle_model_abstract import MuscleModelAbstract


class MuscleBiorbdModel(BiorbdModel):
    def __init__(self, muscles: Iterable[MuscleModelAbstract], *args, **kwargs):
        super().__init__(*args, **kwargs)

        for muscle in muscles:
            if muscle.name not in self.muscle_names():
                raise ValueError(f"Muscle {muscle.name} was not found in the biorbd model")

        self._muscles = muscles
        # TODO Add parameters that normalizes the muscle

    def tendon_force(self, tendon_length: MX) -> MX:
        """
        Compute the tendon force

        Parameters
        ----------
        tendon_length: MX
            The tendon length

        Returns
        -------
        MX
            The tendon force corresponding to the given tendon length
        """
        raise NotImplementedError("TODO")

    def muscle_length_jacobian(self, q: MX) -> MX:
        """
        Compute the muscle length jacobian

        Parameters
        ----------
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        """
        raise NotImplementedError("TODO")

    def muscle_forces(self, activation: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle forces

        Parameters
        ----------
        muscle_activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle forces vector of size (n_muscles x 1)
        """
        raise NotImplementedError("TODO")

    def compute_muscle_joint_torque(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle joint torque

        Parameters
        ----------
        muscle_activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle joint torque vector of size (n_dof x 1)
        """
        self.check_q_size(q)
        self.check_qdot_size(qdot)
        self.check_muscle_size(activations)
        muscle_length_jacobian = self.compute_muscle_length_jacobian(q)
        muscle_forces = self.compute_muscle_forces(q, qdot, activations)

        return muscle_length_jacobian.T @ muscle_forces

    def configure_bioptim_dynamics(
        self, ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
    ):
        """
        Configure the bioptim dynamics

        Returns
        -------
        DynamicsFcn
            The bioptim dynamics function
        """
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)

    def _evaluate_mx(model: Self, mx: MX, **kwargs) -> DM:
        """
        Evaluate a CasADi MX function at a given point.

        Parameters
        ----------
        model: MuscleBiorbdModel
            The model that implements MuscleBiorbdModel.
        mx: MX
            The CasADi MX function to evaluate.
        **kwargs
            The values of the variables at which to evaluate the function, limited to q, qdot, lm, act, vm_c.

        Returns
        -------
        DM
            The result of the evaluation
        """

        q_mx = MX.sym("q", model.nb_q, 1)
        qdot_mx = MX.sym("qdot", model.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", model.nb_muscles, 1)
        act_mx = MX.sym("act", model.nb_muscles, 1)
        vm_c_normalized_mx = MX.sym("vm_c", model.nb_muscles, 1)

        def get_value_from_kwargs(key, default):
            if key in kwargs:
                return kwargs[key]
            return default

        q = get_value_from_kwargs("q", [0] * model.nb_q)
        qdot = get_value_from_kwargs("qdot", [0] * model.nb_qdot)
        lm_normalized = get_value_from_kwargs("lm", [0] * model.nb_muscles)
        act = get_value_from_kwargs("act", [0] * model.nb_muscles)
        vm_c_normalized = get_value_from_kwargs("vm_c", [0] * model.nb_muscles)

        return Function("f", [q_mx, qdot_mx, lm_normalized_mx, act_mx, vm_c_normalized_mx], [mx])(
            q, qdot, lm_normalized, act, vm_c_normalized
        )

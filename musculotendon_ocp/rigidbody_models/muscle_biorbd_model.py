from typing import Iterable, Self, override

import biorbd_casadi as biorbd
from bioptim import BiorbdModel, OptimalControlProgram, NonLinearProgram, ConfigureProblem
from casadi import MX, DM, Function
import numpy as np

from ..muscle_models.muscle_model_abstract import MuscleModelAbstract
from ..muscle_models.hill.muscle_model_hill_rigid_tendon import MuscleModelHillRigidTendon


class MuscleBiorbdModel(BiorbdModel):
    def __init__(self, bio_model: str, muscles: Iterable[MuscleModelAbstract], *args, **kwargs):
        super().__init__(bio_model, *args, **kwargs)

        self._muscle_index_to_biorbd_model = []
        for muscle in muscles:
            if muscle.name not in self.muscle_names:
                raise ValueError(f"Muscle {muscle.name} was not found in the biorbd model")
            self._muscle_index_to_biorbd_model.append(self.muscle_names.index(muscle.name))

        self._muscles = muscles
        # TODO Add parameters that normalizes the muscle

    @property
    def nb_muscles(self) -> int:
        """
        Get the number of muscles

        Returns
        -------
        int
            The number of muscles
        """
        return len(self._muscles)

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

    @override
    def muscle_length_jacobian(self, q) -> MX:
        raise RuntimeError(
            "In the context of this project, the name 'muscle_length_jacobian' is confusing as it is the "
            "jacobian of the muscle-tendon-unit length (as opposed to the muscle-fiber-unit length)."
        )

    @override
    def muscle_tendon_length_jacobian(self, q) -> MX:
        return super(MuscleBiorbdModel, self).muscle_length_jacobian(q)[self._muscle_index_to_biorbd_model, :]

    def _get_muscle_fiber_length(self, updated_model: biorbd.Model, muscle_index: int) -> MX:
        """
        Get the muscle fiber length

        Parameters
        ----------
        updated_model: biorbd.Model
            The updated model using self.model.UpdateKinematicsCustom(q)
        muscle_index: int
            The muscle index to get the muscle fiber length based on self._muscles

        Returns
        -------
        MX
            The muscle fiber length of the requested muscle
        """

        # TODO: Add test for this
        muscle = self._muscles[muscle_index]

        if isinstance(muscle, MuscleModelHillRigidTendon):
            muscle_biorbd_index = self._muscle_index_to_biorbd_model[muscle_index]
            muscle_biorbd = self.model.muscle(muscle_biorbd_index)

            q = np.zeros(self.nb_q)  # Q is actually irrelevant as the kinematics are already updated
            muscle_tendon_length = muscle_biorbd.musculoTendonLength(updated_model, q).to_mx()
            tendon_length = muscle_biorbd.characteristics().tendonSlackLength().to_mx()
            return muscle_tendon_length - tendon_length

        else:
            raise NotImplementedError(
                f"The muscle model {type(muscle)} is not implemented to compute the muscle fiber length"
            )

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

        updated_model = self.model.UpdateKinematicsCustom(q)

        for i, muscle in enumerate(self._muscles):
            muscle_fiber_length = float(
                self.evaluate_mx(self._get_muscle_fiber_length(updated_model, muscle_index=i), q=q)[0, 0]
            )

            # TODO RENDU ICI!!!!
            muscle.compute_muscle_force()

    @override
    def muscle_joint_torque(self, activations: MX, q: MX, qdot: MX) -> MX:
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
        muscle_tendon_length_jacobian = self.muscle_tendon_length_jacobian(q)
        muscle_forces = self.muscle_forces(q, qdot, activations)

        return -muscle_tendon_length_jacobian.T @ muscle_forces

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

    def evaluate_mx(model: Self, mx: MX, **kwargs) -> DM:
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

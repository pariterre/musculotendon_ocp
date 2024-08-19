from typing import Iterable, Callable, override, Any

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

    def _compute_muscle_fiber_length(self, updated_model: biorbd.Model, q: MX, muscle_index: int) -> MX:
        """
        Get the muscle fiber length

        Parameters
        ----------
        updated_model: biorbd.Model
            The updated model using self.model.UpdateKinematicsCustom(q)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
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
            mus_biorbd: biorbd.Muscle = self.model.muscle(self._muscle_index_to_biorbd_model[muscle_index])
            muscle_tendon_length = mus_biorbd.musculoTendonLength(updated_model, q).to_mx()

            return muscle_tendon_length - muscle.tendon_slack_length

        else:
            raise NotImplementedError(
                f"The muscle model {type(muscle)} is not implemented to compute the muscle fiber length"
            )

    def _compute_muscle_fiber_velocity(self, updated_model: biorbd.Model, qdot: MX, muscle_index: int) -> MX:
        """
        Get the muscle fiber velocity

        Parameters
        ----------
        updated_model: biorbd.Model
            The updated model using self.model.UpdateKinematicsCustom(q, qdot)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)
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
            mus_biorbd: biorbd.Muscle = self.model.muscle(self._muscle_index_to_biorbd_model[muscle_index])
            mus_position: biorbd.MuscleGeometry = mus_biorbd.position()

            mus_jacobian = mus_position.jacobianLength().to_mx()
            return mus_jacobian @ qdot

        else:
            raise NotImplementedError(
                f"The muscle model {type(muscle)} is not implemented to compute the muscle fiber length"
            )

    def muscle_lengths(self, q: MX) -> MX:
        """
        Compute the muscle lengths

        Parameters
        ----------
        q: MX
            The generalized coordinates vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle lengths vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q)

        lengths = MX.zeros(self.nb_muscles, 1)
        for i, muscle in enumerate(self._muscles):
            lengths[i] = self._compute_muscle_fiber_length(updated_model, q=q, muscle_index=i)

        return lengths

    def muscle_forces(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle forces

        Parameters
        ----------
        activations: MX
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

        updated_model = self.model.UpdateKinematicsCustom(q, qdot)

        forces = MX.zeros(self.nb_muscles, 1)
        for i, muscle in enumerate(self._muscles):
            muscle_fiber_length = self._compute_muscle_fiber_length(updated_model, q=q, muscle_index=i)
            muscle_fiber_velocity = self._compute_muscle_fiber_velocity(updated_model, qdot=qdot, muscle_index=i)
            forces[i] = muscle.compute_muscle_force(
                activation=activations[i],
                muscle_fiber_length=muscle_fiber_length,
                muscle_fiber_velocity=muscle_fiber_velocity,
            )

        return forces

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
        muscle_forces = self.muscle_forces(activations, q, qdot)

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

    def to_casadi_function(self, mx_function: Callable[[Any], MX], *keys: Iterable[str]) -> Function:
        """
        Convert a CasADi MX to a CasADi Function with specific parameters. To evaluate the function, use evaluate_function.

        Parameters
        ----------
        model: MuscleBiorbdModel
            The model that implements MuscleBiorbdModel.
        mx_function: Callable
            The CasADi MX function to convert to a casadi function.
        keys: Iterable[str]
            The keys of the variables to use in the function, limited to "q", "qdot", "lm", "activations", "vm".
            The order of the keys must match the order of the arguments in the mx_function.

        Returns
        -------
        DM
            The result of the evaluation
        """

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        lm_normalized_mx = MX.sym("lm", self.nb_muscles, 1)
        activations_mx = MX.sym("activations", self.nb_muscles, 1)
        vm_normalized_mx = MX.sym("vm_c", self.nb_muscles, 1)

        keys_to_mx = {}
        for key in keys:
            if key == "q":
                keys_to_mx[key] = q_mx
            elif key == "qdot":
                keys_to_mx[key] = qdot_mx
            elif key == "lm":
                keys_to_mx[key] = lm_normalized_mx
            elif key == "activations":
                keys_to_mx[key] = activations_mx
            elif key == "vm":
                keys_to_mx[key] = vm_normalized_mx
            else:
                raise ValueError(f"Key {key} is not recognized")

        return Function(
            "f", [q_mx, qdot_mx, lm_normalized_mx, activations_mx, vm_normalized_mx], [mx_function(**keys_to_mx)]
        )

    def evaluate_function(self, function_to_evaluate: Function, **kwargs) -> DM:
        """
        Evaluate a CasADi Function with specific parameters. The function must be created with to_function.

        Parameters
        ----------
        function_to_evaluate: Function
            The CasADi Function to evaluate.
        **kwargs
            The values of the variables at which to evaluate the function, limited to q, qdot, lm, act, vm. The
            default values are 0.

        Returns
        -------
        DM
            The result of the evaluation
        """

        def get_value_from_kwargs(key, default):
            if key in kwargs:
                return kwargs[key]
            return default

        q = get_value_from_kwargs("q", [0] * self.nb_q)
        qdot = get_value_from_kwargs("qdot", [0] * self.nb_qdot)
        lm_normalized = get_value_from_kwargs("lm", [0] * self.nb_muscles)
        activations = get_value_from_kwargs("activations", [0] * self.nb_muscles)
        vm_normalized = get_value_from_kwargs("vm", [0] * self.nb_muscles)

        return function_to_evaluate(q, qdot, lm_normalized, activations, vm_normalized)

    def evaluate_mx(self, mx_to_evaluate: MX, **kwargs) -> DM:
        """
        Evaluate a CasADi MX with specific parameters.

        Parameters
        ----------
        mx_to_evaluate: MX
            The CasADi MX to evaluate.
        **kwargs
            The values of the variables at which to evaluate the function, limited to q, qdot, lm, act, vm. The
            default values are 0.

        Returns
        -------
        DM
            The result of the evaluation
        """
        func = self.to_casadi_function(mx_to_evaluate, *kwargs.keys())
        return self.evaluate_function(func, **kwargs)

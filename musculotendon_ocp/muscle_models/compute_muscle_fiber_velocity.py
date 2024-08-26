from functools import cached_property

import biorbd_casadi as biorbd
from casadi import MX, Function, rootfinder

from .muscle_model_abstract import MuscleModelAbstract
from .compute_muscle_fiber_length import ComputeMuscleFiberLengthRigidTendon


class ComputeMuscleFiberVelocityAsVariable:
    """
    This method does not actually compute the muscle fiber velocity but returns a variable that represents the muscle
    fiber velocity. This can be useful when the muscle fiber velocity is a variable in the optimization problem.
    """

    def __init__(self, mx_symbolic: MX = None) -> None:
        self._mx_variable = MX.sym("muscle_fiber_velocity", 1, 1) if mx_symbolic is None else mx_symbolic

    @cached_property
    def mx_variable(self) -> MX:
        return self._mx_variable

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
    ) -> biorbd.MX:
        return self.mx_variable


class ComputeMuscleFiberVelocityRigidTendon(ComputeMuscleFiberVelocityAsVariable):
    """
    This method assumes that the muscle model has a rigid tendon, meaning that the muscle fiber length is equal to the
    musculo-tendon length minus the tendon slack length. The velocity is therefore the velocity of the points on which the
    muscle is attached.
    """

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
    ) -> biorbd.MX:
        biorbd_muscle.updateOrientations(model_kinematic_updated, q, qdot)

        mus_position: biorbd.MuscleGeometry = biorbd_muscle.position()
        mus_jacobian = mus_position.jacobianLength().to_mx()

        return mus_jacobian @ qdot


class ComputeMuscleFiberVelocityFlexibleTendonImplicit(ComputeMuscleFiberVelocityAsVariable):
    """
    Compute the muscle fiber velocity by inverting the force-velocity relationship.
    """

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
    ) -> biorbd.MX:
        if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon")

        # Alias for the MX variables
        activation_mx = MX.sym("activation", 1, 1)
        muscle_fiber_velocity_mx = self.mx_variable
        muscle_fiber_length_mx = muscle.compute_muscle_fiber_length.mx_variable

        # Compute necessary variables
        biorbd_muscle.updateOrientations(model_kinematic_updated, q)
        tendon_length = muscle.compute_tendon_length(
            biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx(), muscle_fiber_length
        )

        # Compute the muscle and tendon forces
        force_tendon = muscle.compute_tendon_force(tendon_length=tendon_length)
        force_muscle = muscle.compute_muscle_force(
            activation=activation_mx,
            muscle_fiber_length=muscle_fiber_length_mx,
            muscle_fiber_velocity=muscle_fiber_velocity_mx,
        )

        # The muscle_fiber_length is found when it equates the muscle and tendon forces
        equality_constraint = Function(
            "g",
            [
                muscle_fiber_velocity_mx,
                muscle_fiber_length_mx,
                activation_mx,
                q if isinstance(q, MX) else MX.sym("dummy_q"),
            ],
            [force_muscle - force_tendon],
        )

        # TODO CHECK IF WE CAN USE THIS EXPLICIT FORMULA INSTEAD OF THE IMPLICIT ONE
        # NOTE: IF NO DAMPING THIS CAN RETURN IMMEDIATELY, OTHERWISE WE NEED TO USE THE ROOTFINDER
        # return muscle._vmax  * muscle._force_velocity_inverse(
        #     (force_tendon / cos(muscle.pennation) - muscle._passive_force - muscle._force_damping)
        #     / (activation * muscle._force_active)
        # )

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": True, "enable_fd": False, "print_in": False, "print_out": False, "max_num_dir": 10},
        )

        return newton_method(i0=0, i1=muscle_fiber_length, i2=activation, i3=q)["o0"]


# TODO Implement the linearized version of the flexible tendon

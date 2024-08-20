import biorbd_casadi as biorbd
from casadi import MX, Function, rootfinder

from .muscle_model_abstract import MuscleModelAbstract


class ComputeMuscleFiberVelocityRigidTendon:
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
        tendon_length: MX,
    ) -> biorbd.MX:
        biorbd_muscle.updateOrientations(model_kinematic_updated, q, qdot)

        mus_position: biorbd.MuscleGeometry = biorbd_muscle.position()
        mus_jacobian = mus_position.jacobianLength().to_mx()

        return mus_jacobian @ qdot


class ComputeMuscleFiberVelocityFlexibleTendon:
    """
    Compute the muscle fiber velocity by inverting the force-velocity relationship.
    """

    def __init__(self, mx_symbolic: MX = None) -> None:
        # TODO TEST THIS
        self.mx_variable = MX.sym("muscle_fiber_velocity", 1, 1) if mx_symbolic is None else mx_symbolic

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        tendon_length: MX,
    ) -> biorbd.MX:

        force_tendon = muscle.compute_tendon_force(tendon_length=tendon_length)
        force_muscle = muscle.compute_muscle_force(
            activation=activation, muscle_fiber_length=muscle_fiber_length, muscle_fiber_velocity=self.mx_variable
        )

        # TODO RENDU ICI!
        muscle_fiber_length_mx = MX.sym("muscle_fiber_length", 1, 1)
        equality_constraint = Function(
            "g", [self.mx_variable, muscle_fiber_length_mx, activation, q], [force_muscle - force_tendon]
        )

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": False, "enable_fd": False, "print_in": False, "print_out": False, "max_num_dir": 10},
        )
        return newton_method(i0=self.mx_variable, i1=muscle_fiber_length, i2=activation, i3=q)["o0"]

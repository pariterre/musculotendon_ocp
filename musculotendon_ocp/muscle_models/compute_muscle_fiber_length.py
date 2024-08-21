import biorbd_casadi as biorbd
from casadi import MX, Function, rootfinder

from .muscle_model_abstract import MuscleModelAbstract


class ComputeMuscleFiberLengthRigidTendon:
    """
    This method assumes that the muscle model has a rigid tendon, meaning that the muscle fiber length is equal to the
    musculo-tendon length minus the tendon slack length.
    """

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
    ) -> biorbd.MX:
        biorbd_muscle.updateOrientations(model_kinematic_updated, q)
        muscle_tendon_length = biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx()

        # TODO Include the pennation angle?
        return muscle_tendon_length - muscle.tendon_slack_length


class ComputeMuscleFiberLengthAsVariable:
    """
    This method does not actually compute the muscle fiber length but returns a variable that represents the muscle. This
    can be useful when the muscle fiber length is a variable in the optimization problem.
    """

    def __init__(self, mx_symbolic: MX = None) -> None:
        self.mx_variable = MX.sym("muscle_fiber_length", 1, 1) if mx_symbolic is None else mx_symbolic

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
    ) -> biorbd.MX:
        # TODO Include the pennation angle?
        return self.mx_variable


class ComputeMuscleFiberLengthInstantaneousEquilibrium(ComputeMuscleFiberLengthAsVariable):
    """
    This method computes the muscle fiber length using the instantaneous equilibrium between the muscle and tendon, that
    is finding the muscle fiber length that satisfies the equation:
    force_muscle(activation, muscle_fiber_length, muscle_fiber_velocity) = force_tendon(tendon_length)
    where tendon_length = musculo_tendon_length - pennated_muscle_fiber_length, and muscle_fiber_velocity is 0.
    """

    def __call__(
        self,
        muscle: MuscleModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
    ) -> biorbd.MX:
        from .hill.muscle_model_hill_flexible_tendon import MuscleModelHillFlexibleTendon

        if not isinstance(muscle, MuscleModelHillFlexibleTendon):
            raise ValueError("The muscle model must be a flexible tendon to compute the instantaneous equilibrium")

        # Alias for the MX variables
        muscle_fiber_length_mx = self.mx_variable

        # Compute the tendon length
        muscle_tendon_length = biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx()
        tendon_length = muscle.compute_tendon_length(
            muscle_tendon_length=muscle_tendon_length, muscle_fiber_length=muscle_fiber_length_mx
        )

        # Compute the muscle and tendon forces
        force_tendon = muscle.compute_tendon_force(tendon_length=tendon_length)
        force_muscle = muscle.compute_muscle_force(
            activation=activation,
            muscle_fiber_length=muscle_fiber_length_mx,
            muscle_fiber_velocity=0,
        )

        # The muscle_fiber_length is found when it equates the muscle and tendon forces
        equality_constraint = Function(
            "g",
            [
                muscle_fiber_length_mx,
                activation if isinstance(activation, MX) else MX.sym("dummy_activation"),
                q if isinstance(q, MX) else MX.sym("dummy_q"),
            ],
            [force_muscle - force_tendon],
        )

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": False, "enable_fd": False, "print_in": False, "print_out": False, "max_num_dir": 10},
        )
        # Evaluate the muscle fiber length
        return newton_method(i1=activation, i2=q)["o0"]

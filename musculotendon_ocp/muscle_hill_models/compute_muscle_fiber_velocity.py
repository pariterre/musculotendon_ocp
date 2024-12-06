from enum import Enum
from functools import cached_property

import biorbd_casadi as biorbd
from casadi import MX, Function, rootfinder, symvar, cos, sqrt

from .muscle_hill_model_abstract import MuscleHillModelAbstract, ComputeMuscleFiberVelocity
from .compute_muscle_fiber_length import ComputeMuscleFiberLengthRigidTendon


"""
Implementations of th ComputeMuscleFiberVelocity protocol
"""


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
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
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
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
    ) -> biorbd.MX:
        biorbd_muscle.updateOrientations(model_kinematic_updated, q, qdot)

        mus_position: biorbd.MuscleGeometry = biorbd_muscle.position()
        mus_jacobian = mus_position.jacobianLength().to_mx()

        return mus_jacobian @ qdot


class ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects(ComputeMuscleFiberVelocityAsVariable):
    """
    Compute the muscle fiber velocity by inverting the force-velocity relationship.
    """

    def __init__(self, mx_symbolic: MX = None) -> None:
        """
        Initialize the ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects class.

        Parameters
        ----------
        mx_symbolic: MX
            The symbolic variable representing the muscle fiber velocity.
        """
        super().__init__(mx_symbolic)

    def __call__(
        self,
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
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
            biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx(), muscle_fiber_length_mx
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

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": True},
        )

        return newton_method(i0=muscle_fiber_velocity_initial_guess, i1=muscle_fiber_length, i2=activation, i3=q)["o0"]


class ComputeMuscleFiberVelocityFlexibleTendonFromVelocityDefects(ComputeMuscleFiberVelocityAsVariable):
    """
    Compute the muscle fiber velocity by inverting the force-velocity relationship.
    """

    def __call__(
        self,
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
    ) -> biorbd.MX:
        if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon")

        # Alias for the MX variables
        activation_mx = MX.sym("activation", 1, 1)
        muscle_fiber_velocity_mx = self.mx_variable
        muscle_fiber_length_mx = muscle.compute_muscle_fiber_length.mx_variable

        # Compute necessary variables
        biorbd_muscle.updateOrientations(model_kinematic_updated, q)
        muscle_tendon_length = biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx()
        tendon_length = muscle.compute_tendon_length(muscle_tendon_length, muscle_fiber_length)

        # Get the normalized muscle length and velocity
        normalized_length = muscle.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = muscle.normalize_muscle_fiber_velocity(muscle_fiber_velocity_mx)

        # Compute the passive, active, velocity and damping factors and the normalized tendon force
        pennation_angle = muscle.compute_pennation_angle(muscle_fiber_length)
        force_passive = muscle.compute_force_passive(normalized_length)
        force_active = muscle.compute_force_active(normalized_length)
        force_damping = muscle.compute_force_damping(normalized_velocity)
        normalized_tendon_force = muscle.compute_tendon_force(tendon_length) / muscle.maximal_force

        # Compute the muscle fiber velocity
        computed_normalized_velocity = muscle.compute_force_velocity.inverse(
            force_velocity_inverse=((normalized_tendon_force / cos(pennation_angle)) - force_passive - force_damping)
            / (activation_mx * force_active)
        )
        muscle_velocity = muscle.denormalize_muscle_fiber_velocity(
            normalized_muscle_fiber_velocity=computed_normalized_velocity
        )

        # If the muscle_fiber_velocity does not depends on the muscle_fiber_velocity_mx, then the muscle_fiber_velocity
        # can be directly computed. Otherwise, the muscle_fiber_velocity_mx is the unknown value that rootfinder tries
        # to optimize.
        if muscle_fiber_velocity_mx.name() not in [var.name() for var in symvar(muscle_velocity)]:
            # Remove the activation_mx
            return Function(
                "tp",
                [
                    activation_mx,
                    muscle_fiber_length_mx,
                    q if isinstance(q, MX) else MX.sym("dummy_q"),
                ],
                [muscle_velocity],
            )(activation, muscle_fiber_length_mx, q)

        # The muscle_fiber_length is found when it equates the muscle and tendon forces
        equality_constraint = Function(
            "g",
            [
                muscle_fiber_velocity_mx,
                muscle_fiber_length_mx,
                activation_mx,
                q if isinstance(q, MX) else MX.sym("dummy_q"),
            ],
            [muscle_velocity - muscle_fiber_velocity_mx],
        )

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": True},
        )

        return newton_method(i0=muscle_fiber_velocity_initial_guess, i1=muscle_fiber_length, i2=activation, i3=q)["o0"]


class ComputeMuscleFiberVelocityFlexibleTendonLinearized(ComputeMuscleFiberVelocityAsVariable):
    """
    Compute the muscle fiber velocity by approximating the force-velocity relationship with a linear approximation.

    The function uses the first order Taylor expansion of the force-velocity relationship around the normalized muscle
    fiber velocity. The Taylor expansion is given by:
    f(v) = f(v0) + f'(v0) * (v - v0)
    """

    def __call__(
        self,
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
    ) -> biorbd.MX:
        if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon")

        # Compute necessary variables
        biorbd_muscle.updateOrientations(model_kinematic_updated, q)
        muscle_tendon_length = biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx()
        tendon_length = muscle.compute_tendon_length(muscle_tendon_length, muscle_fiber_length)

        # Compute some normalized values
        normalized_length = muscle.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = muscle.normalize_muscle_fiber_velocity(muscle_fiber_velocity_initial_guess)
        pennation_angle = muscle.compute_pennation_angle(muscle_fiber_length)

        # Compute the normalized forces
        force_passive = muscle.compute_force_passive(normalized_length)
        force_active = muscle.compute_force_active(normalized_length)
        normalized_tendon_force = muscle.compute_tendon_force(tendon_length) / muscle.maximal_force

        # Compute linear approximation of the muscle fiber velocity
        derivative = muscle.compute_force_velocity.first_derivative(normalized_velocity)
        slope = derivative
        bias = -derivative * normalized_velocity + muscle.compute_force_velocity(normalized_velocity)

        muscle_velocity = muscle.denormalize_muscle_fiber_velocity(
            normalized_muscle_fiber_velocity=(
                (normalized_tendon_force / cos(pennation_angle)) - force_passive - bias * activation * force_active
            )
            / (slope * activation * force_active + muscle.compute_force_damping.factor)
        )

        return muscle_velocity


class ComputeMuscleFiberVelocityFlexibleTendonQuadratic(ComputeMuscleFiberVelocityAsVariable):
    """
    Compute the muscle fiber velocity by approximating the force-velocity relationship with a quadratic approximation.

    The function uses the 2nd order Taylor expansion of the force-velocity relationship around the normalized muscle
    fiber velocity. The Taylor expansion is given by:
    f(v) = f(v0) + f'(v0) * (v - v0) + f''(v0) * (v - v0)^2 / 2
    """

    def __call__(
        self,
        muscle: MuscleHillModelAbstract,
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
    ) -> biorbd.MX:
        if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon")

        # Compute necessary variables
        biorbd_muscle.updateOrientations(model_kinematic_updated, q)
        muscle_tendon_length = biorbd_muscle.musculoTendonLength(model_kinematic_updated, q).to_mx()
        tendon_length = muscle.compute_tendon_length(muscle_tendon_length, muscle_fiber_length)

        # Compute some normalized values
        normalized_length = muscle.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = muscle.normalize_muscle_fiber_velocity(muscle_fiber_velocity_initial_guess)
        pennation_angle = muscle.compute_pennation_angle(muscle_fiber_length)

        # Compute the normalized forces
        force_passive = muscle.compute_force_passive(normalized_length)
        force_active = muscle.compute_force_active(normalized_length)
        normalized_tendon_force = muscle.compute_tendon_force(tendon_length) / muscle.maximal_force

        # Compute the derivatives
        first_derivative = muscle.compute_force_velocity.first_derivative(normalized_velocity)
        second_derivative = muscle.compute_force_velocity.second_derivative(normalized_velocity)

        # Compute the polynomial coefficients of the Taylor expansion at second order of force-velocity relationship
        biais = (
            muscle.compute_force_velocity(normalized_velocity)
            - first_derivative * normalized_velocity
            + second_derivative * normalized_velocity**2 / 2
        )
        slope = first_derivative - second_derivative * normalized_velocity
        quadratic_coeff = second_derivative / 2

        # Compute the polynomial coefficients of the differential equation of the muscle equilibrium equation
        polynomial_quadratic_coeff = activation * force_active * quadratic_coeff
        polynomial_slope = activation * force_active * slope + muscle.compute_force_damping.factor
        polynomial_bias = (
            force_passive - (normalized_tendon_force / cos(pennation_angle)) + biais * activation * force_active
        )

        # Compute the roots of the polynomial
        discriminant = polynomial_slope**2 - 4 * polynomial_quadratic_coeff * polynomial_bias

        # We may have to switch from the first root to the second one at some velocity levels, e.g. positive or negative
        computed_normalized_velocity = (-polynomial_slope + sqrt(discriminant)) / (2 * polynomial_quadratic_coeff)
        # or computed_normalized_velocity = (-polynomial_slope - sqrt(discriminant)) / (2 * polynomial_quadratic_coeff)

        return muscle.denormalize_muscle_fiber_velocity(normalized_muscle_fiber_velocity=computed_normalized_velocity)


class ComputeMuscleFiberVelocityMethods(Enum):
    RigidTendon = ComputeMuscleFiberVelocityRigidTendon
    FlexibleTendonFromForceDefects = ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects
    FlexibleTendonFromVelocityDefects = ComputeMuscleFiberVelocityFlexibleTendonFromVelocityDefects
    FlexibleTendonLinearized = ComputeMuscleFiberVelocityFlexibleTendonLinearized
    FlexibleTendonQuadratic = ComputeMuscleFiberVelocityFlexibleTendonQuadratic

    def __call__(self, *args, **kwargs) -> ComputeMuscleFiberVelocity:
        return self.value(*args, **kwargs)

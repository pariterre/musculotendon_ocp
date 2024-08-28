from typing import override

from casadi import MX, exp, cos, Function, sqrt
from scipy.odr import polynomial

from .compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from .compute_muscle_fiber_velocity import (
    ComputeMuscleFiberVelocityRigidTendon,
    ComputeMuscleFiberVelocityFlexibleTendonImplicit,
)
from .muscle_hill_model_rigid_tendon import MuscleHillModelRigidTendon
from .muscle_hill_model_abstract import ComputeMuscleFiberLength, ComputeMuscleFiberVelocity


class MuscleHillModelFlexibleTendon(MuscleHillModelRigidTendon):
    def __init__(
        self,
        name: str,
        c1: float = 0.2,
        c2: float = 0.995,
        c3: float = 0.250,
        kt: float = 35.0,
        compute_muscle_fiber_length: ComputeMuscleFiberLength | None = None,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocity | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tendon_slack_length: MX
            The tendon slack length
        """
        if compute_muscle_fiber_length is None:
            compute_muscle_fiber_length = ComputeMuscleFiberLengthInstantaneousEquilibrium()
        if isinstance(compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must be a flexible tendon")

        if compute_muscle_fiber_velocity is None:
            compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityFlexibleTendonImplicit()
        if isinstance(compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityRigidTendon):
            raise ValueError("The compute_muscle_fiber_velocity must be a flexible tendon")

        super().__init__(
            name=name,
            compute_muscle_fiber_length=compute_muscle_fiber_length,
            compute_muscle_fiber_velocity=compute_muscle_fiber_velocity,
            **kwargs,
        )

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.kt = kt

    @override
    def compute_muscle_fiber_velocity_from_second_order_approximation(
        self,
        activation: MX,
        muscle_fiber_velocity: MX,
        muscle_fiber_length: MX,
        tendon_length: MX,
    ) -> MX:
        # Compute some normalized values
        normalized_length = self.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = self.normalize_muscle_fiber_velocity(muscle_fiber_velocity)
        pennation_angle = self.compute_pennation_angle(muscle_fiber_length)

        # Compute the normalized forces
        force_passive = self.compute_force_passive(normalized_length)
        force_active = self.compute_force_active(normalized_length)
        normalized_tendon_force = self.compute_tendon_force(tendon_length) / self.maximal_force

        # Compute the derivatives
        derivative = self.compute_force_velocity.derivative(normalized_velocity)
        second_derivative = self.compute_force_velocity.second_derivative(normalized_velocity)

        # Compute the polynomial coefficients of the Taylor expansion at second order of force-velocity relationship
        biais = (
            self.compute_force_velocity(normalized_velocity)
            - derivative * normalized_velocity
            + second_derivative * normalized_velocity**2 / 2
        )
        slope = derivative - second_derivative * normalized_velocity
        quadratic_coeff = second_derivative / 2

        # Compute the polynomial coefficients of the differential equation of the muscle equilibrium equation
        polynomial_quadratic_coeff = activation * force_active * quadratic_coeff
        polynomial_slope = activation * force_active * slope + self.compute_force_damping.factor
        polynomial_bias = (
            force_passive - (normalized_tendon_force / cos(pennation_angle)) + biais * activation * force_active
        )

        # Compute the roots of the polynomial
        discriminant = polynomial_slope**2 - 4 * polynomial_quadratic_coeff * polynomial_bias

        # We may have to switch from the first root to the second one at some velocity levels, e.g. positive or negative
        computed_normalized_velocity = (-polynomial_slope + sqrt(discriminant)) / (2 * polynomial_quadratic_coeff)
        # or computed_normalized_velocity = (-polynomial_slope - sqrt(discriminant)) / (2 * polynomial_quadratic_coeff)

        return self.denormalize_muscle_fiber_velocity(normalized_muscle_fiber_velocity=computed_normalized_velocity)

    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def denormalize_tendon_length(self, normalized_tendon_length: MX) -> MX:
        return normalized_tendon_length * self.tendon_slack_length

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return muscle_tendon_length - self.compute_pennation_angle.apply(muscle_fiber_length, muscle_fiber_length)

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        normalized_tendon_length = self.normalize_tendon_length(tendon_length)

        return self.c1 * exp(self.kt * (normalized_tendon_length - self.c2)) - self.c3


class MuscleHillModelFlexibleTendonAlwaysPositive(MuscleHillModelFlexibleTendon):
    @property
    def offset(self) -> float:
        """
        Get the offset to ensure the tendon force is always positive, by offsetting the force by the value at slack length
        """
        return super(MuscleHillModelFlexibleTendonAlwaysPositive, self).compute_tendon_force(self.tendon_slack_length)

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return (
            super(MuscleHillModelFlexibleTendonAlwaysPositive, self).compute_tendon_force(tendon_length) - self.offset
        )

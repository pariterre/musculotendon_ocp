from typing import override

from casadi import MX, exp, Function, rootfinder

from .muscle_model_hill_rigid_tendon import MuscleModelHillRigidTendon
from ..muscle_model_abstract import ComputeMuscleFiberLengthCallable
from ..compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)


class MuscleModelHillFlexibleTendon(MuscleModelHillRigidTendon):
    def __init__(
        self,
        c1: float = 0.2,
        c2: float = 0.995,
        c3: float = 0.250,
        kt: float = 35.0,
        compute_muscle_fiber_length: ComputeMuscleFiberLengthCallable = ComputeMuscleFiberLengthInstantaneousEquilibrium(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        tendon_slack_length: MX
            The tendon slack length
        """
        if isinstance(compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must be a flexible tendon")
        super().__init__(compute_muscle_fiber_length=compute_muscle_fiber_length, **kwargs)

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.kt = kt

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        normalized_tendon_length = self.normalize_tendon_length(tendon_length)
        offset = 0.01175075667752834

        return self.c1 * exp(self.kt * (normalized_tendon_length - self.c2)) - self.c3 + offset

    def compute_muscle_fiber_length_derivative(self, activation: MX, muscle_fiber_length: MX, tendon_length: MX) -> MX:
        """
        Compute the muscle fiber length derivative (muscle fiber velocity) using the Hill-type muscle model with a
        flexible tendon.

        Parameters
        ----------
        activation: MX
            The muscle activation
        muscle_fiber_length: MX
            The muscle fiber length
        tendon_length: MX
            The tendon length

        Returns
        -------
        MX
            The muscle fiber length derivative (muscle fiber velocity)
        """
        muscle_fiber_velocity = MX.sym("muscle_fiber_velocity", 1, 1)

        force_tendon = self.compute_tendon_force(tendon_length=tendon_length)
        force_muscle = self.compute_muscle_force(
            activation=activation, muscle_fiber_length=muscle_fiber_length, muscle_fiber_velocity=muscle_fiber_velocity
        )

        equality_constraint = Function("g", [muscle_fiber_velocity], [force_muscle - force_tendon])

        # Reminder: the first variable of the function is the unknown value that rootfinder tries to optimize.
        # The others are parameters. To have more unknown values to calculate, one needs to use the vertcat function.
        newton_method = rootfinder(
            "newton_method",
            "newton",
            equality_constraint,
            {"error_on_fail": True, "enable_fd": False, "print_in": False, "print_out": False, "max_num_dir": 10},
        )
        return newton_method()["o0"]


class MuscleModelHillFlexibleTendonLinearized(MuscleModelHillFlexibleTendon):
    @override
    def compute_muscle_fiber_length_derivative(self, activation: MX, muscle_fiber_length: MX, tendon_length: MX) -> MX:
        raise NotImplementedError("The linearized muscle model is not implemented yet")

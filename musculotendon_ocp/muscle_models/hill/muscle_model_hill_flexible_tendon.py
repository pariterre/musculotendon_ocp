from typing import override

from casadi import MX, exp

from ..muscle_model_abstract import MuscleModelAbstract


class MuscleModelHillFlexibleTendon(MuscleModelAbstract):
    def __init__(
        self,
        tendon_slack_length: float,
        c1: float = 0.2,
        c2: float = 0.995,
        c3=0.250,
        kt=35,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tendon_slack_length: MX
            The tendon slack length
        """
        super().__init__(**kwargs)

        self.tendon_slack_length = tendon_slack_length

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.kt = kt

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        """
        Compute the tendon force
        An offset is added to the force to avoid negative forces when the tendon is slack
        """

        normalized_tendon_length = self.normalize_tendon_length(tendon_length)
        offset = 0.01175075667752834

        return self.c1 * exp(self.kt * (normalized_tendon_length - self.c2)) - self.c3 + offset

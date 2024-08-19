from casadi import MX, log, sqrt


class ForceVelocityHillType:
    def __init__(
        self,
        d1: float = -0.318,
        d2: float = -8.149,
        d3: float = -0.374,
        d4: float = 0.886,
    ):
        # TODO The default values may need to be more precise
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the force-velocity relationship

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle velocity
        """
        # alias so the next line is not too long
        velocity = normalized_muscle_fiber_velocity

        return self.d1 * log((self.d2 * velocity + self.d3) + sqrt(((self.d2 * velocity + self.d3) ** 2) + 1)) + self.d4

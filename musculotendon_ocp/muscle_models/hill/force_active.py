from casadi import MX, exp


class ForceActiveHillType:
    def __init__(
        self,
        b11: float = 0.814483478343008,
        b21: float = 1.055033428970575,
        b31: float = 0.162384573599574,
        b41: float = 0.063303448465465,
        b12: float = 0.433004984392647,
        b22: float = 0.716775413397760,
        b32: float = -0.029947116970696,
        b42: float = 0.200356847296188,
        b13: float = 0.100,
        b23: float = 1.000,
        b33: float = 0.354,
        b43: float = 0.000,
    ):
        # TODO The default values may need to be more precise
        self.b11 = b11
        self.b21 = b21
        self.b31 = b31
        self.b41 = b41
        self.b12 = b12
        self.b22 = b22
        self.b32 = b32
        self.b42 = b42
        self.b13 = b13
        self.b23 = b23
        self.b33 = b33
        self.b43 = b43

    def __call__(self, normalized_muscle_fiber_length: MX) -> MX:
        """
        Compute the normalized force from the passive force-length relationship

        Parameters
        ----------
        normalized_muscle_length: MX
            The normalized muscle length

        Returns
        -------
        MX
            The normalized passive force corresponding to the given muscle length
        """

        length = normalized_muscle_fiber_length  # alias so the next line is not too long
        return (
            self.b11 * exp((-0.5) * ((length - self.b21) ** 2) / ((self.b31 + self.b41 * length) ** 2))
            + self.b12 * exp((-0.5) * (length - self.b22) ** 2 / ((self.b32 + self.b42 * length) ** 2))
            + self.b13 * exp((-0.5) * (length - self.b23) ** 2 / ((self.b33 + self.b43 * length) ** 2))
        )

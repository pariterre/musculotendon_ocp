from casadi import MX, exp


class ForcePassiveHillType:
    def __init__(self, kpe: float = 4.0, e0: float = 0.6):
        self.kpe = kpe
        self.e0 = e0

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
        return (exp(self.kpe * (normalized_muscle_fiber_length - 1) / self.e0) - 1) / (exp(self.kpe) - 1)


class ForcePassiveAlwaysPositiveHillType(ForcePassiveHillType):
    @property
    def offset(self) -> float:
        """
        Get the offset to ensure the force is always positive, by offsetting the force by the minimum value
        """
        return super(ForcePassiveAlwaysPositiveHillType, self).__call__(0.0)

    def __call__(self, normalized_muscle_fiber_length: MX) -> MX:
        """
        Same as ForcePassiveHillType but an offset is added to ensure the force is always positive
        """
        return super(ForcePassiveAlwaysPositiveHillType, self).__call__(normalized_muscle_fiber_length) - self.offset

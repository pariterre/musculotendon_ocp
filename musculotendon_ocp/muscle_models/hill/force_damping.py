from casadi import MX, exp


class ForceDampingConstant:
    def __init__(self, factor: float = 0.0):
        self._factor = factor

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the damping

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle velocity
        """
        return self._factor


class ForceDampingLinear:
    def __init__(self, factor: float = 0.1):
        self._factor = factor

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the damping

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle velocity
        """
        return self._factor * normalized_muscle_fiber_velocity

from casadi import MX, cos, sin, asin


class ComputePennationAngleConstant:
    def __init__(self, pennation_angle: float = 0.0):
        if pennation_angle < 0.0:
            raise ValueError("The pennation angle must be positive")
        self.pennation_angle = pennation_angle

    def __call__(self, muscle_fiber_length: MX) -> MX:
        return cos(self.pennation_angle)


class ComputePennationAngleWrtMuscleFiberLength:
    def __init__(self, optimal_pennation_angle: float = 0.0, optimal_muscle_fiber_length: float = 0.0):
        if optimal_pennation_angle < 0.0:
            raise ValueError("The optimal pennation angle must be positive")

        self.optimal_pennation_angle = optimal_pennation_angle
        self.optimal_muscle_fiber_length = optimal_muscle_fiber_length

    def __call__(self, muscle_fiber_length: MX) -> MX:
        return cos(asin(self.optimal_muscle_fiber_length * sin(self.optimal_pennation_angle) / muscle_fiber_length))

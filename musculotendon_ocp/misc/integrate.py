from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp


def precise_rk45(dynamics_func: Callable, y0: np.array, t_span: tuple[float, float], dt: float):
    n_steps = int(t_span[1] / dt)
    if n_steps != t_span[1] / dt:
        raise ValueError("The final time should be a multiple of the time step")

    time_vector = np.linspace(*t_span, n_steps + 1)
    integrated_values = np.ndarray((len(y0), n_steps + 1))
    integrated_values[:, 0] = y0
    for i in range(n_steps):
        t_span_i = (i * dt, (i + 1) * dt)
        out = solve_ivp(dynamics_func, t_span_i, y0).y

        y0 = out[:, -1]
        integrated_values[:, i + 1] = y0

    return time_vector, integrated_values

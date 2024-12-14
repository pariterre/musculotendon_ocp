from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp as _solve_ivp_rk45


def precise_rk45(
    dynamics_func: Callable,
    y0: np.array,
    t_span: tuple[float, float],
    dt: float,
    inter_step_callback: Callable | None = None,
) -> tuple[np.array, np.array]:
    """
    Compute the solution of an initial value problem for a system of ordinary differential equations using the
    Runge-Kutta 4th order method, with adaptive step size.
    The function is called precise because it calls the solver for each time step, instead of calling it once for the
    whole time span. The intermediate values (sub_dt) of the rk45 are however dropped.
    The t_span and dt dictate the number of steps to be taken.

    Parameters
    ----------
    dynamics_func: Callable
        The function that computes the derivative of the system of ODEs
    y0: np.array
        The initial values of the system
    t_span: tuple[float, float]
        The time span of the integration
    dt: float
        The time step
    inter_step_callback: Callable | None
        The function to call after each integration step. It should take the current time and the current values as
        arguments

    Returns
    -------
    tuple[np.array, np.array]
        The time vector (i.e. t_span[0] to t_span[1] by steps of dt) and the integrated values at each time step
    """
    return _perform_precise_integration(_solve_ivp_rk45, dynamics_func, y0, t_span, dt, inter_step_callback)


def precise_rk4(
    dynamics_func: Callable,
    y0: np.array,
    t_span: tuple[float, float],
    dt: float,
    inter_step_callback: Callable | None = None,
):
    """
    Compute the solution of an initial value problem for a system of ordinary differential equations using the
    Runge-Kutta 4th order method. The function is called precise because it calls the solver for each time step, instead
    of calling it once for the whole time span. The t_span and dt dictate the number of steps to be taken.

    Parameters
    ----------
    dynamics_func: Callable
        The function that computes the derivative of the system of ODEs
    y0: np.array
        The initial values of the system
    t_span: tuple[float, float]
        The time span of the integration
    dt: float
        The time step
    inter_step_callback: Callable | None
        The function to call after each integration step. It should take the current time and the current values as
        arguments

    Returns
    -------
    tuple[np.array, np.array]
        The time vector (i.e. t_span[0] to t_span[1] by steps of dt) and the integrated values at each time step
    """
    return _perform_precise_integration(_solve_ivp_rk4, dynamics_func, y0, t_span, dt, inter_step_callback)


def _solve_ivp_rk4(dynamics_func: Callable, t_span: tuple[float, float], y0: np.array) -> np.array:
    t0 = t_span[0]
    dt = t_span[1] - t_span[0]

    k1 = dt * dynamics_func(t0, y0)
    k2 = dt * dynamics_func(t0 + dt / 2, y0 + k1 / 2)
    k3 = dt * dynamics_func(t0 + dt / 2, y0 + k2 / 2)
    k4 = dt * dynamics_func(t0 + dt, y0 + k3)

    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def precise_rk1(
    dynamics_func: Callable,
    y0: np.array,
    t_span: tuple[float, float],
    dt: float,
    inter_step_callback: Callable | None = None,
) -> tuple[np.array, np.array]:
    """
    Compute the solution of an initial value problem for a system of ordinary differential equations using the
    Runge-Kutta 1st order method (Euler's method). The function is called precise because it calls the solver for each
    time step, instead of calling it once for the whole time span. The t_span and dt dictate the number of steps to be taken.

    Parameters
    ----------
    dynamics_func: Callable
        The function that computes the derivative of the system of ODEs
    y0: np.array
        The initial values of the system
    t_span: tuple[float, float]
        The time span of the integration
    dt: float
        The time step
    inter_step_callback: Callable | None
        The function to call after each integration step. It should take the current time and the current values as
        arguments

    Returns
    -------
    tuple[np.array, np.array]
        The time vector (i.e. t_span[0] to t_span[1] by steps of dt) and the integrated values at each time step
    """
    return _perform_precise_integration(_solve_ivp_rk1, dynamics_func, y0, t_span, dt)


def _solve_ivp_rk1(dynamics_func: Callable, t_span: tuple[float, float], y0: np.array) -> np.array:
    t0 = t_span[0]
    dt = t_span[1] - t_span[0]

    k1 = dt * dynamics_func(t0, y0)

    return y0 + k1


def _perform_precise_integration(
    solver: Callable,
    dynamics_func: Callable,
    y0: np.array,
    t_span: tuple[float, float],
    dt: float,
    inter_step_callback: Callable | None = None,
) -> tuple[np.array, np.array]:
    """
    Perform the integration using the specified solver.

    Parameters
    ----------
    solver: Callable
        The integration solver to use, either _solve_ivp_rk4 or _solve_ivp_rk45
    dynamics_func: Callable
        The function that computes the derivative of the system of ODEs
    y0: np.array
        The initial values of the system
    t_span: tuple[float, float]
        The time span of the integration
    dt: float
        The time step
    inter_step_callback: Callable | None
        The function to call after each integration step. It should take the current time and the current values as
        arguments

    Returns
    -------
    tuple[np.array, np.array]
        The time vector (i.e. t_span[0] to t_span[1] by steps of dt) and the integrated values at each time step
    """
    steps_count = int(t_span[1] / dt)
    if steps_count != t_span[1] / dt:
        raise ValueError("The final time should be a multiple of the time step")

    time_vector = np.linspace(*t_span, steps_count + 1)
    values = np.ndarray((len(y0), steps_count + 1))
    values[:, 0] = y0

    for i in range(steps_count):
        out = solver(dynamics_func, t_span=[i * dt, (i + 1) * dt], y0=y0)
        if solver == _solve_ivp_rk45:
            out = out.y[:, -1]

        y0 = out
        values[:, i + 1] = y0

        if inter_step_callback is not None:
            inter_step_callback(time_vector[i + 1], values[:, i + 1])

    return time_vector, values

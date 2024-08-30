from bioptim import NonLinearProgram, DynamicsFunctions
from casadi import MX, Function
import numpy as np


def casadi_function_to_bioptim_graph(
    function_to_graph: Function,
    muscle_fiber_length_dot_func: Function | None,
    nlp: NonLinearProgram,
    states: MX,
    controls: MX,
):
    # TODO Test this
    all_q = DynamicsFunctions.get(nlp.states["q"], states)
    all_qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    all_muscle_fiber_lengths = DynamicsFunctions.get(nlp.states["muscles_fiber_lengths"], states)
    all_muscle_fiber_velocities = DynamicsFunctions.get(nlp.controls["muscles_fiber_velocities"], controls)
    all_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)

    def to_linear_controls(data):
        return (
            np.concatenate([np.interp(np.linspace(0, 1, states.shape[1]), [0, 1], d)[:, None] for d in data], axis=1).T
            if states.shape[1] > 1
            else data
        )

    all_muscle_fiber_velocities = to_linear_controls(all_muscle_fiber_velocities)
    all_activations = to_linear_controls(all_activations)

    out = np.ndarray((function_to_graph.size1_out(0), states.shape[1]))
    for i, (q, qdot, muscle_fiber_lengths, muscle_fiber_velocities, muscle_activations) in enumerate(
        zip(all_q.T, all_qdot.T, all_muscle_fiber_lengths.T, all_muscle_fiber_velocities.T, all_activations.T)
    ):
        muscle_fiber_lengths_dot = (
            []
            if muscle_fiber_length_dot_func is None
            else muscle_fiber_length_dot_func(
                activations=muscle_activations,
                q=q,
                qdot=qdot,
                muscle_fiber_lengths=muscle_fiber_lengths,
                muscle_fiber_velocity_initial_guesses=muscle_fiber_velocities,
            )["output"].__array__()[:, 0]
        )

        out[:, i] = function_to_graph(
            activations=muscle_activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocities=muscle_fiber_lengths_dot,
            muscle_fiber_velocity_initial_guesses=muscle_fiber_velocities,
        )["output"].__array__()[:, 0]

    return out

from functools import partial

from bioptim import OptimalControlProgram, PlotType

from .casadi_interface import (
    casadi_function_to_bioptim_graph,
    prepare_muscle_forces_mx,
    prepare_tendon_forces_mx,
    prepare_fiber_lmdot_mx,
)
from ..rigidbody_models import RigidbodyModelWithMuscles


def add_tendon_forces_plot_to_ocp(ocp: OptimalControlProgram, model: RigidbodyModelWithMuscles):
    # TODO Test this
    ocp.add_plot(
        "Forces",
        lambda *args: casadi_function_to_bioptim_graph(
            function_to_graph=model.to_casadi_function(
                partial(prepare_tendon_forces_mx, model=model), "activations", "q", "qdot"
            ),
            muscle_fiber_length_dot_func=None,
            nlp=ocp.nlp[0],
            states=args[3],
            controls=args[4],
        ),
        plot_type=PlotType.INTEGRATED,
    )


def add_muscle_forces_plot_to_ocp(ocp: OptimalControlProgram, model: RigidbodyModelWithMuscles):
    # TODO Test this
    ocp.add_plot(
        "Forces",
        lambda *args: casadi_function_to_bioptim_graph(
            function_to_graph=model.to_casadi_function(
                partial(prepare_muscle_forces_mx, model=model),
                "activations",
                "q",
                "qdot",
                "muscle_fiber_lengths",
                "muscle_fiber_velocities",
            ),
            muscle_fiber_length_dot_func=model.to_casadi_function(
                partial(prepare_fiber_lmdot_mx, model=model), "activations", "q", "qdot"
            ),
            nlp=ocp.nlp[0],
            states=args[3],
            controls=args[4],
        ),
        plot_type=PlotType.INTEGRATED,
    )

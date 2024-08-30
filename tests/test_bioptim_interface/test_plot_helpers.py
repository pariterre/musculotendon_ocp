import os

from casadi import Function
from musculotendon_ocp import (
    MuscleHillModels,
    RigidbodyModels,
    CasadiHelpers,
)
import numpy as np

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_add_tendon_forces_plot_to_ocp():
    raise NotImplementedError("Test not implemented")


def test_add_muscle_forces_plot_to_ocp():
    raise NotImplementedError("Test not implemented")


def test_casadi_function_to_bioptim_graph():
    raise NotImplementedError("Test not implemented")

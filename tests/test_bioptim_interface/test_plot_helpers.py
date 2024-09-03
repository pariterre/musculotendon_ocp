import os

from bioptim import OptimalControlProgram, DynamicsList, CustomPlot
from casadi import Function, sum1
from musculotendon_ocp import MuscleHillModels, RigidbodyModels, PlotHelpers, DynamicsHelpers
import numpy as np

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_add_tendon_forces_plot_to_ocp():
    # Create a dummy ocp
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
        ],
    )
    dynamics = DynamicsList()
    dynamics.add(DynamicsHelpers.configure)
    ocp = OptimalControlProgram(model, dynamics, 10, 1)

    PlotHelpers.add_tendon_forces_plot_to_ocp(ocp=ocp, model=model)
    custom_plot: CustomPlot = ocp.nlp[0].plot["Forces"]

    equilibrated_muscle_lengths = np.array(
        model.function_to_dm(
            model.muscle_fiber_lengths_equilibrated,
            activations=0.5,
            q=np.ones((model.nb_q)) * -0.3,
            qdot=np.ones(model.nb_q) * 0.0,
        )
    )

    n_frames = 4
    factor = 0.8
    constant_states = np.concatenate(
        (
            np.ones((model.nb_q, n_frames)) * -0.3,
            np.ones((model.nb_qdot, n_frames)) * 0.0,
            np.repeat(equilibrated_muscle_lengths, n_frames, axis=1),
        )
    )
    linear_states = np.concatenate(
        (
            np.linspace(-0.21, -0.21, n_frames).reshape((model.nb_q, n_frames)),
            np.linspace(0.0, 0.0, n_frames).reshape((model.nb_qdot, n_frames)),
            np.linspace(equilibrated_muscle_lengths, equilibrated_muscle_lengths * factor, n_frames, axis=1)[:, :, 0],
        )
    )
    constant_controls = np.concatenate(
        (
            np.ones((model.nb_muscles, 2)) * 0.5,
            np.ones((model.nb_muscles, 2)) * 10.0,
        ),
    )
    linear_controls = np.concatenate(
        (
            np.concatenate((np.ones((model.nb_muscles, 1)) * 0.5, np.ones((model.nb_muscles, 1)) * 0.8), axis=1),
            np.concatenate((np.ones((model.nb_muscles, 1)) * 10.0, np.ones((model.nb_muscles, 1)) * 10.0), axis=1),
        ),
    )

    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], constant_states, constant_controls),
        np.array([[0.0, 0.0, 0.0, 0.0], [504.52203517, 504.52203517, 504.52203517, 504.52203517]]),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], constant_states, linear_controls),
        np.array([[0.0, 0.0, 0.0, 0.0], [504.52203517, 604.61468743, 704.8127768, 804.99918087]]),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], linear_states, constant_controls),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.36485297, 0.36485297, 0.36485297, 0.36485297]]),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], linear_states, linear_controls),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.36485297, 0.77285056, 1.31373721, 1.97178443]]),
    )


def test_add_muscle_forces_plot_to_ocp():
    # Create a dummy ocp
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
        ],
    )
    dynamics = DynamicsList()
    dynamics.add(DynamicsHelpers.configure)
    ocp = OptimalControlProgram(model, dynamics, 10, 1)

    PlotHelpers.add_muscle_forces_plot_to_ocp(ocp=ocp, model=model)
    custom_plot: CustomPlot = ocp.nlp[0].plot["Forces"]

    equilibrated_muscle_lengths = np.array(
        model.function_to_dm(
            model.muscle_fiber_lengths_equilibrated,
            activations=0.5,
            q=np.ones((model.nb_q)) * -0.3,
            qdot=np.ones(model.nb_q) * 0.0,
        )
    )

    n_frames = 4
    factor = 0.8
    constant_states = np.concatenate(
        (
            np.ones((model.nb_q, n_frames)) * -0.3,
            np.ones((model.nb_qdot, n_frames)) * 0.0,
            np.repeat(equilibrated_muscle_lengths, n_frames, axis=1),
        )
    )
    linear_states = np.concatenate(
        (
            np.linspace(-0.21, -0.21, n_frames).reshape((model.nb_q, n_frames)),
            np.linspace(0.0, 0.0, n_frames).reshape((model.nb_qdot, n_frames)),
            np.linspace(equilibrated_muscle_lengths, equilibrated_muscle_lengths * factor, n_frames, axis=1)[:, :, 0],
        )
    )
    constant_controls = np.concatenate(
        (
            np.ones((model.nb_muscles, 2)) * 0.5,
            np.ones((model.nb_muscles, 2)) * 10.0,
        ),
    )
    linear_controls = np.concatenate(
        (
            np.concatenate((np.ones((model.nb_muscles, 1)) * 0.5, np.ones((model.nb_muscles, 1)) * 0.8), axis=1),
            np.concatenate((np.ones((model.nb_muscles, 1)) * 10.0, np.ones((model.nb_muscles, 1)) * 10.0), axis=1),
        ),
    )

    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], constant_states, constant_controls),
        np.array(
            [
                [440.34935506, 440.34935506, 440.34935506, 440.34935506],
                [504.52203517, 504.52203517, 504.52203517, 504.52203517],
            ]
        ),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], constant_states, linear_controls),
        np.array(
            [
                [440.34935506, 478.44766516, 516.54597526, 554.64428537],
                [504.52203517, 504.52203517, 504.52203517, 504.52203517],
            ]
        ),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], linear_states, constant_controls),
        np.array(
            [
                [440.34935506, 409.4133706, 442.79145401, 488.12233993],
                [-0.24999858, -0.24999343, -0.2499696, -0.24985947],
            ]
        ),
    )
    np.testing.assert_almost_equal(
        custom_plot.function(ocp.nlp[0], [], [], linear_states, linear_controls),
        np.array(
            [
                [440.34935506, 466.20239597, 596.42719338, 767.27652864],
                [-0.24999858, -0.24999343, -0.2499696, -0.24985947],
            ]
        ),
    )


def test_casadi_function_to_bioptim_graph():
    # Create a dummy ocp
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
            MuscleHillModels.FlexibleTendon(
                name="Mus1", maximal_force=1000, optimal_length=0.1, tendon_slack_length=0.16, maximal_velocity=5.0
            ),
        ],
    )
    dynamics = DynamicsList()
    dynamics.add(DynamicsHelpers.configure)
    ocp = OptimalControlProgram(model, dynamics, 10, 1)

    dummy_func = Function(
        "dummy",
        [
            model.activations_mx,
            model.q_mx,
            model.qdot_mx,
            model.muscle_fiber_lengths_mx,
            model.muscle_fiber_velocities_mx,
            model.muscle_fiber_velocity_initial_guesses_mx,
        ],
        [
            sum1(model.activations_mx)
            + sum1(model.q_mx)
            + sum1(model.qdot_mx)
            + sum1(model.muscle_fiber_lengths_mx)
            + sum1(model.muscle_fiber_velocities_mx)
            + sum1(model.muscle_fiber_velocity_initial_guesses_mx)
        ],
        [
            "activations",
            "q",
            "qdot",
            "muscle_fiber_lengths",
            "muscle_fiber_velocities",
            "muscle_fiber_velocity_initial_guesses",
        ],
        ["output"],
    )

    n_frames = 4
    constant_states = np.concatenate(
        (
            np.ones((model.nb_q, n_frames)) * -0.3,
            np.ones((model.nb_qdot, n_frames)) * 0.1,
            np.ones((model.nb_muscles, n_frames)) * 0.5,
        )
    )
    linear_states = np.concatenate(
        (
            np.linspace(-0.21, -0.23, n_frames).reshape((model.nb_q, n_frames)),
            np.linspace(0.0, 0.5, n_frames).reshape((model.nb_qdot, n_frames)),
            np.linspace(np.ones(model.nb_muscles) * 0.5, np.ones(model.nb_muscles) * 0.8, n_frames).T,
        )
    )
    constant_controls = np.concatenate(
        (
            np.ones((model.nb_muscles, 2)) * 0.5,
            np.ones((model.nb_muscles, 2)) * 10.0,
        ),
    )
    linear_controls = np.concatenate(
        (
            np.concatenate((np.ones((model.nb_muscles, 1)) * 0.5, np.ones((model.nb_muscles, 1)) * 0.8), axis=1),
            np.concatenate((np.ones((model.nb_muscles, 1)) * 15.0, np.ones((model.nb_muscles, 1)) * 10.0), axis=1),
        ),
    )

    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=None,
            nlp=ocp.nlp[0],
            states=constant_states,
            controls=constant_controls,
        ),
        np.array([[21.8, 21.8, 21.8, 21.8]]),
    )
    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=dummy_func,
            nlp=ocp.nlp[0],
            states=constant_states,
            controls=constant_controls,
        ),
        np.array([[65.4, 65.4, 65.4, 65.4]]),
    )

    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=None,
            nlp=ocp.nlp[0],
            states=constant_states,
            controls=linear_controls,
        ),
        np.array([[31.8, 28.66666667, 25.53333333, 22.4]]),
    )
    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=dummy_func,
            nlp=ocp.nlp[0],
            states=constant_states,
            controls=linear_controls,
        ),
        np.array([[95.4, 86.0, 76.6, 67.2]]),
    )

    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=None,
            nlp=ocp.nlp[0],
            states=linear_states,
            controls=constant_controls,
        ),
        np.array([[21.79, 22.15, 22.51, 22.87]]),
    )
    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=dummy_func,
            nlp=ocp.nlp[0],
            states=linear_states,
            controls=constant_controls,
        ),
        np.array([[65.37, 66.45, 67.53, 68.61]]),
    )

    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=None,
            nlp=ocp.nlp[0],
            states=linear_states,
            controls=linear_controls,
        ),
        np.array([[31.79, 29.01666667, 26.24333333, 23.47]]),
    )
    np.testing.assert_almost_equal(
        PlotHelpers.casadi_function_to_bioptim_graph(
            function_to_graph=dummy_func,
            muscle_fiber_length_dot_func=dummy_func,
            nlp=ocp.nlp[0],
            states=linear_states,
            controls=linear_controls,
        ),
        np.array([[95.37, 87.05, 78.73, 70.41]]),
    )

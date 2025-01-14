from functools import partial
import pathlib

from casadi import MX
from musculotendon_ocp import RigidbodyModels, MuscleHillModels, ComputeMuscleFiberLengthMethods
import numpy as np
import pytest


model_path = (
    pathlib.Path(__file__).parent.resolve()
    / "../../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod"
).as_posix()


def test_compute_muscle_fiber_length_methods():
    assert len(ComputeMuscleFiberLengthMethods) == 3

    rigid_tendon = ComputeMuscleFiberLengthMethods.RigidTendon()
    assert type(rigid_tendon) == ComputeMuscleFiberLengthMethods.RigidTendon.value

    as_variable = ComputeMuscleFiberLengthMethods.AsVariable()
    assert type(as_variable) == ComputeMuscleFiberLengthMethods.AsVariable.value

    instantaneous_equilibrium = ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium()
    assert type(instantaneous_equilibrium) == ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium.value


def test_compute_muscle_fiber_length_rigid_tendon():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])
    compute_muscle_fiber_length = ComputeMuscleFiberLengthMethods.RigidTendon()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    muscle_fiber_length = model.function_to_dm(
        partial(
            compute_muscle_fiber_length,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            activation=activation,
        ),
        q=q,
        qdot=qdot,
    )
    np.testing.assert_almost_equal(muscle_fiber_length, 0.077)


def test_compute_muscle_fiber_length_as_variable():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_length = ComputeMuscleFiberLengthMethods.AsVariable(mx_symbolic=mx_symbolic)

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)
    assert id(mx_symbolic) == id(
        compute_muscle_fiber_length(
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            activation=activation,
            q=np.array([-0.2]),
            qdot=qdot,
        )
    )

    compute_muscle_fiber_length_default = ComputeMuscleFiberLengthMethods.AsVariable()
    muscle_fiber_length = compute_muscle_fiber_length_default(
        muscle=mus,
        model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
        biorbd_muscle=model.model.muscle(0),
        activation=activation,
        q=np.array([-0.2]),
        qdot=qdot,
    )
    assert isinstance(muscle_fiber_length, MX)
    assert muscle_fiber_length.name() == "muscle_fiber_length"


def test_compute_muscle_fiber_length_instantaneous_equilibrium():
    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_length = ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium()
    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    muscle_fiber_length = float(
        compute_muscle_fiber_length(
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            activation=activation,
            q=np.array([-0.2]),
            qdot=qdot,
        )
    )

    np.testing.assert_almost_equal(muscle_fiber_length, 0.05826843914426753)


def test_compute_muscle_fiber_length_instantaneous_equilibrium_wrong_constructor():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_length = ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium()
    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    with pytest.raises(
        ValueError, match="The muscle model must be a flexible tendon to compute the instantaneous equilibrium"
    ):
        compute_muscle_fiber_length(
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            activation=activation,
            q=np.array([-0.2]),
            qdot=qdot,
        )

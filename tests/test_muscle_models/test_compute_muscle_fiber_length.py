from functools import partial
import os

from casadi import MX
from musculotendon_ocp import (
    MuscleBiorbdModel,
    MuscleModelHillRigidTendon,
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthAsVariable,
)
import numpy as np

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_compute_muscle_fiber_length_rigid_tendon():
    mus = MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123)
    model = MuscleBiorbdModel(model_path, muscles=[mus])
    compute_muscle_fiber_length = ComputeMuscleFiberLengthRigidTendon()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)
    np.testing.assert_almost_equal(
        model.function_to_dm(
            partial(
                compute_muscle_fiber_length,
                muscle=mus,
                model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
                biorbd_muscle=model.model.muscle(0),
            ),
            activation=activation,
            q=q,
            qdot=qdot,
        ),
        0.077,
    )


def test_compute_muscle_fiber_length_as_variable():
    mus = MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123)
    model = MuscleBiorbdModel(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_length = ComputeMuscleFiberLengthAsVariable(mx_symbolic=mx_symbolic)

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

    compute_muscle_fiber_length_default = ComputeMuscleFiberLengthAsVariable()
    length = compute_muscle_fiber_length_default(
        muscle=mus,
        model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
        biorbd_muscle=model.model.muscle(0),
        activation=activation,
        q=np.array([-0.2]),
        qdot=qdot,
    )
    assert isinstance(length, MX)
    assert length.name() == "muscle_fiber_length"

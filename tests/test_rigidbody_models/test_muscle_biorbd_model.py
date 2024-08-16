import re

from musculotendon_ocp import MuscleBiorbdModel, MuscleModelHillRigidTendon
import pytest

model_path = "musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod"


def test_muscle_biorbd_model_construction():
    with pytest.raises(ValueError, match="Muscle Wrong muscle was not found in the biorbd model"):
        MuscleBiorbdModel(
            model_path,
            muscles=[MuscleModelHillRigidTendon(name="Wrong muscle", maximal_force=500, optimal_length=0.1)],
        )

    # Can load a model with the right muscles
    model_all = MuscleBiorbdModel(
        model_path,
        muscles=[MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1)],
    )
    assert model_all.nb_muscles == 1

    # Can load a model with less muscles
    model_none = MuscleBiorbdModel(model_path, muscles=[])
    assert model_none.nb_muscles == 0


def test_muscle_biorbd_model_muscle_jacobian():
    model_none = MuscleBiorbdModel(model_path, muscles=[])

    # Do not allow to call muscle_length_jacobian directly
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "In the context of this project, the name 'muscle_length_jacobian' is confusing as it is the jacobian of "
            "the muscle-tendon-unit length (as opposed to the muscle-fiber-unit length)."
        ),
    ):
        model_none.muscle_length_jacobian(q=[])

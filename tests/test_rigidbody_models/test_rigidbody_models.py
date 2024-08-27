import os

from musculotendon_ocp import RigidbodyModels

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_rigidbody_models():
    assert len(RigidbodyModels) == 1

    with_muscles = RigidbodyModels.WithMuscles(model_path, muscles=[])
    assert type(with_muscles) == RigidbodyModels.WithMuscles.value

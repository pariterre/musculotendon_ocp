import pathlib

from musculotendon_ocp import RigidbodyModels


model_path = (
    pathlib.Path(__file__).parent.resolve()
    / "../../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod"
).as_posix()


def test_rigidbody_models():
    assert len(RigidbodyModels) == 1

    with_muscles = RigidbodyModels.WithMuscles(model_path, muscles=[])
    assert type(with_muscles) == RigidbodyModels.WithMuscles.value

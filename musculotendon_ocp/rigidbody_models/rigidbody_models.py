from enum import Enum

from .rigidbody_model_with_muscles import RigidbodyModelWithMuscles


class RigidbodyModels(Enum):
    WithMuscles = RigidbodyModelWithMuscles

    def __call__(self, *args, **kwargs) -> RigidbodyModelWithMuscles:
        return self.value(*args, **kwargs)

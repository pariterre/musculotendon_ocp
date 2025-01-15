from enum import Enum

from .muscle_hill_model_abstract import MuscleHillModelAbstract
from .muscle_hill_model_rigid_tendon import MuscleHillModelRigidTendon
from .muscle_hill_model_flexible_tendon import (
    MuscleHillModelFlexibleTendon,
    MuscleHillModelFlexibleTendonAlwaysPositive,
)


class MuscleHillModels(Enum):
    RigidTendon = MuscleHillModelRigidTendon
    FlexibleTendon = MuscleHillModelFlexibleTendon
    FlexibleTendonAlwaysPositive = MuscleHillModelFlexibleTendonAlwaysPositive

    def __call__(self, *args, **kwargs) -> MuscleHillModelAbstract:
        return self.value(*args, **kwargs)

    @staticmethod
    def deserialize(data: dict) -> MuscleHillModelAbstract:
        method = data["method"]
        for method_enum in MuscleHillModels:
            if method_enum.value.__name__ == method:
                return method_enum.value.deserialize(data)
        raise ValueError(f"Cannot deserialize {method} as MuscleHillModels")

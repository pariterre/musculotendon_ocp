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

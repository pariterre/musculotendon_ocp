from enum import Enum
from typing import Self

from casadi import MX, exp

from .muscle_hill_model_abstract import ComputeForceActive

"""
Implementations of the ComputeForceActive protocol
"""


class ComputeForceActiveHillType:
    def __init__(
        self,
        b11: float = 0.814483478343008,
        b21: float = 1.055033428970575,
        b31: float = 0.162384573599574,
        b41: float = 0.063303448465465,
        b12: float = 0.433004984392647,
        b22: float = 0.716775413397760,
        b32: float = -0.029947116970696,
        b42: float = 0.200356847296188,
        b13: float = 0.100,
        b23: float = 1.000,
        b33: float = 0.354,
        b43: float = 0.000,
    ):
        # TODO The default values may need to be more precise
        self.b11 = b11
        self.b21 = b21
        self.b31 = b31
        self.b41 = b41
        self.b12 = b12
        self.b22 = b22
        self.b32 = b32
        self.b42 = b42
        self.b13 = b13
        self.b23 = b23
        self.b33 = b33
        self.b43 = b43

    @property
    def copy(self) -> Self:
        return ComputeForceActiveHillType(
            b11=self.b11,
            b21=self.b21,
            b31=self.b31,
            b41=self.b41,
            b12=self.b12,
            b22=self.b22,
            b32=self.b32,
            b42=self.b42,
            b13=self.b13,
            b23=self.b23,
            b33=self.b33,
            b43=self.b43,
        )

    def serialize(self) -> dict:
        return {
            "method": type(self).__name__,
            "b11": self.b11,
            "b21": self.b21,
            "b31": self.b31,
            "b41": self.b41,
            "b12": self.b12,
            "b22": self.b22,
            "b32": self.b32,
            "b42": self.b42,
            "b13": self.b13,
            "b23": self.b23,
            "b33": self.b33,
            "b43": self.b43,
        }

    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != ComputeForceActiveHillType.__name__:
            raise ValueError(f"Cannot deserialize {data['method']} as ComputeForceActiveHillType")
        return ComputeForceActiveHillType(
            b11=data["b11"],
            b21=data["b21"],
            b31=data["b31"],
            b41=data["b41"],
            b12=data["b12"],
            b22=data["b22"],
            b32=data["b32"],
            b42=data["b42"],
            b13=data["b13"],
            b23=data["b23"],
            b33=data["b33"],
            b43=data["b43"],
        )

    def __call__(self, normalized_muscle_fiber_length: MX) -> MX:

        length = normalized_muscle_fiber_length  # alias so the next line is not too long
        return (
            self.b11 * exp((-0.5) * ((length - self.b21) ** 2) / ((self.b31 + self.b41 * length) ** 2))
            + self.b12 * exp((-0.5) * (length - self.b22) ** 2 / ((self.b32 + self.b42 * length) ** 2))
            + self.b13 * exp((-0.5) * (length - self.b23) ** 2 / ((self.b33 + self.b43 * length) ** 2))
        )


class ComputeForceActiveMethods(Enum):
    HillType = ComputeForceActiveHillType

    def __call__(self, *args, **kwargs) -> ComputeForceActive:
        return self.value(*args, **kwargs)

    @staticmethod
    def deserialize(data: dict) -> ComputeForceActive:
        method = data["method"]
        for method_enum in ComputeForceActiveMethods:
            if method_enum.value.__name__ == method:
                return method_enum.value.deserialize(data)
        raise ValueError(f"Cannot deserialize {method} as ComputeForceActiveMethods")
